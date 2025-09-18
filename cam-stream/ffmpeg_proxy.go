package main

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"os/exec"
	"sync"
	"time"
)

// FFmpegStreamProxy converts unstable RTSP to stable raw frame stream via pipe
type FFmpegStreamProxy struct {
	originalRTSP  string
	cmd           *exec.Cmd
	ctx           context.Context
	cancel        context.CancelFunc
	mutex         sync.RWMutex
	isRunning     bool
	frameChan     chan *RawFrame
	errorChan     chan error
	stdout        io.ReadCloser
	stderr        io.ReadCloser
	frameWidth    int
	frameHeight   int
	bytesPerFrame int
}

// RawFrame represents a raw video frame
type RawFrame struct {
	Data      []byte
	Width     int
	Height    int
	Timestamp time.Time
}

// FFmpegProxyConfig configuration for FFmpeg proxy
type FFmpegProxyConfig struct {
	Timeout      time.Duration // Connection timeout
	ReconnectMax int           // Max reconnect attempts
	FrameRate    int           // Frames per second
}

// DefaultFFmpegProxyConfig returns default configuration
func DefaultFFmpegProxyConfig() *FFmpegProxyConfig {
	return &FFmpegProxyConfig{
		Timeout:      10 * time.Second,
		ReconnectMax: 10,
		FrameRate:    10, // Default 10 FPS for stability
	}
}

// NewFFmpegStreamProxy creates a new FFmpeg stream proxy that auto-detects resolution
func NewFFmpegStreamProxy(rtspURL string) *FFmpegStreamProxy {
	ctx, cancel := context.WithCancel(context.Background())

	return &FFmpegStreamProxy{
		originalRTSP: rtspURL,
		ctx:          ctx,
		cancel:       cancel,
		frameChan:    make(chan *RawFrame, 5), // buffer 5 frames
		errorChan:    make(chan error, 5),
		// Resolution will be detected from first frame
		frameWidth:    0,
		frameHeight:   0,
		bytesPerFrame: 0,
	}
}

// detectResolution uses C++ stream detector to get actual resolution
func (fsp *FFmpegStreamProxy) detectResolution() error {
	width, height, err := GetResolution(fsp.originalRTSP)
	if err != nil {
		return fmt.Errorf("failed to detect stream resolution: %v", err)
	}

	fsp.frameWidth = width
	fsp.frameHeight = height
	fsp.bytesPerFrame = width * height * 3 // RGB24

	Info(fmt.Sprintf("detected resolution: %dx%d", width, height))
	return nil
}

// Start starts the FFmpeg proxy process
func (fsp *FFmpegStreamProxy) Start() error {
	fsp.mutex.Lock()
	defer fsp.mutex.Unlock()

	if fsp.isRunning {
		return fmt.Errorf("ffmpeg proxy already running")
	}

	// First detect stream resolution using ffmpeg
	if err := fsp.detectResolution(); err != nil {
		return fmt.Errorf("failed to detect resolution: %v", err)
	}

	// Build ffmpeg command args for raw frame output
	args := fsp.buildFFmpegArgs()

	fsp.cmd = exec.CommandContext(fsp.ctx, "ffmpeg", args...)

	// Get stdout and stderr pipes
	stdout, err := fsp.cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("failed to create stdout pipe: %v", err)
	}
	fsp.stdout = stdout

	stderr, err := fsp.cmd.StderrPipe()
	if err != nil {
		return fmt.Errorf("failed to create stderr pipe: %v", err)
	}
	fsp.stderr = stderr

	// Start ffmpeg process
	if err := fsp.cmd.Start(); err != nil {
		return fmt.Errorf("failed to start ffmpeg: %v", err)
	}

	fsp.isRunning = true

	// Start frame reading goroutine
	go fsp.readFrames()

	// Start error monitoring goroutine
	go fsp.monitorErrors()

	return nil
}

// buildFFmpegArgs builds ffmpeg command arguments for raw frame output
func (fsp *FFmpegStreamProxy) buildFFmpegArgs() []string {
	args := []string{
		"-v", "error", // minimal logging
		"-rtsp_transport", "tcp", // force TCP transport
		"-timeout", "10000000", // socket timeout in microseconds
		"-i", fsp.originalRTSP, // input RTSP URL
		"-f", "rawvideo", // output raw video
		"-pix_fmt", "rgb24", // RGB24 pixel format
		"-s", fmt.Sprintf("%dx%d", fsp.frameWidth, fsp.frameHeight), // use detected resolution
		"-r", "10", // frame rate
		"-", // output to stdout
	}

	return args
}

// readFrames reads raw frames from FFmpeg stdout
func (fsp *FFmpegStreamProxy) readFrames() {
	defer func() {
		fsp.mutex.Lock()
		fsp.isRunning = false
		fsp.mutex.Unlock()
		close(fsp.frameChan)
	}()

	buffer := make([]byte, fsp.bytesPerFrame)

	for {
		select {
		case <-fsp.ctx.Done():
			return
		default:
			// Read exactly one frame worth of data
			n, err := io.ReadFull(fsp.stdout, buffer)
			if err != nil {
				if err == io.EOF {
					return
				}
				fsp.errorChan <- fmt.Errorf("failed to read frame from ffmpeg: %v", err)
				return
			}

			if n != fsp.bytesPerFrame {
				fsp.errorChan <- fmt.Errorf("incomplete frame read: got %d bytes, expected %d", n, fsp.bytesPerFrame)
				continue
			}

			// Create frame copy
			frameData := make([]byte, fsp.bytesPerFrame)
			copy(frameData, buffer)

			frame := &RawFrame{
				Data:      frameData,
				Width:     fsp.frameWidth,
				Height:    fsp.frameHeight,
				Timestamp: time.Now(),
			}

			select {
			case fsp.frameChan <- frame:
			case <-fsp.ctx.Done():
				return
			default:
				// Drop frame if channel is full to maintain real-time performance
			}
		}
	}
}

// monitorErrors monitors FFmpeg stderr output
func (fsp *FFmpegStreamProxy) monitorErrors() {
	scanner := bufio.NewScanner(fsp.stderr)
	for scanner.Scan() {
		select {
		case <-fsp.ctx.Done():
			return
		case fsp.errorChan <- fmt.Errorf("ffmpeg error: %s", scanner.Text()):
		default:
			// Drop error if channel is full
		}
	}
}

// GetFrameTimeout gets the next raw frame with timeout
func (fsp *FFmpegStreamProxy) GetFrameTimeout(timeout time.Duration) (*RawFrame, error) {
	select {
	case frame, ok := <-fsp.frameChan:
		if !ok {
			return nil, fmt.Errorf("frame channel closed")
		}
		return frame, nil
	case err := <-fsp.errorChan:
		return nil, err
	case <-time.After(timeout):
		return nil, fmt.Errorf("frame timeout")
	case <-fsp.ctx.Done():
		return nil, fmt.Errorf("proxy stopped")
	}
}

// IsRunning checks if the proxy is running
func (fsp *FFmpegStreamProxy) IsRunning() bool {
	fsp.mutex.RLock()
	defer fsp.mutex.RUnlock()
	return fsp.isRunning
}

// Stop stops the FFmpeg proxy process
func (fsp *FFmpegStreamProxy) Stop() error {
	fsp.mutex.Lock()
	defer fsp.mutex.Unlock()

	if !fsp.isRunning {
		return nil
	}

	// Cancel context
	fsp.cancel()

	// Wait for process to end or force kill
	if fsp.cmd != nil && fsp.cmd.Process != nil {
		done := make(chan error, 1)
		go func() {
			done <- fsp.cmd.Wait()
		}()

		select {
		case <-time.After(5 * time.Second):
			// Force kill process
			fsp.cmd.Process.Kill()
			<-done
		case <-done:
		}
	}

	fsp.isRunning = false
	return nil
}
