package main

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"log"
	"os"
	"sync"
	"time"
)

var (
	globalFrameRate int
	frameInterval   time.Duration
)

const (
	RetryTimeSecond uint = 3
)

type RTSPManager struct {
	cameras   map[string]*CameraStream
	mutex     sync.RWMutex
	outputDir string
	proxyMgr  *FFmpegProxyManager
}

type CameraStream struct {
	ID          string
	URL         string
	Name        string
	isRunning   bool
	stopChannel chan struct{}
	lastFrame   []byte
	lastUpdate  time.Time
	mutex       sync.RWMutex
}

func NewRTSPManager() *RTSPManager {
	outputDir := "output"
	os.MkdirAll(outputDir, 0755)

	return &RTSPManager{
		cameras:   make(map[string]*CameraStream),
		outputDir: outputDir,
		proxyMgr:  NewFFmpegProxyManager(DefaultFFmpegProxyConfig()),
	}
}

func (m *RTSPManager) StartCamera(camera *CameraConfig) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	if stream, exists := m.cameras[camera.ID]; exists && stream.isRunning {
		return fmt.Errorf("camera %s is already running", camera.ID)
	}

	Info(fmt.Sprintf("starting FFmpeg proxy for camera: %s", camera.Name))

	stream := &CameraStream{
		ID:          camera.ID,
		URL:         camera.RTSPUrl,
		Name:        camera.Name,
		stopChannel: make(chan struct{}),
		lastUpdate:  time.Now(),
	}

	go m.captureFramesWithProxy(stream)
	m.cameras[camera.ID] = stream

	return nil
}

// captureFramesWithProxy captures frames using FFmpeg proxy
func (m *RTSPManager) captureFramesWithProxy(stream *CameraStream) {
	stream.mutex.Lock()
	stream.isRunning = true
	stream.mutex.Unlock()

	defer func() {
		stream.mutex.Lock()
		stream.isRunning = false
		stream.mutex.Unlock()
		// Stop the proxy for this camera
		m.proxyMgr.StopProxy(stream.ID)
		log.Printf("ffmpeg proxy stopped for camera: %s", stream.ID)
	}()

	for {
		select {
		case <-stream.stopChannel:
			Info(fmt.Sprintf("stopping frame capture for camera: %s", stream.Name))
			return
		default:
			if err := m.connectAndCaptureWithProxy(stream); err != nil {
				Warn(fmt.Sprintf("proxy connection lost for camera %s: %v, retrying in %ds", stream.ID, err, RetryTimeSecond))

				select {
				case <-stream.stopChannel:
					return
				case <-time.After(time.Duration(RetryTimeSecond) * time.Second):
				}
			}
		}
	}
}

// connectAndCaptureWithProxy connects to FFmpeg proxy and captures frames
func (m *RTSPManager) connectAndCaptureWithProxy(stream *CameraStream) error {
	Info(fmt.Sprintf("starting FFmpeg proxy for camera: %s", stream.Name))

	// Start FFmpeg proxy for this camera
	proxy, err := m.proxyMgr.StartProxy(stream.ID, stream.URL)
	if err != nil {
		return fmt.Errorf("failed to start FFmpeg proxy for camera %s: %v", stream.ID, err)
	}

	Info(fmt.Sprintf("successfully started FFmpeg proxy for camera: %s", stream.Name))

	// Frame rate control
	lastFrameTime := time.Now()

	for {
		select {
		case <-stream.stopChannel:
			Info(fmt.Sprintf("stopping frame capture for camera: %s", stream.Name))
			return nil
		default:
			// Get frame from proxy with timeout
			rawFrame, err := proxy.GetFrameTimeout(5 * time.Second)
			if err != nil {
				return fmt.Errorf("failed to get frame from proxy: %v", err)
			}

			// Frame rate control: check if we should process this frame
			currentTime := time.Now()
			if currentTime.Sub(lastFrameTime) < frameInterval {
				// Skip this frame, continue to next
				continue
			}
			lastFrameTime = currentTime

			// Convert raw frame to JPEG for compatibility with existing code
			jpegData, err := m.rawFrameToJPEG(rawFrame)
			if err != nil {
				Warn(fmt.Sprintf("failed to convert frame to JPEG for camera %s: %v", stream.ID, err))
				continue
			}

			// Process frame data
			m.processFrame(stream, jpegData)
		}
	}
}

// rawFrameToJPEG converts raw RGB24 frame to JPEG bytes
func (m *RTSPManager) rawFrameToJPEG(frame *RawFrame) ([]byte, error) {
	// Create RGBA image from raw RGB24 data
	img := image.NewRGBA(image.Rect(0, 0, frame.Width, frame.Height))

	dataIndex := 0
	for y := 0; y < frame.Height; y++ {
		for x := 0; x < frame.Width; x++ {
			if dataIndex+2 >= len(frame.Data) {
				return nil, fmt.Errorf("insufficient frame data: expected %d bytes, got %d", frame.Width*frame.Height*3, len(frame.Data))
			}

			r := frame.Data[dataIndex]
			g := frame.Data[dataIndex+1]
			b := frame.Data[dataIndex+2]

			img.SetRGBA(x, y, color.RGBA{R: r, G: g, B: b, A: 255})
			dataIndex += 3
		}
	}

	// Encode to JPEG
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: 90}); err != nil {
		return nil, fmt.Errorf("failed to encode JPEG: %v", err)
	}

	return buf.Bytes(), nil
}

func (m *RTSPManager) processFrame(stream *CameraStream, frameData []byte) {
	cameraConfig, exists := safeGetCamera(stream.ID)
	if !exists || cameraConfig == nil {
		Warn(fmt.Sprintf("camera config not available for stream %s (%s), skipping frame processing", stream.ID, stream.Name))
		return
	}

	stream.mutex.Lock()
	stream.lastFrame = make([]byte, len(frameData))
	copy(stream.lastFrame, frameData)
	stream.lastUpdate = time.Now()
	stream.mutex.Unlock()

	if len(cameraConfig.InferenceServerBindings) > 0 {
		// Launch async inference processing - no waiting, no blocking
		ProcessFrameWithAsyncInference(frameData, cameraConfig, m.outputDir)
	}
}

func (m *RTSPManager) StopCamera(cameraID string) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	stream, exists := m.cameras[cameraID]
	if !exists {
		return fmt.Errorf("camera %s not found", cameraID)
	}

	if !stream.isRunning {
		return fmt.Errorf("camera %s is not running", cameraID)
	}

	close(stream.stopChannel)
	delete(m.cameras, cameraID)

	return nil
}

func (m *RTSPManager) StopAll() {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	// Close all stop channels to signal capture goroutines to exit
	// The goroutines will handle stopping their individual FFmpeg proxies in defer
	for _, stream := range m.cameras {
		if stream.isRunning {
			close(stream.stopChannel)
		}
	}

	// Also stop all proxies directly as a safety measure
	m.proxyMgr.StopAll()

	m.cameras = make(map[string]*CameraStream)
}

func (m *RTSPManager) GetCameraStatus(cameraID string) (bool, time.Time) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	if stream, exists := m.cameras[cameraID]; exists {
		stream.mutex.RLock()
		defer stream.mutex.RUnlock()
		return stream.isRunning, stream.lastUpdate
	}
	return false, time.Time{}
}

