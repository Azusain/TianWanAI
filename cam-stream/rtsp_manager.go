package main

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"os"
	"sync"
	"time"
)

var (
	globalFrameRate     int
	globalFrameInterval time.Duration
)

const (
	RetryTimeSecond        uint = 3
	DefaultGetFrameTimeout uint = 3
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
	mutex       sync.RWMutex
}

func NewRTSPManager() *RTSPManager {
	outputDir := OutputDir
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

	stream := &CameraStream{
		ID:          camera.ID,
		URL:         camera.RTSPUrl,
		Name:        camera.Name,
		stopChannel: make(chan struct{}),
	}

	Info(fmt.Sprintf("starting camera: %s", camera.Name))

	// start camera stream processing
	go func() {
		stream.mutex.Lock()
		stream.isRunning = true
		stream.mutex.Unlock()

		defer func() {
			stream.mutex.Lock()
			stream.isRunning = false
			stream.mutex.Unlock()
			m.proxyMgr.StopProxy(stream.ID)
			Info(fmt.Sprintf("camera stopped: %s", stream.Name))
		}()

		// retry loop
		for {
			select {

			case <-stream.stopChannel:
				return

			default:
				err := m.connectAndCaptureWithProxy(stream)
				if err == nil {
					continue
				}
				Warn(fmt.Sprintf("camera %s connection lost: %v, retrying in %ds",
					stream.ID, err, RetryTimeSecond))
				select {
				case <-stream.stopChannel:
					return
				case <-time.After(time.Duration(RetryTimeSecond) * time.Second):
				}

			}
		}
	}()

	m.cameras[camera.ID] = stream
	return nil
}

// connectAndCaptureWithProxy connects to FFmpeg proxy and captures frames
func (m *RTSPManager) connectAndCaptureWithProxy(stream *CameraStream) error {
	// start FFmpeg proxy
	proxy, err := m.proxyMgr.StartProxy(stream.ID, stream.URL)
	if err != nil {
		return fmt.Errorf("failed to start FFmpeg proxy: %v", err)
	}

	Info(fmt.Sprintf("FFmpeg proxy started for camera: %s", stream.Name))
	lastFrameTime := time.Now()

	for {
		select {

		case <-stream.stopChannel:
			return nil

		default:
			// get frame data
			rawFrame, err := proxy.GetFrameTimeout(time.Duration(DefaultGetFrameTimeout) * time.Second)
			if err != nil {
				return fmt.Errorf("failed to get frame: %v", err)
			}

			// frame rate control
			if time.Since(lastFrameTime) < globalFrameInterval {
				continue
			}
			lastFrameTime = time.Now()

			// TODO: why cant we just process raw frame?
			jpegData, err := m.rawFrameToJPEG(rawFrame)
			if err != nil {
				Warn(fmt.Sprintf("failed to convert frame for camera %s: %v", stream.ID, err))
				continue
			}

			// process a single frame.
			cameraConfig, exists := safeGetCamera(stream.ID)
			if !exists || cameraConfig == nil {
				Warn(fmt.Sprintf("camera config not available for stream %s (%s), skipping frame processing", stream.ID, stream.Name))
				continue
			}

			if len(cameraConfig.InferenceServerBindings) <= 0 {
				continue
			}

			ProcessFrameWithAsyncInference(jpegData, cameraConfig, m.outputDir)

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
