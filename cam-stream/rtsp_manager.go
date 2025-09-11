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

	AsyncInfo(fmt.Sprintf("starting FFmpeg proxy for camera: %s", camera.Name))

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
			AsyncInfo(fmt.Sprintf("stopping frame capture for camera: %s", stream.Name))
			return
		default:
			if err := m.connectAndCaptureWithProxy(stream); err != nil {
				AsyncWarn(fmt.Sprintf("proxy connection lost for camera %s: %v, retrying in %ds", stream.ID, err, RetryTimeSecond))

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
	AsyncInfo(fmt.Sprintf("starting FFmpeg proxy for camera: %s", stream.Name))

	// Start FFmpeg proxy for this camera
	proxy, err := m.proxyMgr.StartProxy(stream.ID, stream.URL)
	if err != nil {
		return fmt.Errorf("failed to start FFmpeg proxy for camera %s: %v", stream.ID, err)
	}

	AsyncInfo(fmt.Sprintf("successfully started FFmpeg proxy for camera: %s", stream.Name))

	// Frame rate control
	lastFrameTime := time.Now()

	for {
		select {
		case <-stream.stopChannel:
			AsyncInfo(fmt.Sprintf("stopping frame capture for camera: %s", stream.Name))
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
				AsyncWarn(fmt.Sprintf("failed to convert frame to JPEG for camera %s: %v", stream.ID, err))
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
	// 线程安全地获取摄像头配置
	cameraConfig := getCameraConfig(stream.ID)

	// 更新流的最新帧
	stream.mutex.Lock()
	stream.lastFrame = make([]byte, len(frameData))
	copy(stream.lastFrame, frameData)
	stream.lastUpdate = time.Now()
	stream.mutex.Unlock()

	// 如果配置了推理服务器，实时处理推理
	if cameraConfig != nil && len(cameraConfig.InferenceServerBindings) > 0 {
		modelResults, err := ProcessFrameWithMultipleInference(frameData, cameraConfig, m)
		if err != nil {
			AsyncWarn(fmt.Sprintf("inference failed for camera %s: %v", stream.ID, err))
		} else if modelResults != nil {
			// 异步保存结果，避免阻塞主流程
			go m.saveResultsByModel(stream.Name, modelResults)
		}
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

func (m *RTSPManager) GetLatestFrame(cameraID string) ([]byte, time.Time, error) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	stream, exists := m.cameras[cameraID]
	if !exists {
		return nil, time.Time{}, fmt.Errorf("camera %s not found", cameraID)
	}

	stream.mutex.RLock()
	defer stream.mutex.RUnlock()

	if stream.lastFrame == nil {
		return nil, time.Time{}, fmt.Errorf("no frame available for camera %s", cameraID)
	}

	frameCopy := make([]byte, len(stream.lastFrame))
	copy(frameCopy, stream.lastFrame)
	return frameCopy, stream.lastUpdate, nil
}

// saveResultsByModel 保存检测结果到文件系统
func (m *RTSPManager) saveResultsByModel(cameraName string, modelResults map[string]*ModelResult) {
	// 保存有检测结果的图片到按服务器名称组织的目录中
	for modelType, result := range modelResults {
		if len(result.Detections) == 0 {
			continue // 跳过没有检测结果的模型
		}

		// For fall detection, ensure each ModelResult contains exactly one detection
		if modelType == "fall" && len(result.Detections) != 1 {
			AsyncWarn(fmt.Sprintf("fall detection ModelResult should contain exactly one detection, got %d detections, skipping", len(result.Detections)))
			continue
		}

		serverDir := fmt.Sprintf("%s/%s", m.outputDir, result.ServerID)
		if err := os.MkdirAll(serverDir, 0755); err != nil {
			AsyncWarn(fmt.Sprintf("failed to create directory for server %s: %v", result.ServerID, err))
			continue
		}

		timestamp := time.Now().Format("20060102_150405")
		filename := fmt.Sprintf("%s_%s_detection.jpg", timestamp, modelType)
		filePath := fmt.Sprintf("%s/%s", serverDir, filename)

		if err := os.WriteFile(filePath, result.DisplayResultImage, 0644); err != nil {
			AsyncWarn(fmt.Sprintf("failed to save detection image for model %s: %v", modelType, err))
			continue
		}

		AsyncInfo(fmt.Sprintf("saved detection image for camera %s, model %s to %s (detections: %d)", cameraName, modelType, filePath, len(result.Detections)))

		// Save original image to debug directory if DEBUG mode is enabled
		if globalDebugMode && result.OriginalImage != nil {
			// Create debug server directory
			debugServerDir := fmt.Sprintf("%s/%s", DebugDir, result.ServerID)
			if err := os.MkdirAll(debugServerDir, 0755); err != nil {
				AsyncWarn(fmt.Sprintf("failed to create debug directory for server %s: %v", result.ServerID, err))
			} else {
				// Save original image with same filename as detection image
				debugFilePath := fmt.Sprintf("%s/%s", debugServerDir, filename)
				if err := os.WriteFile(debugFilePath, result.OriginalImage, 0644); err != nil {
					AsyncWarn(fmt.Sprintf("failed to save debug original image: %v", err))
				} else {
					AsyncInfo(fmt.Sprintf("saved DEBUG original image to %s", debugFilePath))
				}
			}
		}

		sendDetectionAlerts(result.DisplayResultImage, result.Detections, cameraName, modelType)
	}
}

func sendDetectionAlerts(imageData []byte, detections []Detection, cameraName, modelType string) {
	// get the real size of the image.
	img, err := jpeg.DecodeConfig(bytes.NewReader(imageData))
	if err != nil {
		AsyncWarn(fmt.Sprintf("failed to decode image config for alerts: %v", err))
		return
	}

	for _, detection := range detections {
		// normalization.
		x1 := float64(detection.X1) / float64(img.Width)
		y1 := float64(detection.Y1) / float64(img.Height)
		x2 := float64(detection.X2) / float64(img.Width)
		y2 := float64(detection.Y2) / float64(img.Height)

		if err := SendAlertIfConfigured(imageData, modelType, cameraName, detection.Confidence, x1, y1, x2, y2); err != nil {
			AsyncWarn(fmt.Sprintf("failed to send alert for detection %s: %v", detection.Class, err))
		} else {
			AsyncInfo(fmt.Sprintf("sent alert for detection %s (confidence: %.3f) from camera %s", detection.Class, detection.Confidence, cameraName))
		}
	}
}
