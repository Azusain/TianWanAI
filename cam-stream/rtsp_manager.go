package main

import (
	"bytes"
	"fmt"
	"image"
	"image/jpeg"
	"log"
	"os"
	"sync"
	"time"

	"gocv.io/x/gocv"
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
}

type CameraStream struct {
	ID           string
	URL          string
	Name         string
	isRunning    bool
	stopChannel  chan struct{}
	lastFrame    []byte
	lastUpdate   time.Time
	mutex        sync.RWMutex
	videoCapture *gocv.VideoCapture
}

func NewRTSPManager() *RTSPManager {
	outputDir := "output"
	os.MkdirAll(outputDir, 0755)

	return &RTSPManager{
		cameras:   make(map[string]*CameraStream),
		outputDir: outputDir,
	}
}

func (m *RTSPManager) StartCamera(camera *CameraConfig) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	if stream, exists := m.cameras[camera.ID]; exists && stream.isRunning {
		return fmt.Errorf("camera %s is already running", camera.ID)
	}

	AsyncInfo(fmt.Sprintf("starting RTSP stream for camera: %s", camera.Name))

	stream := &CameraStream{
		ID:          camera.ID,
		URL:         camera.RTSPUrl,
		Name:        camera.Name,
		stopChannel: make(chan struct{}),
		lastUpdate:  time.Now(),
	}

	go m.captureFrames(stream)
	m.cameras[camera.ID] = stream

	return nil
}

func (m *RTSPManager) captureFrames(stream *CameraStream) {
	stream.mutex.Lock()
	stream.isRunning = true
	stream.mutex.Unlock()

	// 抑制 FFmpeg 底层日志输出
	os.Setenv("AV_LOG_LEVEL", "error")

	defer func() {
		stream.mutex.Lock()
		stream.isRunning = false
		stream.mutex.Unlock()
		log.Printf("rtsp capture stopped for camera: %s", stream.ID)
	}()

	for {
		select {
		case <-stream.stopChannel:
			AsyncInfo(fmt.Sprintf("stopping frame capture for camera: %s", stream.Name))
			return
		default:
			if err := m.connectAndCapture(stream); err != nil {
				AsyncWarn(fmt.Sprintf("connection lost for camera %s: %v, retrying in %ds", stream.ID, err, RetryTimeSecond))

				select {
				case <-stream.stopChannel:
					return
				case <-time.After(time.Duration(RetryTimeSecond) * time.Second):
				}
			}
		}
	}
}

func (m *RTSPManager) connectAndCapture(stream *CameraStream) error {
	baseURL := stream.URL
	// try TCP first.
	tcpURL := baseURL + "?tcp=1"
	AsyncInfo(fmt.Sprintf("trying TCP connection: %s", tcpURL))
	videoCapture, err := gocv.OpenVideoCapture(tcpURL)
	if err != nil || !videoCapture.IsOpened() {
		// fall back to UDP.
		if videoCapture != nil {
			videoCapture.Close()
		}
		AsyncWarn(fmt.Sprintf("TCP connection failed for %s, falling back to UDP: %v", stream.Name, err))

		udpURL := baseURL + "?udp=1"
		AsyncInfo(fmt.Sprintf("trying UDP connection: %s", udpURL))
		videoCapture, err = gocv.OpenVideoCapture(udpURL)
		if err != nil {
			return fmt.Errorf("failed to open RTSP stream with both TCP and UDP: %v", err)
		}
		if !videoCapture.IsOpened() {
			return fmt.Errorf("failed to open video capture with both TCP and UDP")
		}
		AsyncInfo(fmt.Sprintf("successfully connected via UDP for camera: %s", stream.Name))
	} else {
		AsyncInfo(fmt.Sprintf("successfully connected via TCP for camera: %s", stream.Name))
	}

	defer videoCapture.Close()
	stream.videoCapture = videoCapture

	videoCapture.Set(gocv.VideoCaptureBufferSize, 1)

	AsyncInfo(fmt.Sprintf("successfully connected to RTSP stream for camera: %s", stream.Name))

	img := gocv.NewMat()
	defer img.Close()

	continuousReadFailures := 0
	maxReadFailures := 15
	continuousEncodeFailures := 0
	maxEncodeFailures := 5

	// 帧率控制
	lastFrameTime := time.Now()

	for {
		select {
		case <-stream.stopChannel:
			AsyncInfo(fmt.Sprintf("stopping frame capture for camera: %s", stream.Name))
			return nil
		default:
			// 读取帧
			if !videoCapture.Read(&img) || img.Empty() {
				continuousReadFailures++
				if continuousReadFailures >= maxReadFailures {
					return fmt.Errorf("too many continuous read failures (%d), reconnecting", continuousReadFailures)
				}
				time.Sleep(50 * time.Millisecond)
				continue
			}

			// 读取成功，重置读取失败计数
			continuousReadFailures = 0

			// 将 Mat 转换为 JPEG 字节数组
			jpegBytes, err := gocv.IMEncode(gocv.JPEGFileExt, img)
			if err != nil {
				continuousEncodeFailures++
				if continuousEncodeFailures >= maxEncodeFailures {
					return fmt.Errorf("too many continuous encode failures (%d), likely corrupted frames, reconnecting", continuousEncodeFailures)
				}
				AsyncDebug(fmt.Sprintf("frame encode failed for camera %s (failure %d/%d): %v", stream.ID, continuousEncodeFailures, maxEncodeFailures, err))
				continue
			}

			// 编码成功，重置编码失败计数
			continuousEncodeFailures = 0

			// 帧率控制：检查是否应该处理这一帧
			currentTime := time.Now()
			if currentTime.Sub(lastFrameTime) < frameInterval {
				// 跳过这一帧，继续读取下一帧
				continue
			}
			lastFrameTime = currentTime

			// 获取字节数据并创建副本
			frameData := jpegBytes.GetBytes()
			frameDataCopy := make([]byte, len(frameData))
			copy(frameDataCopy, frameData)

			// 清理 JPEG 数据
			jpegBytes.Close()

			// 处理帧数据
			m.processFrame(stream, frameDataCopy)
		}
	}
}

func (m *RTSPManager) imageToJPEG(img image.Image) ([]byte, error) {
	var buf bytes.Buffer
	err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: 90})
	return buf.Bytes(), err
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

	for _, stream := range m.cameras {
		if stream.isRunning {
			close(stream.stopChannel)
		}
	}
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

		// 创建以服务器ID命名的目录
		serverDir := fmt.Sprintf("%s/%s", m.outputDir, result.ServerID)
		if err := os.MkdirAll(serverDir, 0755); err != nil {
			AsyncWarn(fmt.Sprintf("failed to create directory for server %s: %v", result.ServerID, err))
			continue
		}

		// 生成时间戳文件名
		timestamp := time.Now().Format("20060102_150405")
		filename := fmt.Sprintf("%s_%s_detection.jpg", timestamp, modelType)
		filePath := fmt.Sprintf("%s/%s", serverDir, filename)

		// 保存带有检测结果的图像
		if err := os.WriteFile(filePath, result.DisplayResultImage, 0644); err != nil {
			AsyncWarn(fmt.Sprintf("failed to save detection image for model %s: %v", modelType, err))
			continue
		}

		AsyncInfo(fmt.Sprintf("saved detection image for camera %s, model %s to %s (detections: %d)", cameraName, modelType, filePath, len(result.Detections)))

		// 发送检测告警（如果配置了）
		sendDetectionAlerts(result.DisplayResultImage, result.Detections, cameraName, modelType)
	}
}

// sendDetectionAlerts 发送检测告警到管理平台
func sendDetectionAlerts(imageData []byte, detections []Detection, cameraName, modelType string) {
	// 获取实际图像尺寸
	img, err := jpeg.DecodeConfig(bytes.NewReader(imageData))
	if err != nil {
		AsyncWarn(fmt.Sprintf("failed to decode image config for alerts: %v", err))
		return
	}

	for _, detection := range detections {
		// 将像素坐标转换为归一化坐标（0-1范围）使用实际图像尺寸
		x1 := float64(detection.X1) / float64(img.Width)
		y1 := float64(detection.Y1) / float64(img.Height)
		x2 := float64(detection.X2) / float64(img.Width)
		y2 := float64(detection.Y2) / float64(img.Height)

		// 发送告警
		if err := SendAlertIfConfigured(imageData, modelType, cameraName, detection.Confidence, x1, y1, x2, y2); err != nil {
			AsyncWarn(fmt.Sprintf("failed to send alert for detection %s: %v", detection.Class, err))
		} else {
			AsyncInfo(fmt.Sprintf("sent alert for detection %s (confidence: %.3f) from camera %s", detection.Class, detection.Confidence, cameraName))
		}
	}
}
