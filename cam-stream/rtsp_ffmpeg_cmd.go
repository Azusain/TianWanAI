package main

import (
	"bytes"
	"fmt"
	"image/jpeg"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"time"
)

// FFmpegCmdRTSPManager manages RTSP streams using FFmpeg command
type FFmpegCmdRTSPManager struct {
	cameras    map[string]*FFmpegCameraStream
	mutex      sync.RWMutex
	outputDir  string
	stopSignal chan struct{}
}

// FFmpegCameraStream represents an active RTSP camera stream using FFmpeg
type FFmpegCameraStream struct {
	ID          string
	URL         string
	Name        string
	isRunning   bool
	stopChannel chan struct{}
	lastFrame   []byte
	lastUpdate  time.Time
	mutex       sync.RWMutex
	cmd         *exec.Cmd
}

// NewFFmpegCmdRTSPManager creates a new FFmpeg command-based RTSP manager
func NewFFmpegCmdRTSPManager() *FFmpegCmdRTSPManager {
	outputDir := "output"
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		log.Printf("Warning: Failed to create output directory: %v", err)
	}

	return &FFmpegCmdRTSPManager{
		cameras:    make(map[string]*FFmpegCameraStream),
		outputDir:  outputDir,
		stopSignal: make(chan struct{}),
	}
}

// StartCamera starts processing an RTSP stream for the given camera
func (m *FFmpegCmdRTSPManager) StartCamera(camera *CameraConfig) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	// Check if camera is already running
	if stream, exists := m.cameras[camera.ID]; exists && stream.isRunning {
		return fmt.Errorf("camera %s is already running", camera.ID)
	}

	log.Printf("Starting FFmpeg RTSP stream for camera: %s (%s)", camera.Name, camera.ID)
	log.Printf("RTSP URL: %s", camera.RTSPUrl)

	// Create camera stream
	stream := &FFmpegCameraStream{
		ID:          camera.ID,
		URL:         camera.RTSPUrl,
		Name:        camera.Name,
		stopChannel: make(chan struct{}),
		lastUpdate:  time.Now(),
	}

	// Start FFmpeg process to capture frames
	go m.captureFrames(stream)

	m.cameras[camera.ID] = stream
	log.Printf("Successfully started FFmpeg RTSP stream for camera: %s", camera.ID)

	return nil
}

// captureFrames runs FFmpeg to continuously capture frames from RTSP stream
func (m *FFmpegCmdRTSPManager) captureFrames(stream *FFmpegCameraStream) {
	stream.mutex.Lock()
	stream.isRunning = true
	stream.mutex.Unlock()

	defer func() {
		stream.mutex.Lock()
		stream.isRunning = false
		stream.mutex.Unlock()
		if stream.cmd != nil && stream.cmd.Process != nil {
			stream.cmd.Process.Kill()
		}
		log.Printf("FFmpeg capture stopped for camera: %s", stream.ID)
	}()

	// Output path for frames
	outputPath := filepath.Join(m.outputDir, fmt.Sprintf("camera_%s_latest.jpg", stream.ID))

	for {
		select {
		case <-stream.stopChannel:
			return
		default:
			// Get frame rate from global config
			frameRate := fmt.Sprintf("%d", globalConfig.FrameRate)
			
			// FFmpeg command to capture frames continuously
			// -rtsp_transport tcp: Use TCP for more reliable connection
			// -i: Input RTSP URL
			// -r: Output frame rate from config
			// -update 1: Continuously update the same file
			// -q:v 2: JPEG quality (2 is high quality)
			stream.cmd = exec.Command("ffmpeg",
				"-rtsp_transport", "tcp",
				"-i", stream.URL,
				"-r", frameRate, // Frame rate from config
				"-update", "1", // Update same file
				"-q:v", "2", // High quality JPEG
				"-y", // Overwrite output
				outputPath,
			)

			// Start FFmpeg
			log.Printf("Starting FFmpeg for camera %s", stream.ID)
			stderr := &bytes.Buffer{}
			stream.cmd.Stderr = stderr

			if err := stream.cmd.Start(); err != nil {
				log.Printf("Failed to start FFmpeg for camera %s: %v", stream.ID, err)
				time.Sleep(5 * time.Second) // Wait before retry
				continue
			}

			// Monitor the process
			go m.monitorFrameFile(stream, outputPath)

			// Wait for process to complete or be killed
			err := stream.cmd.Wait()
			if err != nil {
				select {
				case <-stream.stopChannel:
					// Normal stop
					return
				default:
					// Error occurred
					log.Printf("FFmpeg error for camera %s: %v", stream.ID, err)
					log.Printf("FFmpeg stderr: %s", stderr.String())
					time.Sleep(5 * time.Second) // Wait before retry
				}
			}
		}
	}
}

// monitorFrameFile monitors the output file and updates lastFrame
func (m *FFmpegCmdRTSPManager) monitorFrameFile(stream *FFmpegCameraStream, outputPath string) {
	// Calculate monitoring interval based on frame rate (with minimum of 33ms for 30+ FPS)
	monitorInterval := time.Duration(1000/globalConfig.FrameRate) * time.Millisecond
	if monitorInterval < 33*time.Millisecond {
		monitorInterval = 33 * time.Millisecond // Cap at ~30 FPS monitoring
	}
	ticker := time.NewTicker(monitorInterval)
	defer ticker.Stop()

	// Camera config will be reloaded on each frame to ensure latest settings

	for {
		select {
		case <-stream.stopChannel:
			return
		case <-ticker.C:
			// Read the latest frame from disk
			frameData, err := os.ReadFile(outputPath)
			if err != nil {
				continue // File might not exist yet
			}

			// Validate it's a valid JPEG
			_, err = jpeg.Decode(bytes.NewReader(frameData))
			if err != nil {
				continue // Invalid JPEG, skip
			}

			// Get latest camera config for inference (reload to get updated thresholds)
			var cameraConfig *CameraConfig
			for _, cam := range dataStore.Cameras {
				if cam.ID == stream.ID {
					cameraConfig = cam
					break
				}
			}

			// Process frame with inference if configured
			processedFrame := frameData
			var modelResults map[string]*ModelResult
			if cameraConfig != nil && (len(cameraConfig.InferenceServerBindings) > 0 || len(cameraConfig.InferenceServers) > 0 || cameraConfig.ServerUrl != "") {
				var err error
				processedFrame, modelResults, err = ProcessFrameWithInference(frameData, cameraConfig)
				if err != nil {
					log.Printf("Warning: Failed to process frame with inference for camera %s: %v", stream.ID, err)
					processedFrame = frameData // Use original frame on error
				}
			} else {
				// No inference, just add timestamp overlay
				var detections []Detection // Empty detections
				processedFrame, err = DrawDetections(frameData, detections, stream.Name)
				if err != nil {
					processedFrame = frameData // Use original on error
				}
			}
			
			// Save results by model type (new architecture)
			if modelResults != nil {
				m.saveResultsByModel(stream.ID, stream.Name, frameData, processedFrame, modelResults)
			}
			
			// Always update global latest for web interface
			globalLatest := filepath.Join(m.outputDir, "latest_processed.jpg")
			os.WriteFile(globalLatest, processedFrame, 0644)

			// Update stream with processed frame
			stream.mutex.Lock()
			stream.lastFrame = processedFrame
			stream.lastUpdate = time.Now()
			stream.mutex.Unlock()

			// Frame processing completed silently
		}
	}
}

// StopCamera stops the RTSP stream for the given camera ID
func (m *FFmpegCmdRTSPManager) StopCamera(cameraID string) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	stream, exists := m.cameras[cameraID]
	if !exists {
		return fmt.Errorf("camera %s not found", cameraID)
	}

	if !stream.isRunning {
		return fmt.Errorf("camera %s is not running", cameraID)
	}

	log.Printf("Stopping FFmpeg RTSP stream for camera: %s", cameraID)

	// Signal the goroutine to stop
	close(stream.stopChannel)

	// Kill the FFmpeg process
	if stream.cmd != nil && stream.cmd.Process != nil {
		stream.cmd.Process.Kill()
	}

	// Remove from active cameras
	delete(m.cameras, cameraID)

	log.Printf("Successfully stopped FFmpeg RTSP stream for camera: %s", cameraID)
	return nil
}

// StopAll stops all active RTSP streams
func (m *FFmpegCmdRTSPManager) StopAll() {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	log.Printf("Stopping all FFmpeg RTSP streams...")

	for cameraID := range m.cameras {
		stream := m.cameras[cameraID]
		if stream.isRunning {
			close(stream.stopChannel)
			if stream.cmd != nil && stream.cmd.Process != nil {
				stream.cmd.Process.Kill()
			}
		}
	}

	m.cameras = make(map[string]*FFmpegCameraStream)
	close(m.stopSignal)

	log.Printf("All FFmpeg RTSP streams stopped")
}

// GetCameraStatus returns the status of a specific camera
func (m *FFmpegCmdRTSPManager) GetCameraStatus(cameraID string) (bool, time.Time) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	if stream, exists := m.cameras[cameraID]; exists {
		stream.mutex.RLock()
		defer stream.mutex.RUnlock()
		return stream.isRunning, stream.lastUpdate
	}

	return false, time.Time{}
}

// GetLatestFrame returns the latest captured frame for a camera
func (m *FFmpegCmdRTSPManager) GetLatestFrame(cameraID string) ([]byte, time.Time, error) {
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

	// Return a copy of the frame data
	frameCopy := make([]byte, len(stream.lastFrame))
	copy(frameCopy, stream.lastFrame)

	return frameCopy, stream.lastUpdate, nil
}

// saveResultsByModel saves detection results organized by inference server name
func (m *FFmpegCmdRTSPManager) saveResultsByModel(cameraID, cameraName string, originalFrame, processedFrame []byte, modelResults map[string]*ModelResult) {
	timestamp := time.Now().Format("20060102_150405")
	
	
	for modelType, result := range modelResults {
		if !result.ShouldSave {
			continue
		}
		
		// Get server name for directory creation
		serverName := result.ServerID
		if server, exists := dataStore.InferenceServers[result.ServerID]; exists {
			serverName = server.Name
		}
		
		// Create inference server-specific directory
		serverDir := filepath.Join(m.outputDir, serverName)
		if err := os.MkdirAll(serverDir, 0755); err != nil {
			log.Printf("Warning: Failed to create server directory for %s: %v", serverName, err)
			continue
		}
		
		// Save processed frame with detections
		filename := fmt.Sprintf("%s_%s_%s.jpg", timestamp, cameraID, modelType)
		processedPath := filepath.Join(serverDir, filename)
		
		if err := os.WriteFile(processedPath, processedFrame, 0644); err != nil {
			log.Printf("Warning: Failed to save processed frame for server %s: %v", serverName, err)
			continue
		}
		
		log.Printf("Saved detection frame - Camera: %s, Server: %s (%s), Detections: %d, Path: %s", 
			cameraName, serverName, modelType, len(result.Detections), processedPath)
		
		// Send alerts for each detection if global alert system is enabled
		if len(result.Detections) > 0 {
			m.sendDetectionAlerts(cameraName, originalFrame, modelType, result.Detections)
		}
	}
}

// sendDetectionAlerts sends alert for each detection using global alert configuration
func (m *FFmpegCmdRTSPManager) sendDetectionAlerts(cameraName string, imageData []byte, modelType string, detections []Detection) {
	for _, detection := range detections {
		// Convert pixel coordinates to normalized coordinates (0-1) as required by API
		// First get image dimensions
		imgReader := bytes.NewReader(imageData)
		img, err := jpeg.DecodeConfig(imgReader)
		if err != nil {
			log.Printf("Warning: Failed to decode image config for alert: %v", err)
			continue
		}
		
		// Convert to normalized coordinates
		x1 := float64(detection.X1) / float64(img.Width)
		y1 := float64(detection.Y1) / float64(img.Height)
		x2 := float64(detection.X2) / float64(img.Width)
		y2 := float64(detection.Y2) / float64(img.Height)
		
		// Send alert using global configuration and camera name as KKS
		if err := SendAlertIfConfigured(imageData, modelType, cameraName, detection.Confidence, x1, y1, x2, y2); err != nil {
			log.Printf("Warning: Failed to send alert for camera %s: %v", cameraName, err)
		}
	}
}
