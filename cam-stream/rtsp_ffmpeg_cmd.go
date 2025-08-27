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
			// FFmpeg command to capture frames continuously
			// -rtsp_transport tcp: Use TCP for more reliable connection
			// -i: Input RTSP URL
			// -r 2: Output 2 frames per second
			// -update 1: Continuously update the same file
			// -q:v 2: JPEG quality (2 is high quality)
			stream.cmd = exec.Command("ffmpeg",
				"-rtsp_transport", "tcp",
				"-i", stream.URL,
				"-r", "2", // 2 FPS
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
	ticker := time.NewTicker(500 * time.Millisecond) // Check every 500ms
	defer ticker.Stop()

	// Get camera config for inference
	var cameraConfig *CameraConfig
	for _, cam := range dataStore.Cameras {
		if cam.ID == stream.ID {
			cameraConfig = cam
			break
		}
	}

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

			// Process frame with inference if configured
			processedFrame := frameData
			shouldSave := false
			if cameraConfig != nil && cameraConfig.ServerUrl != "" {
				var err error
				processedFrame, shouldSave, err = ProcessFrameWithInference(frameData, cameraConfig)
				if err != nil {
					log.Printf("Warning: Failed to process frame with inference for camera %s: %v", stream.ID, err)
					processedFrame = frameData // Use original frame on error
					shouldSave = false
				}
			} else {
				// No inference, just add timestamp overlay - never save
				var detections []Detection // Empty detections
				processedFrame, err = DrawDetections(frameData, detections, stream.Name)
				if err != nil {
					processedFrame = frameData // Use original on error
				}
				shouldSave = false // Never save when no inference
			}
			
			// Only save if we detected real objects (not test detections)
			if shouldSave {
				// Create camera-specific directory
				cameraDir := filepath.Join(m.outputDir, stream.ID)
				if err := os.MkdirAll(cameraDir, 0755); err != nil {
					log.Printf("Warning: Failed to create camera directory for %s: %v", stream.ID, err)
				} else {
					// Save processed frame with detections using timestamp
					timestamp := time.Now().Format("20060102_150405")
					processedPath := filepath.Join(cameraDir, fmt.Sprintf("%s.jpg", timestamp))
					if err := os.WriteFile(processedPath, processedFrame, 0644); err != nil {
						log.Printf("Warning: Failed to save processed frame for camera %s: %v", stream.ID, err)
					} else {
						log.Printf("Saved detection frame for camera %s: %s", stream.ID, processedPath)
					}
				}
			}
			
			// Always update global latest for web interface
			globalLatest := filepath.Join(m.outputDir, "latest_processed.jpg")
			os.WriteFile(globalLatest, processedFrame, 0644)

			// Update stream with processed frame
			stream.mutex.Lock()
			stream.lastFrame = processedFrame
			stream.lastUpdate = time.Now()
			stream.mutex.Unlock()

			// Log occasionally to show it's working
			if time.Now().Unix()%10 == 0 {
				log.Printf("Camera %s: captured frame, size: %d bytes", stream.ID, len(processedFrame))
			}
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
