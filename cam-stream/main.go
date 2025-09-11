package main

import (
	"fmt"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"
)

const (
	ConfigFile = "configs/config.json"
	OutputDir  = "output"
)

// Global config variable
var globalConfig *Config

// initializeFrameRate initializes frame rate configuration from environment variable
func initializeFrameRate() {
	// Read frame rate from environment variable, default to 25
	frameRateStr := os.Getenv("FRAME_RATE")
	if frameRateStr == "" {
		// default 25 FPS
		globalFrameRate = 25
	} else {
		if fps, err := strconv.Atoi(frameRateStr); err != nil {
			AsyncWarn(fmt.Sprintf("invalid FRAME_RATE value '%s', using default 25 FPS", frameRateStr))
			globalFrameRate = 25
		} else if fps <= 0 || fps > 120 {
			AsyncWarn(fmt.Sprintf("FRAME_RATE %d out of range (1-120), using default 25 FPS", fps))
			globalFrameRate = 25
		} else {
			globalFrameRate = fps
		}
	}

	// Calculate frame interval
	frameInterval = time.Duration(1000/globalFrameRate) * time.Millisecond

	AsyncInfo(fmt.Sprintf("frame rate limit: %d FPS (interval: %v)", globalFrameRate, frameInterval))
}

// autoStartRunningCameras starts all cameras that were marked as running and enabled
func autoStartRunningCameras(rtspManager *RTSPManager) error {
	var errors []string
	runningCount := 0

	for _, camera := range dataStore.Cameras {
		if camera.Enabled && camera.Running {
			runningCount++
			AsyncInfo(fmt.Sprintf("auto-starting camera: %s (%s)", camera.Name, camera.ID))
			if err := rtspManager.StartCamera(camera); err != nil {
				errorMsg := fmt.Sprintf("failed to restart camera %s (%s): %v", camera.Name, camera.ID, err)
				AsyncWarn(errorMsg)
				errors = append(errors, errorMsg)
			} else {
				AsyncInfo(fmt.Sprintf("successfully restarted camera stream: %s (%s)", camera.Name, camera.ID))
			}
		}
	}

	AsyncInfo(fmt.Sprintf("auto-start completed: %d cameras processed", runningCount))

	if len(errors) > 0 {
		return fmt.Errorf("some cameras failed to start: %d/%d failed", len(errors), runningCount)
	}

	return nil
}

func main() {
	// AV_LOG_QUIET = 24
	os.Setenv("OPENCV_FFMPEG_LOGLEVEL", "24")
	os.Setenv("AV_LOG_FORCE_NOCOLOR", "1")

	AsyncInfo("starting Multi-Camera Stream Platform (API Focus)")

	// Initialize frame rate configuration from environment
	initializeFrameRate()

	// Load persistent data store
	if err := loadDataStore(); err != nil {
		AsyncInfo(fmt.Sprintf("failed to load data store: %v", err))
		os.Exit(1)
	}

	// Load configuration
	config, err := LoadConfig(ConfigFile)
	if err != nil {
		AsyncWarn("failed to load config, using defaults")
		globalConfig = DefaultConfig()
	} else {
		globalConfig = config
	}
	webPort := globalConfig.WebPort

	// Create output directory
	if err := os.MkdirAll(OutputDir, 0755); err != nil {
		AsyncWarn(fmt.Sprintf("failed to create output directory: %v", err))
	}

	// Create RTSP manager using gortsplib (pure Go implementation)
	rtspManager := NewRTSPManager()

	// Auto-start cameras that were running before shutdown
	if err := autoStartRunningCameras(rtspManager); err != nil {
		AsyncWarn(fmt.Sprintf("failed to auto-start some cameras: %v", err))
	}

	// Create and start web server with API support
	webServer := NewWebServer(OutputDir, webPort)
	webServer.SetRTSPManager(rtspManager)

	// Setup signal handling
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	// Start web server
	go func() {
		if err := webServer.Start(); err != nil {
			AsyncInfo(fmt.Sprintf("web server error: %v", err))
		}
	}()

	// Wait for exit signal
	AsyncInfo("multi-camera platform is running. Press Ctrl+C to stop.")
	AsyncInfo(fmt.Sprintf("camera Management: http://localhost:%d/cameras", webPort))
	AsyncInfo(fmt.Sprintf("image Viewer: http://localhost:%d", webPort))
	AsyncInfo(fmt.Sprintf("API Debug: http://localhost:%d/api/debug", webPort))
	<-sigChan

	AsyncInfo("received shutdown signal, stopping...")

	// Stop all cameras
	rtspManager.StopAll()

	// Close async logger
	CloseGlobalAsyncLogger()

	AsyncInfo("multi-camera platform stopped")
}
