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
	DebugDir   = "debug"
)

// Global config variable
var globalConfig *Config

// Global DEBUG mode flag
var globalDebugMode bool

// initializeFrameRate initializes frame rate configuration from environment variable
func initializeFrameRate() {
	// Read frame rate from environment variable, default to 25
	frameRateStr := os.Getenv("FRAME_RATE")
	if frameRateStr == "" {
		// default 25 FPS
		globalFrameRate = 25
	} else {
		if fps, err := strconv.Atoi(frameRateStr); err != nil {
			Warn(fmt.Sprintf("invalid FRAME_RATE value '%s', using default 25 FPS", frameRateStr))
			globalFrameRate = 25
		} else if fps <= 0 || fps > 120 {
			Warn(fmt.Sprintf("FRAME_RATE %d out of range (1-120), using default 25 FPS", fps))
			globalFrameRate = 25
		} else {
			globalFrameRate = fps
		}
	}

	// Calculate frame interval
	frameInterval = time.Duration(1000/globalFrameRate) * time.Millisecond

	Info(fmt.Sprintf("frame rate limit: %d FPS (interval: %v)", globalFrameRate, frameInterval))
}

// initializeDebugMode initializes DEBUG mode from environment variable
func initializeDebugMode() {
	debugStr := os.Getenv("DEBUG")
	globalDebugMode = debugStr != "" && debugStr != "0" && debugStr != "false"

	if globalDebugMode {
		Info("🐛 DEBUG MODE ENABLED - Original images will be saved to debug directory")
		// Create debug directory
		if err := os.MkdirAll(DebugDir, 0755); err != nil {
			Warn(fmt.Sprintf("failed to create debug directory: %v", err))
		}
	}
}

// autoStartRunningCameras starts all cameras that were marked as running and enabled
func autoStartRunningCameras(rtspManager *RTSPManager) error {
	var errors []string
	runningCount := 0

	// Use thread-safe access to iterate over cameras
	var camerasToStart []*CameraConfig
	safeReadDataStore(func() {
		for _, camera := range dataStore.Cameras {
			if camera.Enabled && camera.Running {
				camerasToStart = append(camerasToStart, camera)
			}
		}
	})

	for _, camera := range camerasToStart {
		runningCount++
		Info(fmt.Sprintf("auto-starting camera: %s (%s)", camera.Name, camera.ID))
		if err := rtspManager.StartCamera(camera); err != nil {
			errorMsg := fmt.Sprintf("failed to restart camera %s (%s): %v", camera.Name, camera.ID, err)
			Warn(errorMsg)
			errors = append(errors, errorMsg)
		} else {
			Info(fmt.Sprintf("successfully restarted camera stream: %s (%s)", camera.Name, camera.ID))
		}
	}

	Info(fmt.Sprintf("auto-start completed: %d cameras processed", runningCount))

	if len(errors) > 0 {
		return fmt.Errorf("some cameras failed to start: %d/%d failed", len(errors), runningCount)
	}

	return nil
}

func main() {
	// AV_LOG_QUIET = 24
	os.Setenv("OPENCV_FFMPEG_LOGLEVEL", "24")
	os.Setenv("AV_LOG_FORCE_NOCOLOR", "1")

	Info("starting Multi-Camera Stream Platform (API Focus)")

	// Initialize frame rate configuration from environment
	initializeFrameRate()

	// Initialize DEBUG mode from environment
	initializeDebugMode()

	// Load persistent data store
	if err := loadDataStore(); err != nil {
		Info(fmt.Sprintf("failed to load data store: %v", err))
		os.Exit(1)
	}

	// Load configuration
	config, err := LoadConfig(ConfigFile)
	if err != nil {
		Warn("failed to load config, using defaults")
		globalConfig = DefaultConfig()
	} else {
		globalConfig = config
	}
	webPort := globalConfig.WebPort

	// Create output directory
	if err := os.MkdirAll(OutputDir, 0755); err != nil {
		Warn(fmt.Sprintf("failed to create output directory: %v", err))
	}

	// Create RTSP manager using gortsplib (pure Go implementation)
	rtspManager := NewRTSPManager()

	// Auto-start cameras that were running before shutdown
	if err := autoStartRunningCameras(rtspManager); err != nil {
		Warn(fmt.Sprintf("failed to auto-start some cameras: %v", err))
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
			Info(fmt.Sprintf("web server error: %v", err))
		}
	}()

	// Wait for exit signal
	Info("multi-camera platform is running. Press Ctrl+C to stop.")
	Info(fmt.Sprintf("camera Management: http://localhost:%d/cameras", webPort))
	Info(fmt.Sprintf("image Viewer: http://localhost:%d", webPort))
	Info(fmt.Sprintf("API Debug: http://localhost:%d/api/debug", webPort))
	<-sigChan

	Info("received shutdown signal, stopping...")

	// Stop all cameras
	rtspManager.StopAll()

	// Close async logger
	CloseGlobalAsyncLogger()

	Info("multi-camera platform stopped")
}
