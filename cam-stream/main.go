package main

import (
	"fmt"
	"os"
	"os/signal"
	"runtime"
	"strconv"
	"syscall"
	"time"
)

const (
	OutputDir           = "output"
	DebugDir            = "debug"
	DefaultWebPort uint = 8080
)

// Readonly so we dont need to protect it with lock.
var globalDebugMode bool

func autoStartRunningCameras(rtspManager *RTSPManager) error {
	var camerasToStart []*CameraConfig
	var errors []string
	runningCount := 0

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
			continue
		}
		Info(fmt.Sprintf("successfully restarted camera stream: %s (%s)", camera.Name, camera.ID))
	}

	if len(errors) > 0 {
		return fmt.Errorf("some cameras failed to start: %d/%d failed", len(errors), runningCount)
	}

	return nil
}

func recoverFromPanic() {
	if r := recover(); r != nil {
		// get stack trace and log panic details
		buf := make([]byte, 4096)
		n := runtime.Stack(buf, false)
		stackTrace := string(buf[:n])
		Warn(fmt.Sprintf("panic recovered: %v", r))
		Warn(fmt.Sprintf("stack trace:\n%s", stackTrace))
		// Give some time for logs to flush
		time.Sleep(1 * time.Second)
	}
}

func runApplication() error {
	defer recoverFromPanic()

	// Initialize frame rate configuration from environment
	frameRateStr := os.Getenv("FRAME_RATE")
	if frameRateStr == "" {
		globalFrameRate = 25
	} else if fps, err := strconv.Atoi(frameRateStr); err == nil && fps <= 120 && fps > 0 {
		globalFrameRate = fps
	} else {
		Error(fmt.Sprintf("invalid FRAME_RATE value '%s', using default 25 FPS", frameRateStr))
		os.Exit(-1)
	}
	globalFrameInterval = time.Duration(1000/globalFrameRate) * time.Millisecond
	Info(fmt.Sprintf("frame rate limit: %d FPS (interval: %v)", globalFrameRate, globalFrameInterval))

	// DEBUG Mode.
	debugStr := os.Getenv("DEBUG")
	globalDebugMode = debugStr != "" && debugStr != "0" && debugStr != "false"
	if globalDebugMode {
		Info("🐛 DEBUG MODE ENABLED - Original images will be saved to debug directory")
		if err := os.MkdirAll(DebugDir, 0755); err != nil {
			Warn(fmt.Sprintf("failed to create debug directory: %v", err))
		}
	}

	// Load persistent data store
	if err := LoadDataStore(); err != nil {
		return fmt.Errorf("failed to load data store: %v", err)
	}

	// TODO: load configuration from local file.
	globalWebPort := DefaultWebPort

	if err := os.MkdirAll(OutputDir, 0755); err != nil {
		Error(fmt.Sprintf("failed to create output directory: %v", err))
		os.Exit(-1)
	}

	rtspManager := NewRTSPManager()

	// cleanup function.
	defer func() {
		defer recoverFromPanic()
		if rtspManager != nil {
			rtspManager.StopAll()
		}
	}()

	if err := autoStartRunningCameras(rtspManager); err != nil {
		Warn(fmt.Sprintf("failed to auto-start some cameras: %v", err))
	}

	webServer := NewWebServer(OutputDir, globalWebPort, rtspManager)

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	errorChan := make(chan error, 1)
	go func() {
		defer recoverFromPanic()
		if err := webServer.Start(); err != nil {
			Warn(fmt.Sprintf("web server error: %v", err))
			errorChan <- err
		}
	}()

	// block and wait for signals.
	select {
	case <-sigChan:
		Info("received shutdown signal, stopping...")
		return nil
	case err := <-errorChan:
		Warn(fmt.Sprintf("application error occurred: %v", err))
		return err
	}

}

func main() {
	// Set up environment variables
	// AV_LOG_QUIET = 24
	os.Setenv("OPENCV_FFMPEG_LOGLEVEL", "24")
	os.Setenv("AV_LOG_FORCE_NOCOLOR", "1")

	for {
		if err := runApplication(); err != nil {
			Error(fmt.Sprintf("restart attempt due to error: %v", err))
			continue
		}
		Info("application shutdown normally")
		break
	}

	// release resource
	CloseGlobalAsyncLogger()
}
