package main

import (
	"cam-stream/common/config"
	"cam-stream/common/log"
	"cam-stream/common/store"
	"cam-stream/service"
	"fmt"
	"os"
	"os/signal"
	"runtime"
	"strconv"
	"syscall"
	"time"
)

func autoStartRunningCameras(rtspManager *service.RTSPManager) error {
	var camerasToStart []*store.CameraConfig
	var errors []string
	runningCount := 0

	store.SafeReadDataStore(func() {
		for _, camera := range store.Data.Cameras {
			if camera.Enabled && camera.Running {
				camerasToStart = append(camerasToStart, camera)
			}
		}
	})

	for _, camera := range camerasToStart {
		runningCount++
		log.Info(fmt.Sprintf("auto-starting camera: %s (%s)", camera.Name, camera.ID))
		if err := rtspManager.StartCamera(camera); err != nil {
			errorMsg := fmt.Sprintf("failed to restart camera %s (%s): %v", camera.Name, camera.ID, err)
			log.Warn(errorMsg)
			errors = append(errors, errorMsg)
			continue
		}
		log.Info(fmt.Sprintf("successfully restarted camera stream: %s (%s)", camera.Name, camera.ID))
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
		log.Warn(fmt.Sprintf("panic recovered: %v", r))
		log.Warn(fmt.Sprintf("stack trace:\n%s", stackTrace))
		// Give some time for logs to flush
		time.Sleep(1 * time.Second)
	}
}

func runApplication() error {
	defer recoverFromPanic()

	// Initialize frame rate configuration from environment
	frameRateStr := os.Getenv("FRAME_RATE")
	if frameRateStr == "" {
		config.GlobalFrameRate = 25
	} else if fps, err := strconv.Atoi(frameRateStr); err == nil && fps <= 120 && fps > 0 {
		config.GlobalFrameRate = fps
	} else {
		log.Error(fmt.Sprintf("invalid FRAME_RATE value '%s', using default 25 FPS", frameRateStr))
		os.Exit(-1)
	}
	config.GlobalFrameInterval = time.Duration(1000/config.GlobalFrameRate) * time.Millisecond
	log.Info(fmt.Sprintf("frame rate limit: %d FPS (interval: %v)", config.GlobalFrameRate, config.GlobalFrameInterval))

	// DEBUG Mode.
	debugStr := os.Getenv("DEBUG")
	config.GlobalDebugMode = debugStr != "" && debugStr != "0" && debugStr != "false"
	if config.GlobalDebugMode {
		log.Info("🐛 DEBUG MODE ENABLED - Original images will be saved to debug directory")
		if err := os.MkdirAll(config.DebugDir, 0755); err != nil {
			log.Warn(fmt.Sprintf("failed to create debug directory: %v", err))
		}
	}

	// Load persistent data store
	if err := store.LoadDataStore(); err != nil {
		return fmt.Errorf("failed to load data store: %v", err)
	}

	// TODO: load configuration from local file.
	globalWebPort := config.DefaultWebPort

	if err := os.MkdirAll(config.OutputDir, 0755); err != nil {
		log.Error(fmt.Sprintf("failed to create output directory: %v", err))
		os.Exit(-1)
	}

	rtspManager := service.NewRTSPManager()

	// cleanup function.
	defer func() {
		defer recoverFromPanic()
		if rtspManager != nil {
			rtspManager.StopAll()
		}
	}()

	if err := autoStartRunningCameras(rtspManager); err != nil {
		log.Warn(fmt.Sprintf("failed to auto-start some cameras: %v", err))
	}

	webServer := service.NewWebServer(config.OutputDir, globalWebPort, rtspManager)

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	errorChan := make(chan error, 1)
	go func() {
		defer recoverFromPanic()
		if err := webServer.Start(); err != nil {
			log.Warn(fmt.Sprintf("web server error: %v", err))
			errorChan <- err
		}
	}()

	// block and wait for signals.
	select {
	case <-sigChan:
		log.Info("received shutdown signal, stopping...")
		return nil
	case err := <-errorChan:
		log.Warn(fmt.Sprintf("application error occurred: %v", err))
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
			log.Error(fmt.Sprintf("restart attempt due to error: %v", err))
			continue
		}
		log.Info("application shutdown normally")
		break
	}

	// release resource
	log.CloseGlobalAsyncLogger()
}
