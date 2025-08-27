package main

import (
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
)

const (
	ConfigFile = "config.json"
	OutputDir  = "output"
)

// autoStartRunningCameras starts all cameras that were marked as running and enabled
func autoStartRunningCameras(rtspManager *FFmpegCmdRTSPManager) error {
	var errors []string
	runningCount := 0

	for _, camera := range dataStore.Cameras {
		if camera.Enabled && camera.Running {
			runningCount++
			log.Printf("Auto-starting camera: %s (%s)", camera.Name, camera.ID)
			if err := rtspManager.StartCamera(camera); err != nil {
				errorMsg := fmt.Sprintf("Failed to restart camera %s (%s): %v", camera.Name, camera.ID, err)
				log.Printf("Warning: %s", errorMsg)
				errors = append(errors, errorMsg)
			} else {
				log.Printf("Successfully restarted camera stream: %s (%s)", camera.Name, camera.ID)
			}
		}
	}

	log.Printf("Auto-start completed: %d cameras processed", runningCount)

	if len(errors) > 0 {
		return fmt.Errorf("some cameras failed to start: %d/%d failed", len(errors), runningCount)
	}

	return nil
}

func main() {
	log.Printf("Starting Multi-Camera Stream Platform (API Focus)")

	// Load persistent data store
	if err := loadDataStore(); err != nil {
		log.Fatalf("Failed to load data store: %v", err)
	}

	// Load configuration
	config, err := LoadConfig(ConfigFile)
	var webPort int
	if err != nil {
		log.Printf("Warning: Failed to load config, using default web port 3001")
		webPort = 3001
	} else {
		webPort = config.WebPort
	}

	// Create output directory
	if err := os.MkdirAll(OutputDir, 0755); err != nil {
		log.Printf("Warning: Failed to create output directory: %v", err)
	}

	// Create FFmpeg command RTSP manager (uses local ffmpeg to pull real video streams)
	rtspManager := NewFFmpegCmdRTSPManager()

	// Auto-start cameras that were running before shutdown
	if err := autoStartRunningCameras(rtspManager); err != nil {
		log.Printf("Warning: Failed to auto-start some cameras: %v", err)
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
			log.Printf("Web server error: %v", err)
		}
	}()

	// Wait for exit signal
	log.Printf("Multi-camera platform is running. Press Ctrl+C to stop.")
	log.Printf("Camera Management: http://localhost:%d/cameras", webPort)
	log.Printf("Image Viewer: http://localhost:%d", webPort)
	log.Printf("API Debug: http://localhost:%d/api/debug", webPort)
	<-sigChan

	log.Printf("Received shutdown signal, stopping...")
	
	// Stop all cameras
	rtspManager.StopAll()
	
	log.Printf("Multi-camera platform stopped")
}
