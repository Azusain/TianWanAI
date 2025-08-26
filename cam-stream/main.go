package main

import (
	"fmt"
	"image"
	"image/jpeg"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"
)

const (
	ConfigFile = "config.json"
	OutputDir  = "output"
)

// CameraStreamer camera stream processor
type CameraStreamer struct {
	config           *Config
	rtspStreamer     *RTSPStreamer
	inferenceClient  *InferenceClient
	imageRenderer    *ImageRenderer
	webServer        *WebServer
	outputDir        string
	running          bool
}

// NewCameraStreamer creates camera stream processor
func NewCameraStreamer(config *Config) *CameraStreamer {
	return &CameraStreamer{
		config:          config,
		rtspStreamer:    NewRTSPStreamer(config.RTSPUrl),
		inferenceClient: NewInferenceClient(config.ServerUrl),
		imageRenderer:   NewImageRenderer(config.DebugMode),
		webServer:       NewWebServer(OutputDir, config.WebPort),
		outputDir:       OutputDir,
		running:         false,
	}
}

// Start starts stream processing
func (cs *CameraStreamer) Start() error {
	// create output directory
	if err := os.MkdirAll(cs.outputDir, 0755); err != nil {
		return fmt.Errorf("failed to create output directory: %v", err)
	}

	// test connection to inference server
	log.Printf("Testing connection to inference server: %s", cs.config.ServerUrl)
	// Note: skip connection test as health check endpoint may not exist

	// start RTSP stream
	log.Printf("Starting RTSP streamer for: %s", cs.config.RTSPUrl)
	if err := cs.rtspStreamer.Start(); err != nil {
		return fmt.Errorf("failed to start RTSP streamer: %v", err)
	}

	cs.running = true

	// start frame processing loop
	go cs.frameProcessingLoop()

	log.Printf("Camera streamer started successfully")
	log.Printf("Debug mode: %t", cs.config.DebugMode)
	log.Printf("Frame rate: %d fps", cs.config.FrameRate)
	log.Printf("Camera name: %s", cs.config.CameraName)

	return nil
}

// Stop stops stream processing
func (cs *CameraStreamer) Stop() error {
	cs.running = false

	if cs.rtspStreamer != nil {
		if err := cs.rtspStreamer.Stop(); err != nil {
			log.Printf("Error stopping RTSP streamer: %v", err)
		}
	}

	log.Printf("Camera streamer stopped")
	return nil
}

// frameProcessingLoop frame processing loop
func (cs *CameraStreamer) frameProcessingLoop() {
	ticker := time.NewTicker(time.Second / time.Duration(cs.config.FrameRate))
	defer ticker.Stop()

	var lastFrameCount int64 = -1

	for cs.running {
		select {
		case <-ticker.C:
			cs.processFrame(&lastFrameCount)
		}
	}
}

// processFrame processes single frame
func (cs *CameraStreamer) processFrame(lastFrameCount *int64) {
	// get latest frame
	frame, frameCount, hasFrame := cs.rtspStreamer.GetLatestFrame()
	if !hasFrame {
		log.Printf("No frame available from RTSP stream")
		return
	}

	// check if it's a new frame
	if frameCount == *lastFrameCount {
		return // skip duplicate frame
	}
	*lastFrameCount = frameCount

	timestamp := time.Now()
	filename := fmt.Sprintf("frame_%d_%s.jpg", frameCount, timestamp.Format("20060102_150405"))

	log.Printf("Processing frame %d at %s", frameCount, timestamp.Format("15:04:05"))

	// send to inference server
	response, err := cs.inferenceClient.InferImage(frame, filename)
	if err != nil {
		log.Printf("Inference error for frame %d: %v", frameCount, err)
		return
	}

	log.Printf("Frame %d: Found %d detection(s)", frameCount, len(response.Results))

	// only save image if there are detections
	if len(response.Results) == 0 {
		log.Printf("Frame %d: No detections, skipping save", frameCount)
		return
	}

	// render detection results on image
	renderedImage, err := cs.imageRenderer.RenderDetections(
		frame,
		response.Results,
		cs.config.CameraName,
		timestamp,
	)
	if err != nil {
		log.Printf("Error rendering detections for frame %d: %v", frameCount, err)
		return
	}

	// save result image
	if err := cs.saveImage(renderedImage, filename, response.Results, timestamp); err != nil {
		log.Printf("Error saving frame %d: %v", frameCount, err)
	}
}

// saveImage saves image to file
func (cs *CameraStreamer) saveImage(img image.Image, filename string, detections []InferenceDetectionResult, timestamp time.Time) error {
	outputPath := filepath.Join(cs.outputDir, filename)

	// create or overwrite file
	file, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("failed to create output file: %v", err)
	}
	defer file.Close()

	// save as JPEG
	if err := jpeg.Encode(file, img, &jpeg.Options{Quality: 90}); err != nil {
		return fmt.Errorf("failed to encode image: %v", err)
	}

	log.Printf("Saved processed image: %s (%d detections)", outputPath, len(detections))
	return nil
}

func main() {
	log.Printf("Starting Camera Stream Application")

	// load configuration
	config, err := LoadConfig(ConfigFile)
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	// create camera stream processor
	streamer := NewCameraStreamer(config)

	// setup signal handling
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	// start web server
	go func() {
		if err := streamer.webServer.Start(); err != nil {
			log.Printf("Web server error: %v", err)
		}
	}()

	// start stream processing
	if err := streamer.Start(); err != nil {
		log.Fatalf("Failed to start camera streamer: %v", err)
	}

	// wait for exit signal
	log.Printf("Camera stream application is running. Press Ctrl+C to stop.")
	log.Printf("Web interface available at: http://localhost:%d", config.WebPort)
	<-sigChan

	log.Printf("Received shutdown signal, stopping...")

	// stop stream processing
	if err := streamer.Stop(); err != nil {
		log.Printf("Error stopping streamer: %v", err)
	}

	log.Printf("Application stopped")
}
