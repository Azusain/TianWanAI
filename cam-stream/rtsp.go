package main

import (
	"fmt"
	"image"
	"log"
	"sync"
	"time"

	"github.com/deepch/vdk/format/rtsp"
)

// RTSPStreamer RTSP stream processor
type RTSPStreamer struct {
	rtspURL    string
	client     *rtsp.Client
	mutex      sync.Mutex
	running    bool
	lastFrame  image.Image
	frameCount int64
}

// NewRTSPStreamer creates a new RTSP stream processor
func NewRTSPStreamer(rtspURL string) *RTSPStreamer {
	return &RTSPStreamer{
		rtspURL: rtspURL,
		running: false,
	}
}

// Start starts RTSP stream receiving
func (r *RTSPStreamer) Start() error {
	r.mutex.Lock()
	defer r.mutex.Unlock()

	if r.running {
		return fmt.Errorf("RTSP streamer is already running")
	}

	// connect to RTSP stream
	client, err := rtsp.Dial(r.rtspURL)
	if err != nil {
		return fmt.Errorf("failed to connect to RTSP stream: %v", err)
	}

	r.client = client
	r.running = true

	// start frame processing goroutine
	go r.processFrames()

	log.Printf("RTSP streamer started for URL: %s", r.rtspURL)
	return nil
}

// Stop stops RTSP stream receiving
func (r *RTSPStreamer) Stop() error {
	r.mutex.Lock()
	defer r.mutex.Unlock()

	if !r.running {
		return nil
	}

	r.running = false
	if r.client != nil {
		r.client.Close()
	}

	log.Printf("RTSP streamer stopped")
	return nil
}

// processFrames processes incoming RTSP frames
func (r *RTSPStreamer) processFrames() {
	defer func() {
		if rec := recover(); rec != nil {
			log.Printf("RTSP frame processor recovered from panic: %v", rec)
		}
	}()

	for r.running {
		packet, err := r.client.ReadPacket()
		if err != nil {
			log.Printf("Error reading RTSP packet: %v", err)
			time.Sleep(1 * time.Second)
			continue
		}

		// process video packets
		if packet.IsKeyFrame && packet.Idx == 0 {
			// decode frame to image
			img, err := r.decodePacket(packet)
			if err != nil {
				log.Printf("Error decoding frame: %v", err)
				continue
			}

			r.mutex.Lock()
			r.lastFrame = img
			r.frameCount++
			r.mutex.Unlock()
		}
	}
}

// decodePacket decodes video packet to image
func (r *RTSPStreamer) decodePacket(packet interface{}) (image.Image, error) {
	// Note: This is a placeholder for actual video decoding
	// In a real implementation, you would decode H.264/H.265 packets to images
	// This requires additional libraries like FFmpeg bindings
	return nil, fmt.Errorf("video decoding not implemented - requires FFmpeg integration")
}

// GetLatestFrame gets the latest frame
func (r *RTSPStreamer) GetLatestFrame() (image.Image, int64, bool) {
	r.mutex.Lock()
	defer r.mutex.Unlock()

	if r.lastFrame == nil {
		return nil, 0, false
	}

	return r.lastFrame, r.frameCount, true
}

// IsRunning checks if the streamer is running
func (r *RTSPStreamer) IsRunning() bool {
	r.mutex.Lock()
	defer r.mutex.Unlock()
	return r.running
}
