package main

import (
	"encoding/json"
	"fmt"
	"os"
)

// Config application configuration structure
type Config struct {
	RTSPUrl      string `json:"rtsp_url"`
	ServerUrl    string `json:"server_url"`
	CameraName   string `json:"camera_name"`
	DebugMode    bool   `json:"debug_mode"`
	FrameRate    int    `json:"frame_rate"`     // frames per second for sampling
	OutputWidth  int    `json:"output_width"`  // output image width
	OutputHeight int    `json:"output_height"` // output image height
	WebPort      int    `json:"web_port"`       // web interface port
}

// DefaultConfig returns default configuration
func DefaultConfig() *Config {
	return &Config{
		RTSPUrl:      "rtsp://admin:password@192.168.1.100:554/stream1",
		ServerUrl:    "http://localhost:8080/tshirt",
		CameraName:   "Camera-01",
		DebugMode:    true,
		FrameRate:    1, // 1 frame per second
		OutputWidth:  1920,
		OutputHeight: 1080,
		WebPort:      3000, // web interface port
	}
}

// LoadConfig loads configuration from file
func LoadConfig(filename string) (*Config, error) {
	config := DefaultConfig()

	if _, err := os.Stat(filename); os.IsNotExist(err) {
		// config file does not exist, create default config file
		if err := SaveConfig(config, filename); err != nil {
			return nil, fmt.Errorf("failed to create default config: %v", err)
		}
		fmt.Printf("Created default config file: %s\n", filename)
		return config, nil
	}

	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %v", err)
	}

	if err := json.Unmarshal(data, config); err != nil {
		return nil, fmt.Errorf("failed to parse config file: %v", err)
	}

	return config, nil
}

// SaveConfig saves configuration to file
func SaveConfig(config *Config, filename string) error {
	data, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal config: %v", err)
	}

	if err := os.WriteFile(filename, data, 0644); err != nil {
		return fmt.Errorf("failed to write config file: %v", err)
	}

	return nil
}
