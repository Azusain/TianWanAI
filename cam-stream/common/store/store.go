package store

import (
	"cam-stream/common/config"
	"cam-stream/common/log"
	"encoding/json"
	"fmt"
	"os"
	"sync"
	"time"
)

// Data structures
type InferenceServer struct {
	ID          string    `json:"id"`
	Name        string    `json:"name"` // User-friendly name/alias
	URL         string    `json:"url"`
	ModelType   string    `json:"model_type"`            // e.g., "yolo", "detectron2", "custom"
	Description string    `json:"description,omitempty"` // Optional description
	Enabled     bool      `json:"enabled"`
	CreatedAt   time.Time `json:"created_at"`
	UpdatedAt   time.Time `json:"updated_at"`
}

// InferenceServerBinding represents a binding between camera and inference server with threshold
type InferenceServerBinding struct {
	ServerID     string  `json:"server_id"`
	Threshold    float64 `json:"threshold"`     // Minimum confidence threshold (0.0-1.0) for saving images
	MaxThreshold float64 `json:"max_threshold"` // Maximum confidence threshold (0.0-1.0) for saving images
}

type CameraConfig struct {
	ID                      string                   `json:"id"`
	Name                    string                   `json:"name"` // Now directly contains KKS encoding
	RTSPUrl                 string                   `json:"rtsp_url"`
	InferenceServerBindings []InferenceServerBinding `json:"inference_server_bindings,omitempty"` // Array of server bindings with thresholds
	Enabled                 bool                     `json:"enabled"`
	Running                 bool                     `json:"running"`
	CreatedAt               time.Time                `json:"created_at"`
	UpdatedAt               time.Time                `json:"updated_at"`
}

// AlertServerConfig represents the global alert server configuration
type AlertServerConfig struct {
	URL       string    `json:"url"`     // Alert platform URL
	Enabled   bool      `json:"enabled"` // Whether alert is enabled globally
	UpdatedAt time.Time `json:"updated_at"`
}

// FallDetectionTaskState represents the state of a fall detection task
type FallDetectionTaskState struct {
	TaskID    string    `json:"task_id"`   // Task ID from tianwan service
	CameraID  string    `json:"camera_id"` // Camera ID
	ServerID  string    `json:"server_id"` // Inference server ID
	Status    string    `json:"status"`    // "running", "stopped", "error"
	StartedAt time.Time `json:"started_at"`
	UpdatedAt time.Time `json:"updated_at"`
	ErrorMsg  string    `json:"error_msg,omitempty"`

	// Polling mechanism fields (not persisted to JSON)
	PollTicker   *time.Ticker `json:"-"` // Ticker for periodic result polling
	PollStopChan chan bool    `json:"-"` // Channel to stop the polling goroutine
}

type DataStore struct {
	Cameras          map[string]*CameraConfig    `json:"cameras"`
	InferenceServers map[string]*InferenceServer `json:"inference_servers"`
	AlertServer      *AlertServerConfig          `json:"alert_server,omitempty"` // Global alert server config
}

// Global data store
var Data = &DataStore{
	Cameras:          make(map[string]*CameraConfig),
	InferenceServers: make(map[string]*InferenceServer),
}

// Global mutex to protect dataStore concurrent access
var dataStoreMutex sync.RWMutex

// Runtime-only fall detection task tracking (not persisted)
var FallDetectionTasks = make(map[string]*FallDetectionTaskState)
var fallDetectionTasksMutex sync.RWMutex

// Thread-safe helper functions
func SafeGetCamera(id string) (*CameraConfig, bool) {
	dataStoreMutex.RLock()
	defer dataStoreMutex.RUnlock()
	camera, exists := Data.Cameras[id]
	return camera, exists
}

func SafeGetInferenceServer(id string) (*InferenceServer, bool) {
	dataStoreMutex.RLock()
	defer dataStoreMutex.RUnlock()
	server, exists := Data.InferenceServers[id]
	return server, exists
}

func SafeUpdateDataStore(fn func()) {
	dataStoreMutex.Lock()
	defer dataStoreMutex.Unlock()
	fn()
}

func SafeReadDataStore(fn func()) {
	dataStoreMutex.RLock()
	defer dataStoreMutex.RUnlock()
	fn()
}

func SafeUpdateTasks(fn func()) {
	fallDetectionTasksMutex.Lock()
	defer fallDetectionTasksMutex.Unlock()
	fn()
}

func SafeReadTasks(fn func()) {
	fallDetectionTasksMutex.RLock()
	defer fallDetectionTasksMutex.RUnlock()
	fn()
}

// Data persistence functions
func LoadDataStore() error {
	if err := os.MkdirAll(config.DataDir, 0755); err != nil {
		return fmt.Errorf("failed to create data directory: %v", err)
	}

	data, err := os.ReadFile(config.DataFile)
	if err != nil {
		if os.IsNotExist(err) {
			log.Info("data file not found, starting with empty store")
			return nil
		}
		return fmt.Errorf("failed to read data file: %v", err)
	}

	var tempStore DataStore
	if err := json.Unmarshal(data, &tempStore); err != nil {
		return fmt.Errorf("failed to parse data file: %v", err)
	}

	SafeUpdateDataStore(func() {
		*Data = tempStore
		if Data.Cameras == nil {
			Data.Cameras = make(map[string]*CameraConfig)
		}
		if Data.InferenceServers == nil {
			Data.InferenceServers = make(map[string]*InferenceServer)
		}
	})

	var camerasCount, serversCount int
	var alertConfigured bool
	var alertEnabled bool
	var alertURL string
	SafeReadDataStore(func() {
		camerasCount = len(Data.Cameras)
		serversCount = len(Data.InferenceServers)
		if Data.AlertServer != nil {
			alertConfigured = true
			alertEnabled = Data.AlertServer.Enabled
			alertURL = Data.AlertServer.URL
		}
	})

	log.Info(fmt.Sprintf("loaded %d cameras and %d inference servers from storage", camerasCount, serversCount))

	if alertConfigured {
		if alertEnabled {
			log.Info(fmt.Sprintf("alert server configured and enabled: %s", alertURL))
		} else {
			log.Info(fmt.Sprintf("alert server configured but disabled: %s", alertURL))
		}
	} else {
		log.Info("alert server not configured - alerts disabled")
	}

	return nil
}

func SaveDataStore() error {
	if err := os.MkdirAll(config.DataDir, 0755); err != nil {
		return fmt.Errorf("failed to create data directory: %v", err)
	}

	var data []byte
	var err error
	var camerasCount, serversCount int
	SafeReadDataStore(func() {
		data, err = json.MarshalIndent(Data, "", "  ")
		camerasCount = len(Data.Cameras)
		serversCount = len(Data.InferenceServers)
	})

	if err != nil {
		return fmt.Errorf("failed to marshal data: %v", err)
	}

	if err := os.WriteFile(config.DataFile, data, 0644); err != nil {
		return fmt.Errorf("failed to write data file: %v", err)
	}

	log.Info(fmt.Sprintf("saved data store with %d cameras and %d inference servers", camerasCount, serversCount))
	return nil
}
