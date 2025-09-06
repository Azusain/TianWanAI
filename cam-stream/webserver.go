package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/gorilla/mux"
)

// WebServer handles web interface
type WebServer struct {
	outputDir   string
	port        int
	rtspManager *FFmpegCmdRTSPManager
}

// NewWebServer creates a new web server
func NewWebServer(outputDir string, port int) *WebServer {
	return &WebServer{
		outputDir: outputDir,
		port:      port,
	}
}

// SetRTSPManager sets the RTSP manager for camera operations
func (ws *WebServer) SetRTSPManager(manager *FFmpegCmdRTSPManager) {
	ws.rtspManager = manager
}

// Start starts the web server
func (ws *WebServer) Start() error {
	router := mux.NewRouter()

	// Add CORS middleware
	router.Use(func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Access-Control-Allow-Origin", "*")
			w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
			w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With")

			if r.Method == "OPTIONS" {
				w.WriteHeader(http.StatusOK)
				return
			}

			next.ServeHTTP(w, r)
		})
	})

	// Camera API Routes - MUST be registered BEFORE catch-all routes
	api := router.PathPrefix("/api").Subrouter()
	api.HandleFunc("/cameras", ws.handleAPICameras).Methods("GET", "POST", "OPTIONS")
	api.HandleFunc("/cameras/{id}", ws.handleAPICameraByID).Methods("GET", "PUT", "DELETE", "OPTIONS")
	api.HandleFunc("/cameras/{id}/start", ws.handleAPIStartCamera).Methods("POST", "OPTIONS")
	api.HandleFunc("/cameras/{id}/stop", ws.handleAPIStopCamera).Methods("POST", "OPTIONS")

	// Inference Server API Routes
	api.HandleFunc("/inference-servers", ws.handleAPIInferenceServers).Methods("GET", "POST", "OPTIONS")
	api.HandleFunc("/inference-servers/{id}", ws.handleAPIInferenceServerByID).Methods("GET", "PUT", "DELETE", "OPTIONS")

	// Alert Server API Routes
	api.HandleFunc("/alert-server", ws.handleAPIAlertServer).Methods("GET", "PUT", "OPTIONS")

	api.HandleFunc("/status", ws.handleAPIStatus).Methods("GET", "OPTIONS")
	api.HandleFunc("/debug", ws.handleAPIDebug).Methods("GET", "OPTIONS")

	// Image management API routes
	api.HandleFunc("/images/{cameraId}", ws.handleAPIImages).Methods("GET", "OPTIONS")
	api.HandleFunc("/images/{cameraId}/{filename}", ws.handleAPIImageFile).Methods("GET", "OPTIONS")

	// Web Routes
	router.HandleFunc("/", ws.handleIndex).Methods("GET")
	router.HandleFunc("/cameras", ws.handleCameraManagement).Methods("GET")
	router.HandleFunc("/images", ws.handleImages).Methods("GET")
	router.HandleFunc("/alerts", ws.handleAlerts).Methods("GET")

	// Static file server for output directory (images)
	router.PathPrefix("/output/").Handler(http.StripPrefix("/output/", http.FileServer(http.Dir("output/"))))

	// Static file server for HTML files - MUST be LAST as it's a catch-all
	router.PathPrefix("/").Handler(http.FileServer(http.Dir("./")))

	log.Printf("Starting web server on port %d", ws.port)
	log.Printf("Access web interface at: http://localhost:%d", ws.port)

	return http.ListenAndServe(fmt.Sprintf(":%d", ws.port), router)
}

// handleIndex serves the main page with links to different sections
func (ws *WebServer) handleIndex(w http.ResponseWriter, r *http.Request) {
	content := `
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cam-Stream 监控平台</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #f8f9fa;
            color: #333;
            padding: 50px 20px;
            margin: 0;
        }
        .container {
            max-width: 600px;
            margin: 0 auto;
            text-align: center;
        }
        h1 {
            font-size: 2rem;
            margin-bottom: 10px;
            color: #2c3e50;
        }
        .subtitle {
            color: #666;
            margin-bottom: 40px;
            font-size: 1rem;
        }
        .nav-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }
        .nav-item {
            background: white;
            padding: 30px 20px;
            border-radius: 8px;
            text-decoration: none;
            color: #333;
            border: 1px solid #e0e0e0;
            transition: all 0.2s;
        }
        .nav-item:hover {
            border-color: #3498db;
            box-shadow: 0 4px 12px rgba(52, 152, 219, 0.15);
        }
        .nav-icon {
            font-size: 2rem;
            margin-bottom: 15px;
            display: block;
        }
        .nav-title {
            font-weight: 600;
            font-size: 1rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Cam-Stream 监控平台</h1>
        <div class="subtitle">智能视频监控与AI检测系统</div>
        
        <div class="nav-grid">
            <a href="/cameras" class="nav-item">
                <span class="nav-icon">⚙️</span>
                <div class="nav-title">摄像头管理</div>
            </a>
            <a href="/images" class="nav-item">
                <span class="nav-icon">📷</span>
                <div class="nav-title">图片查看器</div>
            </a>
            <a href="/alerts" class="nav-item">
                <span class="nav-icon">🚨</span>
                <div class="nav-title">告警配置</div>
            </a>
            <a href="/api/debug" class="nav-item">
                <span class="nav-icon">🔧</span>
                <div class="nav-title">系统调试</div>
            </a>
        </div>
    </div>
</body>
</html>
	`
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write([]byte(content))
}

// handleCameraManagement serves the camera management page
func (ws *WebServer) handleCameraManagement(w http.ResponseWriter, r *http.Request) {
	content, err := ioutil.ReadFile("camera_management.html")
	if err != nil {
		http.Error(w, "Could not load camera management interface: "+err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write(content)
}

// handleImages serves the image viewer page
func (ws *WebServer) handleImages(w http.ResponseWriter, r *http.Request) {
	content, err := ioutil.ReadFile("image_viewer.html")
	if err != nil {
		http.Error(w, "Could not load image viewer: "+err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write(content)
}

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
	ServerID  string  `json:"server_id"`
	Threshold float64 `json:"threshold"` // Confidence threshold (0.0-1.0) for saving images
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
	// Keep these for backward compatibility during migration
	// InferenceServers []string `json:"inference_servers,omitempty"` // Deprecated - migrate to bindings
	// ServerUrl        string   `json:"server_url,omitempty"`        // Deprecated
	// ModelType        string   `json:"model_type,omitempty"`        // Deprecated
	// PlatformURL      string   `json:"platform_url,omitempty"`      // Deprecated - use global alert config
	// CameraKKS        string   `json:"camera_kks,omitempty"`        // Deprecated - use camera name directly
}

type APIResponse struct {
	Success bool        `json:"success"`
	Message string      `json:"message"`
	Data    interface{} `json:"data,omitempty"`
	Error   string      `json:"error,omitempty"`
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
}

type DataStore struct {
	Cameras          map[string]*CameraConfig    `json:"cameras"`
	InferenceServers map[string]*InferenceServer `json:"inference_servers"`
	AlertServer      *AlertServerConfig          `json:"alert_server,omitempty"` // Global alert server config
	Counters         struct {
		Camera          int `json:"camera"`
		InferenceServer int `json:"inference_server"`
	} `json:"counters"`
}

const (
	DataFile = "_data/cameras.json"
	DataDir  = "_data"
)

// Global data store
var dataStore = &DataStore{
	Cameras:          make(map[string]*CameraConfig),
	InferenceServers: make(map[string]*InferenceServer),
}

// Runtime-only fall detection task tracking (not persisted)
var fallDetectionTasks = make(map[string]*FallDetectionTaskState)

// Data persistence functions
func loadDataStore() error {
	if err := os.MkdirAll(DataDir, 0755); err != nil {
		return fmt.Errorf("failed to create data directory: %v", err)
	}

	data, err := ioutil.ReadFile(DataFile)
	if err != nil {
		if os.IsNotExist(err) {
			log.Printf("Data file not found, starting with empty store")
			return nil
		}
		return fmt.Errorf("failed to read data file: %v", err)
	}

	if err := json.Unmarshal(data, dataStore); err != nil {
		return fmt.Errorf("failed to parse data file: %v", err)
	}

	if dataStore.Cameras == nil {
		dataStore.Cameras = make(map[string]*CameraConfig)
	}
	if dataStore.InferenceServers == nil {
		dataStore.InferenceServers = make(map[string]*InferenceServer)
	}

	log.Printf("Loaded %d cameras and %d inference servers from storage", len(dataStore.Cameras), len(dataStore.InferenceServers))

	// Log alert server configuration status
	if dataStore.AlertServer != nil {
		if dataStore.AlertServer.Enabled {
			log.Printf("Alert server configured and enabled: %s", dataStore.AlertServer.URL)
		} else {
			log.Printf("Alert server configured but disabled: %s", dataStore.AlertServer.URL)
		}
	} else {
		log.Printf("Alert server not configured - alerts disabled")
	}

	return nil
}

func saveDataStore() error {
	if err := os.MkdirAll(DataDir, 0755); err != nil {
		return fmt.Errorf("failed to create data directory: %v", err)
	}

	data, err := json.MarshalIndent(dataStore, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal data: %v", err)
	}

	if err := ioutil.WriteFile(DataFile, data, 0644); err != nil {
		return fmt.Errorf("failed to write data file: %v", err)
	}

	log.Printf("Saved data store with %d cameras and %d inference servers", len(dataStore.Cameras), len(dataStore.InferenceServers))
	return nil
}

func generateCameraID() string {
	dataStore.Counters.Camera++
	return fmt.Sprintf("cam_%d", dataStore.Counters.Camera)
}

func generateInferenceServerID() string {
	dataStore.Counters.InferenceServer++
	return fmt.Sprintf("inf_%d", dataStore.Counters.InferenceServer)
}

// API Handlers
func (ws *WebServer) handleAPICameras(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	switch r.Method {
	case "GET":
		var cameraList []*CameraConfig
		for _, camera := range dataStore.Cameras {
			cameraList = append(cameraList, camera)
		}

		response := APIResponse{
			Success: true,
			Message: "Cameras retrieved successfully",
			Data:    cameraList,
		}
		json.NewEncoder(w).Encode(response)

	case "POST":
		var newCamera CameraConfig
		if err := json.NewDecoder(r.Body).Decode(&newCamera); err != nil {
			response := APIResponse{
				Success: false,
				Message: "Invalid request body",
				Error:   err.Error(),
			}
			w.WriteHeader(http.StatusBadRequest)
			json.NewEncoder(w).Encode(response)
			return
		}

		if newCamera.Name == "" || newCamera.RTSPUrl == "" {
			response := APIResponse{
				Success: false,
				Message: "Name and RTSP URL are required",
			}
			w.WriteHeader(http.StatusBadRequest)
			json.NewEncoder(w).Encode(response)
			return
		}

		if newCamera.ID == "" {
			newCamera.ID = generateCameraID()
		}

		newCamera.CreatedAt = time.Now()
		newCamera.UpdatedAt = time.Now()
		// Auto-start cameras when they are created
		newCamera.Running = true
		newCamera.Enabled = true

		dataStore.Cameras[newCamera.ID] = &newCamera

		if err := saveDataStore(); err != nil {
			log.Printf("Warning: Failed to save data store: %v", err)
		}

		// Actually start the RTSP stream processing
		if ws.rtspManager != nil {
			if err := ws.rtspManager.StartCamera(&newCamera); err != nil {
				log.Printf("Warning: Failed to start RTSP stream for camera %s: %v", newCamera.ID, err)
				// Don't fail the API call, just log the warning
			}
		}

		log.Printf("Created camera: %s (%s)", newCamera.ID, newCamera.Name)

		response := APIResponse{
			Success: true,
			Message: "Camera created successfully",
			Data:    &newCamera,
		}

		w.WriteHeader(http.StatusCreated)
		json.NewEncoder(w).Encode(response)
	}
}

func (ws *WebServer) handleAPICameraByID(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	vars := mux.Vars(r)
	id := vars["id"]

	camera, exists := dataStore.Cameras[id]
	if !exists {
		response := APIResponse{
			Success: false,
			Message: "Camera not found",
		}
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(response)
		return
	}

	switch r.Method {
	case "GET":
		response := APIResponse{
			Success: true,
			Message: "Camera retrieved successfully",
			Data:    camera,
		}
		json.NewEncoder(w).Encode(response)

	case "PUT":
		var updatedCamera CameraConfig
		if err := json.NewDecoder(r.Body).Decode(&updatedCamera); err != nil {
			response := APIResponse{
				Success: false,
				Message: "Invalid request body",
				Error:   err.Error(),
			}
			w.WriteHeader(http.StatusBadRequest)
			json.NewEncoder(w).Encode(response)
			return
		}

		updatedCamera.ID = id
		updatedCamera.CreatedAt = camera.CreatedAt
		updatedCamera.UpdatedAt = time.Now()

		dataStore.Cameras[id] = &updatedCamera

		if err := saveDataStore(); err != nil {
			log.Printf("Warning: Failed to save data store: %v", err)
		}

		log.Printf("Updated camera: %s", id)

		response := APIResponse{
			Success: true,
			Message: "Camera updated successfully",
			Data:    &updatedCamera,
		}
		json.NewEncoder(w).Encode(response)

	case "DELETE":
		// Stop RTSP stream first
		if ws.rtspManager != nil {
			if err := ws.rtspManager.StopCamera(id); err != nil {
				log.Printf("Warning: Failed to stop RTSP stream for camera %s: %v", id, err)
			}
		}

		delete(dataStore.Cameras, id)

		if err := saveDataStore(); err != nil {
			log.Printf("Warning: Failed to save data store: %v", err)
		}

		log.Printf("Deleted camera: %s", id)

		response := APIResponse{
			Success: true,
			Message: "Camera deleted successfully",
		}
		json.NewEncoder(w).Encode(response)
	}
}

func (ws *WebServer) handleAPIStartCamera(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	vars := mux.Vars(r)
	id := vars["id"]

	camera, exists := dataStore.Cameras[id]
	if !exists {
		response := APIResponse{
			Success: false,
			Message: "Camera not found",
		}
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(response)
		return
	}

	camera.Running = true
	camera.Enabled = true
	camera.UpdatedAt = time.Now()

	// Start fall detection tasks for any bound fall detection servers
	for _, binding := range camera.InferenceServerBindings {
		server, serverExists := dataStore.InferenceServers[binding.ServerID]
		if serverExists && server.Enabled && server.ModelType == "fall" {
			ws.startFallDetectionTask(camera, server)
		}
	}

	if err := saveDataStore(); err != nil {
		slog.Warn(fmt.Sprintf("Warning: Failed to save data store: %v", err))
	}

	slog.Info(fmt.Sprintf("started camera: %s", id))

	response := APIResponse{
		Success: true,
		Message: "Camera started successfully",
		Data:    camera,
	}
	json.NewEncoder(w).Encode(response)
}

func (ws *WebServer) handleAPIStopCamera(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	vars := mux.Vars(r)
	id := vars["id"]

	camera, exists := dataStore.Cameras[id]
	if !exists {
		response := APIResponse{
			Success: false,
			Message: "Camera not found",
		}
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(response)
		return
	}

	camera.Running = false
	camera.UpdatedAt = time.Now()

	// Stop fall detection tasks for any bound fall detection servers
	for _, binding := range camera.InferenceServerBindings {
		server, serverExists := dataStore.InferenceServers[binding.ServerID]
		if serverExists && server.ModelType == "fall" {
			ws.stopFallDetectionTasksForCamera(camera.ID, server.ID)
		}
	}

	if err := saveDataStore(); err != nil {
		log.Printf("Warning: Failed to save data store: %v", err)
	}

	log.Printf("Stopped camera: %s", id)

	response := APIResponse{
		Success: true,
		Message: "Camera stopped successfully",
		Data:    camera,
	}
	json.NewEncoder(w).Encode(response)
}

func (ws *WebServer) handleAPIStatus(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	runningCount := 0
	enabledCount := 0

	for _, camera := range dataStore.Cameras {
		if camera.Running {
			runningCount++
		}
		if camera.Enabled {
			enabledCount++
		}
	}

	response := APIResponse{
		Success: true,
		Message: "System status retrieved successfully",
		Data: map[string]interface{}{
			"manager_running":    true,
			"running_cameras":    runningCount,
			"total_cameras":      len(dataStore.Cameras),
			"enabled_cameras":    enabledCount,
			"disabled_cameras":   len(dataStore.Cameras) - enabledCount,
			"persistent_storage": true,
			"data_file":          DataFile,
		},
	}

	json.NewEncoder(w).Encode(response)
}

func (ws *WebServer) handleAPIDebug(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	var cameraIDs []string
	for id := range dataStore.Cameras {
		cameraIDs = append(cameraIDs, id)
	}

	debugInfo := map[string]interface{}{
		"message":            "Debug route is working!",
		"timestamp":          time.Now().Format(time.RFC3339),
		"routes_registered":  "API routes are properly registered",
		"cors_enabled":       true,
		"total_cameras":      len(dataStore.Cameras),
		"camera_ids":         cameraIDs,
		"persistent_storage": true,
		"data_file_exists":   fileExists(DataFile),
		"request_method":     r.Method,
		"request_path":       r.URL.Path,
	}

	response := APIResponse{
		Success: true,
		Message: "Debug information retrieved successfully",
		Data:    debugInfo,
	}

	json.NewEncoder(w).Encode(response)
}

// Image management structures
type ImageInfo struct {
	Filename  string    `json:"filename"`
	Size      int64     `json:"size"`
	CreatedAt time.Time `json:"created_at"`
}

type ImageListResponse struct {
	Images      []ImageInfo `json:"images"`
	TotalCount  int         `json:"total_count"`
	TotalPages  int         `json:"total_pages"`
	CurrentPage int         `json:"current_page"`
}

// handleAPIImages returns paginated list of images for a specific camera
func (ws *WebServer) handleAPIImages(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	vars := mux.Vars(r)
	cameraId := vars["cameraId"]

	// Check if camera exists
	if _, exists := dataStore.Cameras[cameraId]; !exists {
		response := APIResponse{
			Success: false,
			Message: "Camera not found",
		}
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(response)
		return
	}

	// Parse query parameters for pagination
	pageStr := r.URL.Query().Get("page")
	limitStr := r.URL.Query().Get("limit")

	page := 1
	limit := 24 // Default page size

	if pageStr != "" {
		if p, err := strconv.Atoi(pageStr); err == nil && p > 0 {
			page = p
		}
	}

	if limitStr != "" {
		if l, err := strconv.Atoi(limitStr); err == nil && l > 0 && l <= 100 {
			limit = l
		}
	}

	// Get images from camera directory
	cameraDir := filepath.Join(ws.outputDir, cameraId)
	images, err := getImagesFromDirectory(cameraDir)
	if err != nil {
		response := APIResponse{
			Success: false,
			Message: "Failed to read camera images",
			Error:   err.Error(),
		}
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(response)
		return
	}

	// Sort images by creation time (newest first)
	sort.Slice(images, func(i, j int) bool {
		return images[i].CreatedAt.After(images[j].CreatedAt)
	})

	// Calculate pagination
	totalCount := len(images)
	totalPages := (totalCount + limit - 1) / limit
	if totalPages == 0 {
		totalPages = 1
	}

	// Get page slice
	start := (page - 1) * limit
	end := start + limit
	if start >= totalCount {
		images = []ImageInfo{}
	} else {
		if end > totalCount {
			end = totalCount
		}
		images = images[start:end]
	}

	responseData := ImageListResponse{
		Images:      images,
		TotalCount:  totalCount,
		TotalPages:  totalPages,
		CurrentPage: page,
	}

	response := APIResponse{
		Success: true,
		Message: "Images retrieved successfully",
		Data:    responseData,
	}

	json.NewEncoder(w).Encode(response)
}

// handleAPIImageFile serves individual image files
func (ws *WebServer) handleAPIImageFile(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	cameraId := vars["cameraId"]
	filename := vars["filename"]

	// Check if camera exists
	if _, exists := dataStore.Cameras[cameraId]; !exists {
		http.Error(w, "Camera not found", http.StatusNotFound)
		return
	}

	// Validate filename (security check)
	if strings.Contains(filename, "..") || strings.Contains(filename, "/") || strings.Contains(filename, "\\") {
		http.Error(w, "Invalid filename", http.StatusBadRequest)
		return
	}

	// Construct file path
	filePath := filepath.Join(ws.outputDir, cameraId, filename)

	// Check if file exists
	if !fileExists(filePath) {
		http.Error(w, "Image not found", http.StatusNotFound)
		return
	}

	// Set appropriate headers
	w.Header().Set("Content-Type", "image/jpeg")
	w.Header().Set("Cache-Control", "public, max-age=3600") // Cache for 1 hour

	// Serve the file
	http.ServeFile(w, r, filePath)
}

// getImagesFromDirectory reads all JPEG images from a directory and returns their info
func getImagesFromDirectory(dir string) ([]ImageInfo, error) {
	var images []ImageInfo

	// Check if directory exists
	if _, err := os.Stat(dir); os.IsNotExist(err) {
		return images, nil // Return empty slice, not an error
	}

	// Read directory
	files, err := ioutil.ReadDir(dir)
	if err != nil {
		return nil, fmt.Errorf("failed to read directory: %v", err)
	}

	// Filter and collect JPEG files
	for _, file := range files {
		if file.IsDir() {
			continue
		}

		// Check if it's a JPEG file
		filename := file.Name()
		lowerName := strings.ToLower(filename)
		if !strings.HasSuffix(lowerName, ".jpg") && !strings.HasSuffix(lowerName, ".jpeg") {
			continue
		}

		images = append(images, ImageInfo{
			Filename:  filename,
			Size:      file.Size(),
			CreatedAt: file.ModTime(),
		})
	}

	return images, nil
}

func fileExists(filename string) bool {
	_, err := os.Stat(filename)
	return !os.IsNotExist(err)
}

// Inference Server API Handlers
func (ws *WebServer) handleAPIInferenceServers(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	switch r.Method {
	case "GET":
		var serverList []*InferenceServer
		for _, server := range dataStore.InferenceServers {
			serverList = append(serverList, server)
		}

		response := APIResponse{
			Success: true,
			Message: "Inference servers retrieved successfully",
			Data:    serverList,
		}
		json.NewEncoder(w).Encode(response)

	case "POST":
		var newServer InferenceServer
		if err := json.NewDecoder(r.Body).Decode(&newServer); err != nil {
			response := APIResponse{
				Success: false,
				Message: "Invalid request body",
				Error:   err.Error(),
			}
			w.WriteHeader(http.StatusBadRequest)
			json.NewEncoder(w).Encode(response)
			return
		}

		if newServer.Name == "" || newServer.URL == "" {
			response := APIResponse{
				Success: false,
				Message: "Name and URL are required",
			}
			w.WriteHeader(http.StatusBadRequest)
			json.NewEncoder(w).Encode(response)
			return
		}

		// Set default model type if not provided
		if newServer.ModelType == "" {
			newServer.ModelType = "auto"
		}

		if newServer.ID == "" {
			newServer.ID = generateInferenceServerID()
		}

		newServer.CreatedAt = time.Now()
		newServer.UpdatedAt = time.Now()
		newServer.Enabled = true // Default to enabled

		dataStore.InferenceServers[newServer.ID] = &newServer

		if err := saveDataStore(); err != nil {
			log.Printf("Warning: Failed to save data store: %v", err)
		}

		log.Printf("Created inference server: %s (%s)", newServer.ID, newServer.Name)

		response := APIResponse{
			Success: true,
			Message: "Inference server created successfully",
			Data:    &newServer,
		}

		w.WriteHeader(http.StatusCreated)
		json.NewEncoder(w).Encode(response)
	}
}

func (ws *WebServer) handleAPIInferenceServerByID(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	vars := mux.Vars(r)
	id := vars["id"]

	server, exists := dataStore.InferenceServers[id]
	if !exists {
		response := APIResponse{
			Success: false,
			Message: "Inference server not found",
		}
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(response)
		return
	}

	switch r.Method {
	case "GET":
		response := APIResponse{
			Success: true,
			Message: "Inference server retrieved successfully",
			Data:    server,
		}
		json.NewEncoder(w).Encode(response)

	case "PUT":
		var updatedServer InferenceServer
		if err := json.NewDecoder(r.Body).Decode(&updatedServer); err != nil {
			response := APIResponse{
				Success: false,
				Message: "Invalid request body",
				Error:   err.Error(),
			}
			w.WriteHeader(http.StatusBadRequest)
			json.NewEncoder(w).Encode(response)
			return
		}

		updatedServer.ID = id
		updatedServer.CreatedAt = server.CreatedAt
		updatedServer.UpdatedAt = time.Now()

		dataStore.InferenceServers[id] = &updatedServer

		if err := saveDataStore(); err != nil {
			log.Printf("Warning: Failed to save data store: %v", err)
		}

		log.Printf("Updated inference server: %s", id)

		response := APIResponse{
			Success: true,
			Message: "Inference server updated successfully",
			Data:    &updatedServer,
		}
		json.NewEncoder(w).Encode(response)

	case "DELETE":
		for _, camera := range dataStore.Cameras {
			for i, serverBinding := range camera.InferenceServerBindings {
				if serverBinding.ServerID == id {
					camera.InferenceServerBindings = append(camera.InferenceServerBindings[:i], camera.InferenceServerBindings[i+1:]...)
					camera.UpdatedAt = time.Now()
					break
				}
			}
		}

		delete(dataStore.InferenceServers, id)

		if err := saveDataStore(); err != nil {
			log.Printf("Warning: Failed to save data store: %v", err)
		}

		log.Printf("Deleted inference server: %s", id)

		response := APIResponse{
			Success: true,
			Message: "Inference server deleted successfully",
		}
		json.NewEncoder(w).Encode(response)
	}
}

// Alert Server API Handler
func (ws *WebServer) handleAPIAlertServer(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	switch r.Method {
	case "GET":
		// Return current alert server configuration
		var alertConfig *AlertServerConfig
		if dataStore.AlertServer != nil {
			alertConfig = dataStore.AlertServer
		} else {
			// Return default/empty configuration
			alertConfig = &AlertServerConfig{
				URL:       "",
				Enabled:   false,
				UpdatedAt: time.Now(),
			}
		}

		response := APIResponse{
			Success: true,
			Message: "Alert server configuration retrieved successfully",
			Data:    alertConfig,
		}
		json.NewEncoder(w).Encode(response)

	case "PUT":
		// Update alert server configuration
		var updatedConfig AlertServerConfig
		if err := json.NewDecoder(r.Body).Decode(&updatedConfig); err != nil {
			response := APIResponse{
				Success: false,
				Message: "Invalid request body",
				Error:   err.Error(),
			}
			w.WriteHeader(http.StatusBadRequest)
			json.NewEncoder(w).Encode(response)
			return
		}

		updatedConfig.UpdatedAt = time.Now()
		dataStore.AlertServer = &updatedConfig

		if err := saveDataStore(); err != nil {
			log.Printf("Warning: Failed to save data store: %v", err)
		}

		log.Printf("Updated alert server configuration: URL=%s, Enabled=%t", updatedConfig.URL, updatedConfig.Enabled)

		response := APIResponse{
			Success: true,
			Message: "Alert server configuration updated successfully",
			Data:    &updatedConfig,
		}
		json.NewEncoder(w).Encode(response)
	}
}

// handleAlerts serves the alert configuration page
func (ws *WebServer) handleAlerts(w http.ResponseWriter, r *http.Request) {
	content := `
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>告警配置 - Cam-Stream 平台</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
            background-color: #f8f9fa;
            color: #333;
            margin: 0;
            padding: 20px;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 30px;
            padding: 20px 0;
            border-bottom: 2px solid #e9ecef;
        }
        .header h1 {
            margin: 0;
            font-size: 2rem;
            color: #2c3e50;
        }
        .nav-link {
            text-decoration: none;
            color: #6c757d;
            padding: 8px 16px;
            border-radius: 4px;
            transition: background-color 0.2s;
        }
        .nav-link:hover {
            background-color: #e9ecef;
        }
        .alert-config {
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            padding: 30px;
            margin-bottom: 30px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #495057;
        }
        .form-group input[type="text"], .form-group input[type="url"] {
            width: 100%;
            padding: 12px;
            border: 2px solid #e9ecef;
            border-radius: 6px;
            font-size: 14px;
            transition: border-color 0.2s;
            box-sizing: border-box;
        }
        .form-group input:focus {
            outline: none;
            border-color: #3498db;
        }
        .checkbox-group {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .checkbox-group input[type="checkbox"] {
            width: 18px;
            height: 18px;
        }
        .btn {
            background-color: #3498db;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            transition: background-color 0.2s;
        }
        .btn:hover {
            background-color: #2980b9;
        }
        .btn:disabled {
            background-color: #95a5a6;
            cursor: not-allowed;
        }
        .status {
            padding: 12px 16px;
            border-radius: 6px;
            margin-bottom: 20px;
            display: none;
        }
        .status.success {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .status.error {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .info-box {
            background-color: #e7f3ff;
            border: 1px solid #b8daff;
            border-radius: 6px;
            padding: 16px;
            margin-bottom: 20px;
        }
        .info-box h3 {
            margin: 0 0 10px 0;
            color: #004085;
        }
        .info-box p {
            margin: 5px 0;
            color: #004085;
        }
        .loading {
            display: none;
            color: #6c757d;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>告警配置</h1>
            <a href="/" class="nav-link">← 返回仪表板</a>
        </div>

        <div class="info-box">
            <h3>全局告警服务器配置</h3>
            <p>配置全局告警服务器，用于接收所有摄像头的检测告警。</p>
            <p>启用后，检测告警将根据API规范发送到配置的URL地址。</p>
        </div>

        <div class="alert-config">
            <div class="status" id="status"></div>
            
            <form id="alertForm">
                <div class="form-group">
                    <label for="alertUrl">告警平台URL：</label>
                    <input type="url" id="alertUrl" placeholder="http://localhost:8080/alert" required>
                </div>

                <div class="form-group">
                    <div class="checkbox-group">
                        <input type="checkbox" id="alertEnabled">
                        <label for="alertEnabled">启用告警系统</label>
                    </div>
                </div>

                <button type="submit" class="btn" id="saveBtn">
                    <span class="loading" id="loading">保存中...</span>
                    <span id="saveText">保存配置</span>
                </button>
            </form>
        </div>
    </div>

    <script>
        const alertForm = document.getElementById('alertForm');
        const statusDiv = document.getElementById('status');
        const saveBtn = document.getElementById('saveBtn');
        const loading = document.getElementById('loading');
        const saveText = document.getElementById('saveText');

        // 页面加载时获取当前配置
        async function loadConfiguration() {
            try {
                const response = await fetch('/api/alert-server');
                const result = await response.json();
                
                if (result.success) {
                    const config = result.data;
                    document.getElementById('alertUrl').value = config.url || '';
                    document.getElementById('alertEnabled').checked = config.enabled || false;
                }
            } catch (error) {
                showStatus('加载当前配置失败', 'error');
            }
        }

        // 保存配置
        alertForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const alertUrl = document.getElementById('alertUrl').value;
            const alertEnabled = document.getElementById('alertEnabled').checked;

            // 显示加载状态
            saveBtn.disabled = true;
            loading.style.display = 'inline';
            saveText.style.display = 'none';

            try {
                const response = await fetch('/api/alert-server', {
                    method: 'PUT',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        url: alertUrl,
                        enabled: alertEnabled
                    })
                });

                const result = await response.json();
                
                if (result.success) {
                    showStatus('告警配置保存成功！', 'success');
                } else {
                    showStatus('保存配置失败：' + (result.error || result.message), 'error');
                }
            } catch (error) {
                showStatus('网络错误：' + error.message, 'error');
            } finally {
                // 重置加载状态
                saveBtn.disabled = false;
                loading.style.display = 'none';
                saveText.style.display = 'inline';
            }
        });

        function showStatus(message, type) {
            statusDiv.textContent = message;
            statusDiv.className = 'status ' + type;
            statusDiv.style.display = 'block';
            
            // 5秒后隐藏状态提示
            setTimeout(() => {
                statusDiv.style.display = 'none';
            }, 5000);
        }

        // 页面加载时获取配置
        loadConfiguration();
    </script>
</body>
</html>
	`
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write([]byte(content))
}

// Fall Detection Task Management Functions

// startFallDetectionTask starts a fall detection task for a camera with a fall detection server
func (ws *WebServer) startFallDetectionTask(camera *CameraConfig, server *InferenceServer) {
	// Check if there's already a running task for this camera-server combination
	for _, task := range fallDetectionTasks {
		if task.CameraID == camera.ID && task.ServerID == server.ID && task.Status == "running" {
			log.Printf("fall detection task already running for camera %s with server %s (task: %s)", camera.ID, server.ID, task.TaskID)
			return
		}
	}

	// Start the fall detection task
	taskID, err := StartFallDetection(server, camera)
	if err != nil {
		log.Printf("failed to start fall detection task for camera %s with server %s: %v", camera.ID, server.ID, err)
		// Create error task state
		fallDetectionTasks["error_"+camera.ID+"_"+server.ID] = &FallDetectionTaskState{
			TaskID:    "",
			CameraID:  camera.ID,
			ServerID:  server.ID,
			Status:    "error",
			StartedAt: time.Now(),
			UpdatedAt: time.Now(),
			ErrorMsg:  err.Error(),
		}
		return
	}

	// Create successful task state using the returned taskID as key
	fallDetectionTasks[taskID] = &FallDetectionTaskState{
		TaskID:    taskID,
		CameraID:  camera.ID,
		ServerID:  server.ID,
		Status:    "running",
		StartedAt: time.Now(),
		UpdatedAt: time.Now(),
	}

	log.Printf("started fall detection task %s for camera %s with server %s", taskID, camera.ID, server.ID)
}

// stopFallDetectionTasksForCamera stops all fall detection tasks for a specific camera-server combination
func (ws *WebServer) stopFallDetectionTasksForCamera(cameraID, serverID string) {
	var tasksToStop []string

	// Find all running tasks for this camera-server combination
	for taskID, task := range fallDetectionTasks {
		if task.CameraID == cameraID && task.ServerID == serverID && task.Status == "running" {
			tasksToStop = append(tasksToStop, taskID)
		}
	}

	if len(tasksToStop) == 0 {
		log.Printf("no running fall detection tasks found for camera %s with server %s", cameraID, serverID)
		return
	}

	server, serverExists := dataStore.InferenceServers[serverID]
	if !serverExists {
		log.Printf("inference server %s not found when stopping fall detection tasks", serverID)
		return
	}

	// Stop each task
	for _, taskID := range tasksToStop {
		task := fallDetectionTasks[taskID]
		err := StopFallDetection(server, taskID)
		if err != nil {
			log.Printf("failed to stop fall detection task %s: %v", taskID, err)
			// Update task state to error
			task.Status = "error"
			task.ErrorMsg = err.Error()
			task.UpdatedAt = time.Now()
		} else {
			// Update task state to stopped
			task.Status = "stopped"
			task.UpdatedAt = time.Now()
			log.Printf("stopped fall detection task %s for camera %s with server %s", taskID, cameraID, serverID)
		}
	}
}
