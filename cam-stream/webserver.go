package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"gocv.io/x/gocv"
	"github.com/gorilla/mux"
)

// HTMLTemplates holds cached HTML content
type HTMLTemplates struct {
	index           []byte
	cameraManagement []byte
	imageViewer     []byte
	alerts          []byte
}

// WebServer handles web interface
type WebServer struct {
	outputDir   string
	port        int
	rtspManager *RTSPManager
	templates   *HTMLTemplates
}

// loadHTMLTemplates loads all HTML files into memory at startup
func loadHTMLTemplates() (*HTMLTemplates, error) {
	templates := &HTMLTemplates{}
	
	// Load index.html
	if data, err := ioutil.ReadFile("index.html"); err != nil {
		return nil, fmt.Errorf("failed to load index.html: %v", err)
	} else {
		templates.index = data
	}
	
	// Load camera_management.html
	if data, err := ioutil.ReadFile("camera_management.html"); err != nil {
		return nil, fmt.Errorf("failed to load camera_management.html: %v", err)
	} else {
		templates.cameraManagement = data
	}
	
	// Load image_viewer.html
	if data, err := ioutil.ReadFile("image_viewer.html"); err != nil {
		return nil, fmt.Errorf("failed to load image_viewer.html: %v", err)
	} else {
		templates.imageViewer = data
	}
	
	// Load alerts.html
	if data, err := ioutil.ReadFile("alerts.html"); err != nil {
		return nil, fmt.Errorf("failed to load alerts.html: %v", err)
	} else {
		templates.alerts = data
	}
	
	AsyncInfo("successfully loaded all HTML templates into memory")
	return templates, nil
}

// NewWebServer creates a new web server
func NewWebServer(outputDir string, port int) *WebServer {
	templates, err := loadHTMLTemplates()
	if err != nil {
		AsyncWarn(fmt.Sprintf("failed to load HTML templates: %v", err))
	}
	
	return &WebServer{
		outputDir: outputDir,
		port:      port,
		templates: templates,
	}
}

// SetRTSPManager sets the RTSP manager for camera operations
func (ws *WebServer) SetRTSPManager(manager *RTSPManager) {
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

	// Inference Server API Routes
	api.HandleFunc("/inference-servers", ws.handleAPIInferenceServers).Methods("GET", "POST", "OPTIONS")
	api.HandleFunc("/inference-servers/{id}", ws.handleAPIInferenceServerByID).Methods("GET", "PUT", "DELETE", "OPTIONS")

	// Alert Server API Routes
	api.HandleFunc("/alert-server", ws.handleAPIAlertServer).Methods("GET", "PUT", "OPTIONS")

	api.HandleFunc("/status", ws.handleAPIStatus).Methods("GET", "OPTIONS")
	api.HandleFunc("/debug", ws.handleAPIDebug).Methods("GET", "OPTIONS")
	api.HandleFunc("/ping", ws.handleAPIPing).Methods("GET", "OPTIONS")

	// Image management API routes
	api.HandleFunc("/images/{cameraId}", ws.handleAPIImages).Methods("GET", "OPTIONS")
	api.HandleFunc("/images/{cameraId}/{filename}", ws.handleAPIImageFile).Methods("GET", "OPTIONS")

	// Server-based image API routes
	api.HandleFunc("/server-images", ws.handleAPIServerImages).Methods("GET", "OPTIONS")

	// Web Routes
	router.HandleFunc("/", ws.handleIndex).Methods("GET")
	router.HandleFunc("/cameras", ws.handleCameraManagement).Methods("GET")
	router.HandleFunc("/images", ws.handleImages).Methods("GET")
	router.HandleFunc("/alerts", ws.handleAlerts).Methods("GET")

	// Static file server for output directory (images)
	router.PathPrefix("/output/").Handler(http.StripPrefix("/output/", http.FileServer(http.Dir("output/"))))

	// Static file server for HTML files - MUST be LAST as it's a catch-all
	router.PathPrefix("/").Handler(http.FileServer(http.Dir("./")))

	AsyncInfo(fmt.Sprintf("starting web server on port %d", ws.port))
	AsyncInfo(fmt.Sprintf("access web interface at: http://localhost:%d", ws.port))

	return http.ListenAndServe(fmt.Sprintf(":%d", ws.port), router)
}

// handleIndex serves the main page with links to different sections
func (ws *WebServer) handleIndex(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write(ws.templates.index)
}

// handleCameraManagement serves the camera management page
func (ws *WebServer) handleCameraManagement(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write(ws.templates.cameraManagement)
}

// handleImages serves the image viewer page
func (ws *WebServer) handleImages(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write(ws.templates.imageViewer)
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
			AsyncInfo("data file not found, starting with empty store")
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

	AsyncInfo(fmt.Sprintf("loaded %d cameras and %d inference servers from storage", len(dataStore.Cameras), len(dataStore.InferenceServers)))

	// Log alert server configuration status
	if dataStore.AlertServer != nil {
		if dataStore.AlertServer.Enabled {
			AsyncInfo(fmt.Sprintf("alert server configured and enabled: %s", dataStore.AlertServer.URL))
		} else {
			AsyncInfo(fmt.Sprintf("alert server configured but disabled: %s", dataStore.AlertServer.URL))
		}
	} else {
		AsyncInfo("alert server not configured - alerts disabled")
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

	AsyncInfo(fmt.Sprintf("saved data store with %d cameras and %d inference servers", len(dataStore.Cameras), len(dataStore.InferenceServers)))
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
			AsyncWarn(fmt.Sprintf("failed to save data store: %v", err))
		}

		// Actually start the RTSP stream processing
		if ws.rtspManager != nil {
			if err := ws.rtspManager.StartCamera(&newCamera); err != nil {
				AsyncWarn(fmt.Sprintf("failed to start RTSP stream for camera %s: %v", newCamera.ID, err))
				// Don't fail the API call, just log the warning
			}
		}

		// Start fall detection tasks for any bound fall detection servers
		for _, binding := range newCamera.InferenceServerBindings {
			server, serverExists := dataStore.InferenceServers[binding.ServerID]
			if serverExists && server.Enabled && server.ModelType == "fall" {
				ws.startFallDetectionTask(&newCamera, server)
			}
		}

		AsyncInfo(fmt.Sprintf("created camera: %s (%s)", newCamera.ID, newCamera.Name))

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
			AsyncWarn(fmt.Sprintf("failed to save data store: %v", err))
		}

		AsyncInfo(fmt.Sprintf("updated camera: %s", id))

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
				AsyncWarn(fmt.Sprintf("failed to stop RTSP stream for camera %s: %v", id, err))
			}
		}

		// Stop fall detection tasks for this camera
		for _, binding := range camera.InferenceServerBindings {
			server, serverExists := dataStore.InferenceServers[binding.ServerID]
			if serverExists && server.ModelType == "fall" {
				ws.stopFallDetectionTasksForCamera(camera.ID, server.ID)
			}
		}

		delete(dataStore.Cameras, id)

		if err := saveDataStore(); err != nil {
			AsyncWarn(fmt.Sprintf("failed to save data store: %v", err))
		}

		AsyncInfo(fmt.Sprintf("deleted camera: %s", id))

		response := APIResponse{
			Success: true,
			Message: "Camera deleted successfully",
		}
		json.NewEncoder(w).Encode(response)
	}
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

// handleAPIPing returns basic liveness and GoCV/OpenCV build info
func (ws *WebServer) handleAPIPing(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	info := map[string]interface{}{
		"timestamp":      time.Now().Format(time.RFC3339),
		"gocv_version":   gocv.Version(),
		"opencv_version": gocv.OpenCVVersion(),
	}

	resp := APIResponse{
		Success: true,
		Message: "pong",
		Data:    info,
	}
	json.NewEncoder(w).Encode(resp)
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
	w.Header().Set("Cache-Control", "no-cache, no-store, must-revalidate") // No cache for real-time monitoring
	w.Header().Set("Pragma", "no-cache")
	w.Header().Set("Expires", "0")

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
		newServer.Enabled = true

		dataStore.InferenceServers[newServer.ID] = &newServer

		if err := saveDataStore(); err != nil {
			AsyncWarn(fmt.Sprintf("failed to save data store: %v", err))
		}

		AsyncInfo(fmt.Sprintf("created inference server: %s (%s)", newServer.ID, newServer.Name))

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
			AsyncWarn(fmt.Sprintf("failed to save data store: %v", err))
		}

		AsyncInfo(fmt.Sprintf("updated inference server: %s", id))

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
			AsyncWarn(fmt.Sprintf("failed to save data store: %v", err))
		}

		AsyncInfo(fmt.Sprintf("deleted inference server: %s", id))

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
			AsyncWarn(fmt.Sprintf("failed to save data store: %v", err))
		}

		AsyncInfo(fmt.Sprintf("updated alert server configuration: URL=%s, Enabled=%t", updatedConfig.URL, updatedConfig.Enabled))

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
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write(ws.templates.alerts)
}

// Fall Detection Task Management Functions

// startFallDetectionTask starts a fall detection task for a camera with a fall detection server
func (ws *WebServer) startFallDetectionTask(camera *CameraConfig, server *InferenceServer) {
	// Check if there's already a running task for this camera-server combination
	for _, task := range fallDetectionTasks {
		if task.CameraID == camera.ID && task.ServerID == server.ID && task.Status == "running" {
			AsyncInfo(fmt.Sprintf("fall detection task already running for camera %s with server %s (task: %s)", camera.ID, server.ID, task.TaskID))
			return
		}
	}

	// Start the fall detection task
	taskID, err := StartFallDetection(server, camera)
	if err != nil {
		AsyncWarn(fmt.Sprintf("failed to start fall detection task for camera %s with server %s: %v", camera.ID, server.ID, err))
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

	AsyncInfo(fmt.Sprintf("started fall detection task %s for camera %s with server %s", taskID, camera.ID, server.ID))
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
			AsyncInfo(fmt.Sprintf("no running fall detection tasks found for camera %s with server %s", cameraID, serverID))
		return
	}

	server, serverExists := dataStore.InferenceServers[serverID]
	if !serverExists {
		AsyncWarn(fmt.Sprintf("inference server %s not found when stopping fall detection tasks", serverID))
		return
	}

	// Stop each task
	for _, taskID := range tasksToStop {
		task := fallDetectionTasks[taskID]
		err := StopFallDetection(server, taskID)
			if err != nil {
				AsyncWarn(fmt.Sprintf("failed to stop fall detection task %s: %v", taskID, err))
				// Update task state to error
			task.Status = "error"
			task.ErrorMsg = err.Error()
			task.UpdatedAt = time.Now()
		} else {
			// Update task state to stopped
			task.Status = "stopped"
			task.UpdatedAt = time.Now()
				AsyncInfo(fmt.Sprintf("stopped fall detection task %s for camera %s with server %s", taskID, cameraID, serverID))
		}
	}
}
