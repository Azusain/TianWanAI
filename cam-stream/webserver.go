package main

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/gorilla/mux"
)

// HTMLTemplates holds cached HTML content
type HTMLTemplates struct {
	index            []byte
	cameraManagement []byte
	imageViewer      []byte
	alerts           []byte
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

	templatesDir := "templates"
	// Load index.html
	if data, err := ioutil.ReadFile(filepath.Join(templatesDir, "index.html")); err != nil {
		return nil, fmt.Errorf("failed to load index.html: %v", err)
	} else {
		templates.index = data
	}

	// Load camera_management.html
	if data, err := ioutil.ReadFile(filepath.Join(templatesDir, "camera_management.html")); err != nil {
		return nil, fmt.Errorf("failed to load camera_management.html: %v", err)
	} else {
		templates.cameraManagement = data
	}

	// Load image_viewer.html
	if data, err := ioutil.ReadFile(filepath.Join(templatesDir, "image_viewer.html")); err != nil {
		return nil, fmt.Errorf("failed to load image_viewer.html: %v", err)
	} else {
		templates.imageViewer = data
	}

	// Load alerts.html
	if data, err := ioutil.ReadFile(filepath.Join(templatesDir, "alerts.html")); err != nil {
		return nil, fmt.Errorf("failed to load alerts.html: %v", err)
	} else {
		templates.alerts = data
	}

	Info("successfully loaded all HTML templates into memory")
	return templates, nil
}

// NewWebServer creates a new web server
func NewWebServer(outputDir string, port int) *WebServer {
	templates, err := loadHTMLTemplates()
	if err != nil {
		Warn(fmt.Sprintf("failed to load HTML templates: %v", err))
		// ensure non-nil to avoid nil dereference; actual fix is to include templates in runtime image
		templates = &HTMLTemplates{}
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

	// Image servers API routes (by inference server ID)
	api.HandleFunc("/image-servers", ws.handleAPIImageServers).Methods("GET", "OPTIONS")
	api.HandleFunc("/server-images/{serverId}", ws.handleAPIServerImages).Methods("GET", "OPTIONS")

	// Config import/export routes
	api.HandleFunc("/config/export", ws.handleAPIConfigExport).Methods("GET", "OPTIONS")
	api.HandleFunc("/config/import", ws.handleAPIConfigImport).Methods("POST", "OPTIONS")

	// Web Routes
	router.HandleFunc("/", ws.handleIndex).Methods("GET")
	router.HandleFunc("/cameras", ws.handleCameraManagement).Methods("GET")
	router.HandleFunc("/images", ws.handleImages).Methods("GET")
	router.HandleFunc("/alerts", ws.handleAlerts).Methods("GET")

	// Static file server for output directory (images)
	router.PathPrefix("/output/").Handler(http.StripPrefix("/output/", http.FileServer(http.Dir("output/"))))

	// Static file server for HTML files - MUST be LAST as it's a catch-all
	router.PathPrefix("/").Handler(http.FileServer(http.Dir("./")))

	Info(fmt.Sprintf("starting web server on port %d", ws.port))
	Info(fmt.Sprintf("access web interface at: http://localhost:%d", ws.port))

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

	// Polling mechanism fields (not persisted to JSON)
	pollTicker   *time.Ticker `json:"-"` // Ticker for periodic result polling
	pollStopChan chan bool    `json:"-"` // Channel to stop the polling goroutine
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
			Info("data file not found, starting with empty store")
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

	Info(fmt.Sprintf("loaded %d cameras and %d inference servers from storage", len(dataStore.Cameras), len(dataStore.InferenceServers)))

	// Log alert server configuration status
	if dataStore.AlertServer != nil {
		if dataStore.AlertServer.Enabled {
			Info(fmt.Sprintf("alert server configured and enabled: %s", dataStore.AlertServer.URL))
		} else {
			Info(fmt.Sprintf("alert server configured but disabled: %s", dataStore.AlertServer.URL))
		}
	} else {
		Info("alert server not configured - alerts disabled")
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

	Info(fmt.Sprintf("saved data store with %d cameras and %d inference servers", len(dataStore.Cameras), len(dataStore.InferenceServers)))
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
			Warn(fmt.Sprintf("failed to save data store: %v", err))
		}

		// Actually start the RTSP stream processing
		if ws.rtspManager != nil {
			if err := ws.rtspManager.StartCamera(&newCamera); err != nil {
				Warn(fmt.Sprintf("failed to start RTSP stream for camera %s: %v", newCamera.ID, err))
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

		Info(fmt.Sprintf("created camera: %s (%s)", newCamera.ID, newCamera.Name))

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
			Warn(fmt.Sprintf("failed to save data store: %v", err))
		}

		Info(fmt.Sprintf("updated camera: %s", id))

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
				Warn(fmt.Sprintf("failed to stop RTSP stream for camera %s: %v", id, err))
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
			Warn(fmt.Sprintf("failed to save data store: %v", err))
		}

		Info(fmt.Sprintf("deleted camera: %s", id))

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

// handleAPIPing returns basic liveness and FFmpeg version info
func (ws *WebServer) handleAPIPing(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	// get FFmpeg version
	ffmpegVersion := "unknown"
	if cmd := exec.Command("ffmpeg", "-version"); cmd != nil {
		if output, err := cmd.Output(); err == nil {
			// extract first line which contains version info
			lines := strings.Split(string(output), "\n")
			if len(lines) > 0 {
				ffmpegVersion = strings.TrimSpace(lines[0])
			}
		}
	}

	// get ffprobe version
	ffprobeVersion := "unknown"
	if cmd := exec.Command("ffprobe", "-version"); cmd != nil {
		if output, err := cmd.Output(); err == nil {
			// extract first line which contains version info
			lines := strings.Split(string(output), "\n")
			if len(lines) > 0 {
				ffprobeVersion = strings.TrimSpace(lines[0])
			}
		}
	}

	info := map[string]interface{}{
		"timestamp":       time.Now().Format(time.RFC3339),
		"ffmpeg_version":  ffmpegVersion,
		"ffprobe_version": ffprobeVersion,
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

// ImageServer represents a server folder with optional friendly name
type ImageServer struct {
	ID   string `json:"id"`
	Name string `json:"name"`
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

// handleAPIImageServers lists server folders under the output directory
func (ws *WebServer) handleAPIImageServers(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	// read directories in outputDir
	entries, err := ioutil.ReadDir(ws.outputDir)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(APIResponse{Success: false, Message: "failed to read output directory", Error: err.Error()})
		return
	}

	var servers []ImageServer
	for _, e := range entries {
		if e.IsDir() {
			id := e.Name()
			name := id
			// map to friendly name if available
			if s, ok := dataStore.InferenceServers[id]; ok && s != nil && s.Name != "" {
				name = s.Name
			}
			servers = append(servers, ImageServer{ID: id, Name: name})
		}
	}

	// sort by name for stable ordering
	sort.Slice(servers, func(i, j int) bool { return servers[i].Name < servers[j].Name })

	json.NewEncoder(w).Encode(APIResponse{Success: true, Message: "image servers retrieved successfully", Data: servers})
}

// handleAPIServerImages returns paginated images for a given serverId directory
func (ws *WebServer) handleAPIServerImages(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	vars := mux.Vars(r)
	serverId := vars["serverId"]

	// parse pagination
	page := 1
	limit := 24
	if p := r.URL.Query().Get("page"); p != "" {
		if v, err := strconv.Atoi(p); err == nil && v > 0 {
			page = v
		}
	}
	if l := r.URL.Query().Get("limit"); l != "" {
		if v, err := strconv.Atoi(l); err == nil && v > 0 && v <= 100 {
			limit = v
		}
	}

	dir := filepath.Join(ws.outputDir, serverId)
	images, err := getImagesFromDirectory(dir)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(APIResponse{Success: false, Message: "failed to read server images", Error: err.Error()})
		return
	}

	// sort newest first by created_at (ModTime)
	sort.Slice(images, func(i, j int) bool { return images[i].CreatedAt.After(images[j].CreatedAt) })

	totalCount := len(images)
	totalPages := (totalCount + limit - 1) / limit
	if totalPages == 0 {
		totalPages = 1
	}

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

	json.NewEncoder(w).Encode(APIResponse{Success: true, Message: "images retrieved successfully", Data: ImageListResponse{Images: images, TotalCount: totalCount, TotalPages: totalPages, CurrentPage: page}})
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
			// TODO: wtf is this?
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
			Warn(fmt.Sprintf("failed to save data store: %v", err))
		}

		Info(fmt.Sprintf("created inference server: %s (%s)", newServer.ID, newServer.Name))

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
			Warn(fmt.Sprintf("failed to save data store: %v", err))
		}

		Info(fmt.Sprintf("updated inference server: %s", id))

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
			Warn(fmt.Sprintf("failed to save data store: %v", err))
		}

		Info(fmt.Sprintf("deleted inference server: %s", id))

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
			Warn(fmt.Sprintf("failed to save data store: %v", err))
		}

		Info(fmt.Sprintf("updated alert server configuration: URL=%s, Enabled=%t", updatedConfig.URL, updatedConfig.Enabled))

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

// startFallDetectionResultPolling starts a polling goroutine for fall detection results
func (ws *WebServer) startFallDetectionResultPolling(task *FallDetectionTaskState, server *InferenceServer, camera *CameraConfig) {
	// Create ticker for high-frequency polling (every 200ms to handle 25 FPS)
	// This ensures we don't miss results from the inference server
	task.pollTicker = time.NewTicker(200 * time.Millisecond)
	task.pollStopChan = make(chan bool, 1)

	// Start polling goroutine
	go func() {
		Info(fmt.Sprintf("started fall detection result polling for task %s (camera: %s, server: %s)", task.TaskID, camera.Name, server.Name))

		for {
			select {
			case <-task.pollTicker.C:
				// Fetch fall detection results
				limit := 20
				results, err := GetFallDetectionResults(server, task.TaskID, &limit)
				if err != nil {
					Warn(fmt.Sprintf("failed to poll fall detection results for task %s: %v", task.TaskID, err))
					continue
				}

				if len(results) == 0 {
					continue // no results, continue polling
				}

				Info(fmt.Sprintf("polling found %d fall detection results for task %s", len(results), task.TaskID))

				// Process results using existing logic
				// Find camera binding for threshold check
				var binding *InferenceServerBinding
				for _, b := range camera.InferenceServerBindings {
					if b.ServerID == server.ID {
						binding = &b
						break
					}
				}

				if binding == nil {
					Warn(fmt.Sprintf("binding not found for server %s", server.ID))
					continue
				}

				ws.processFallResultsFromPolling(results, server, camera, binding)

			case <-task.pollStopChan:
				// Safely stop ticker within goroutine to avoid race condition
				if task.pollTicker != nil {
					task.pollTicker.Stop()
				}
				Info(fmt.Sprintf("stopped fall detection result polling for task %s", task.TaskID))
				return
			}
		}
	}()
}

// processFallResultsFromPolling processes fall detection results from polling
func (ws *WebServer) processFallResultsFromPolling(results []FallDetectionResultItem, server *InferenceServer, camera *CameraConfig, binding *InferenceServerBinding) {
	// Process each fall detection result independently and save immediately
	for _, result := range results {
		// Decode the image from backend
		var imageData []byte
		if result.Image != "" {
			decoded, err := base64.StdEncoding.DecodeString(result.Image)
			if err != nil {
				Warn(fmt.Sprintf("failed to decode fall detection image: %v", err))
				continue
			}
			imageData = decoded
		} else {
			Warn("fall detection result has empty image data")
			continue
		}

		// Convert to Detection format
		confidence := result.Results.Score
		if confidence > 1.0 {
			confidence = confidence / 100.0
		}

		// Check threshold range (min and max)
		if confidence < binding.Threshold {
			continue
		}
		// Check max threshold if set (0 means no max limit)
		if binding.MaxThreshold > 0 && confidence > binding.MaxThreshold {
			continue
		}

		detection := Detection{
			Class:      "FALL_DETECTED",
			Confidence: confidence,
			X1:         int(result.Results.Location.Left),
			Y1:         int(result.Results.Location.Top),
			X2:         int(result.Results.Location.Left + result.Results.Location.Width),
			Y2:         int(result.Results.Location.Top + result.Results.Location.Height),
		}

		// Draw detection on the original image
		drawnImage, err := DrawDetections(imageData, []Detection{detection}, camera.Name, false)
		if err != nil {
			Warn(fmt.Sprintf("failed to draw image for fall detection: %v", err))
			continue
		}
		// debug image.
		debugImage, err := DrawDetections(imageData, []Detection{detection}, camera.Name, true)
		if err != nil {
			Warn(fmt.Sprintf("failed to draw debug image for fall detection: %v", err))
			continue
		}

		// Store original image copy for DEBUG mode
		var originalImageCopy []byte
		if globalDebugMode {
			originalImageCopy = make([]byte, len(imageData))
			copy(originalImageCopy, imageData)
		}

		// Create ModelResult for this SINGLE detection
		singleModelResult := map[string]*ModelResult{
			server.ModelType: {
				ModelType:          server.ModelType,
				ServerID:           binding.ServerID,
				Detections:         []Detection{detection},
				DisplayResultImage: drawnImage,
				DisplayDebugImage:  debugImage,
				OriginalImage:      originalImageCopy,
				Error:              nil,
			},
		}

		// Save this single fall detection result immediately
		if ws.rtspManager != nil {
			ws.rtspManager.saveResultsByModel(camera.Name, singleModelResult)
			Info(fmt.Sprintf("saved fall detection result: confidence=%.2f, camera=%s", confidence, camera.Name))
		} else {
			Warn("rtspManager is nil, cannot save fall detection result")
		}
	}
}

// stopFallDetectionResultPolling stops the polling goroutine for a task
func (ws *WebServer) stopFallDetectionResultPolling(task *FallDetectionTaskState) {
	if task.pollStopChan != nil {
		task.pollStopChan <- true
		close(task.pollStopChan)
		task.pollStopChan = nil
	}
	if task.pollTicker != nil {
		task.pollTicker.Stop()
		task.pollTicker = nil
	}
}

// startFallDetectionTask starts a fall detection task for a camera with a fall detection server
func (ws *WebServer) startFallDetectionTask(camera *CameraConfig, server *InferenceServer) {
	// Check if there's already a running task for this camera-server combination
	for _, task := range fallDetectionTasks {
		if task.CameraID == camera.ID && task.ServerID == server.ID && task.Status == "running" {
			Info(fmt.Sprintf("fall detection task already running for camera %s with server %s (task: %s)", camera.ID, server.ID, task.TaskID))
			return
		}
	}

	// Start the fall detection task
	taskID, err := StartFallDetection(server, camera)
	if err != nil {
		Warn(fmt.Sprintf("failed to start fall detection task for camera %s with server %s: %v", camera.ID, server.ID, err))
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
	task := &FallDetectionTaskState{
		TaskID:    taskID,
		CameraID:  camera.ID,
		ServerID:  server.ID,
		Status:    "running",
		StartedAt: time.Now(),
		UpdatedAt: time.Now(),
	}
	fallDetectionTasks[taskID] = task

	// Start independent result polling for this task
	ws.startFallDetectionResultPolling(task, server, camera)

	Info(fmt.Sprintf("started fall detection task %s with result polling for camera %s with server %s", taskID, camera.ID, server.ID))
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
		Info(fmt.Sprintf("no running fall detection tasks found for camera %s with server %s", cameraID, serverID))
		return
	}

	server, serverExists := dataStore.InferenceServers[serverID]
	if !serverExists {
		Warn(fmt.Sprintf("inference server %s not found when stopping fall detection tasks", serverID))
		return
	}

	// Stop each task
	for _, taskID := range tasksToStop {
		task := fallDetectionTasks[taskID]

		// Stop the result polling first
		ws.stopFallDetectionResultPolling(task)

		err := StopFallDetection(server, taskID)
		if err != nil {
			Warn(fmt.Sprintf("failed to stop fall detection task %s: %v", taskID, err))
			// Update task state to error
			task.Status = "error"
			task.ErrorMsg = err.Error()
			task.UpdatedAt = time.Now()
		} else {
			// Update task state to stopped
			task.Status = "stopped"
			task.UpdatedAt = time.Now()
			Info(fmt.Sprintf("stopped fall detection task %s with polling for camera %s with server %s", taskID, cameraID, serverID))
		}
	}
}

// Config Export/Import API Handlers

// handleAPIConfigExport exports the current camera configuration as JSON
func (ws *WebServer) handleAPIConfigExport(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Disposition", "attachment; filename=cameras.json")

	// Export current dataStore as JSON
	data, err := json.MarshalIndent(dataStore, "", "  ")
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		response := APIResponse{
			Success: false,
			Message: "Failed to export configuration",
			Error:   err.Error(),
		}
		json.NewEncoder(w).Encode(response)
		return
	}

	// Return the JSON data directly for download
	w.Write(data)
	Info("configuration exported successfully")
}

// handleAPIConfigImport imports camera configuration from uploaded JSON
func (ws *WebServer) handleAPIConfigImport(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	if r.Method != "POST" {
		w.WriteHeader(http.StatusMethodNotAllowed)
		response := APIResponse{
			Success: false,
			Message: "Method not allowed",
		}
		json.NewEncoder(w).Encode(response)
		return
	}

	// Read the uploaded JSON data
	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		response := APIResponse{
			Success: false,
			Message: "Failed to read request body",
			Error:   err.Error(),
		}
		json.NewEncoder(w).Encode(response)
		return
	}

	// Parse the JSON data
	var importedData DataStore
	if err := json.Unmarshal(body, &importedData); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		response := APIResponse{
			Success: false,
			Message: "Invalid JSON format",
			Error:   err.Error(),
		}
		json.NewEncoder(w).Encode(response)
		return
	}

	// Validate imported data
	if importedData.Cameras == nil {
		importedData.Cameras = make(map[string]*CameraConfig)
	}
	if importedData.InferenceServers == nil {
		importedData.InferenceServers = make(map[string]*InferenceServer)
	}

	// Stop all running cameras and fall detection tasks before import
	if ws.rtspManager != nil {
		for cameraID := range dataStore.Cameras {
			ws.rtspManager.StopCamera(cameraID)
		}
	}

	// Stop all fall detection tasks
	for _, task := range fallDetectionTasks {
		ws.stopFallDetectionResultPolling(task)
	}
	fallDetectionTasks = make(map[string]*FallDetectionTaskState)

	// Replace current dataStore with imported data
	dataStore = &importedData

	// Ensure max_threshold field is set for existing bindings without it
	for _, camera := range dataStore.Cameras {
		for i := range camera.InferenceServerBindings {
			binding := &camera.InferenceServerBindings[i]
			if binding.MaxThreshold == 0 {
				// Set default max threshold to 1.0 if not specified
				binding.MaxThreshold = 1.0
			}
		}
	}

	// Save the imported configuration
	if err := saveDataStore(); err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		response := APIResponse{
			Success: false,
			Message: "Failed to save imported configuration",
			Error:   err.Error(),
		}
		json.NewEncoder(w).Encode(response)
		return
	}

	// Start imported cameras
	if ws.rtspManager != nil {
		for _, camera := range dataStore.Cameras {
			if camera.Enabled && camera.Running {
				if err := ws.rtspManager.StartCamera(camera); err != nil {
					Warn(fmt.Sprintf("failed to start imported camera %s: %v", camera.ID, err))
				}

				// Start fall detection tasks for fall detection servers
				for _, binding := range camera.InferenceServerBindings {
					server, serverExists := dataStore.InferenceServers[binding.ServerID]
					if serverExists && server.Enabled && server.ModelType == "fall" {
						ws.startFallDetectionTask(camera, server)
					}
				}
			}
		}
	}

	Info(fmt.Sprintf("configuration imported successfully: %d cameras, %d inference servers", 
		len(dataStore.Cameras), len(dataStore.InferenceServers)))

	response := APIResponse{
		Success: true,
		Message: fmt.Sprintf("Configuration imported successfully: %d cameras, %d inference servers", 
			len(dataStore.Cameras), len(dataStore.InferenceServers)),
		Data: map[string]interface{}{
			"cameras_count":          len(dataStore.Cameras),
			"inference_servers_count": len(dataStore.InferenceServers),
		},
	}
	json.NewEncoder(w).Encode(response)
}
