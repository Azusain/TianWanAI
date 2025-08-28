package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
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
	api.HandleFunc("/status", ws.handleAPIStatus).Methods("GET", "OPTIONS")
	api.HandleFunc("/debug", ws.handleAPIDebug).Methods("GET", "OPTIONS")
	
	// Image management API routes
	api.HandleFunc("/images/{cameraId}", ws.handleAPIImages).Methods("GET", "OPTIONS")
	api.HandleFunc("/images/{cameraId}/{filename}", ws.handleAPIImageFile).Methods("GET", "OPTIONS")

	// Web Routes
	router.HandleFunc("/", ws.handleIndex).Methods("GET")
	router.HandleFunc("/cameras", ws.handleCameraManagement).Methods("GET")
	router.HandleFunc("/images", ws.handleImages).Methods("GET")
	
	// Static file server for output directory (images)
	router.PathPrefix("/output/").Handler(http.StripPrefix("/output/", http.FileServer(http.Dir("output/"))))
	
	// Static file server for HTML files - MUST be LAST as it's a catch-all
	router.PathPrefix("/").Handler(http.FileServer(http.Dir("./")))

	log.Printf("Starting web server on port %d", ws.port)
	log.Printf("Access web interface at: http://localhost:%d", ws.port)

	return http.ListenAndServe(fmt.Sprintf(":%d", ws.port), router)
}

// handleIndex serves the image viewer page
func (ws *WebServer) handleIndex(w http.ResponseWriter, r *http.Request) {
	content, err := ioutil.ReadFile("image_viewer.html")
	if err != nil {
		http.Error(w, "Could not load image viewer interface: "+err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write(content)
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
type CameraConfig struct {
	ID        string    `json:"id"`
	Name      string    `json:"name"`
	RTSPUrl   string    `json:"rtsp_url"`
	ServerUrl string    `json:"server_url,omitempty"`
	ModelType string    `json:"model_type,omitempty"`
	Enabled   bool      `json:"enabled"`
	Running   bool      `json:"running"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}


type APIResponse struct {
	Success bool        `json:"success"`
	Message string      `json:"message"`
	Data    interface{} `json:"data,omitempty"`
	Error   string      `json:"error,omitempty"`
}

type DataStore struct {
	Cameras map[string]*CameraConfig `json:"cameras"`
	Counters struct {
		Camera int `json:"camera"`
	} `json:"counters"`
}

const (
	DataFile = "_data/cameras.json"
	DataDir  = "_data"
)

// Global data store
var dataStore = &DataStore{
	Cameras: make(map[string]*CameraConfig),
}

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

	log.Printf("Loaded %d cameras from storage", len(dataStore.Cameras))
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

	log.Printf("Saved data store with %d cameras", len(dataStore.Cameras))
	return nil
}

func generateCameraID() string {
	dataStore.Counters.Camera++
	return fmt.Sprintf("cam_%d", dataStore.Counters.Camera)
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

	if err := saveDataStore(); err != nil {
		log.Printf("Warning: Failed to save data store: %v", err)
	}

	log.Printf("Started camera: %s", id)

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
		"message":              "Debug route is working!",
		"timestamp":            time.Now().Format(time.RFC3339),
		"routes_registered":    "API routes are properly registered",
		"cors_enabled":         true,
		"total_cameras":        len(dataStore.Cameras),
		"camera_ids":           cameraIDs,
		"persistent_storage":   true,
		"data_file_exists":     fileExists(DataFile),
		"request_method":       r.Method,
		"request_path":         r.URL.Path,
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
	Images     []ImageInfo `json:"images"`
	TotalCount int         `json:"total_count"`
	TotalPages int         `json:"total_pages"`
	CurrentPage int        `json:"current_page"`
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
