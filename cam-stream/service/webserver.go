package service

import (
	"cam-stream/common"
	"cam-stream/common/config"
	"cam-stream/common/log"
	"cam-stream/common/store"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"
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
	OutputDir   string
	Port        uint
	RtspManager *RTSPManager
	Templates   *HTMLTemplates
}

// loadHTMLTemplates loads all HTML files into memory at startup
func loadHTMLTemplates() (*HTMLTemplates, error) {
	templates := &HTMLTemplates{}
	templatesDir := config.TemplatesDir
	// Load index.html
	if data, err := os.ReadFile(filepath.Join(templatesDir, "index.html")); err != nil {
		return nil, fmt.Errorf("failed to load index.html: %v", err)
	} else {
		templates.index = data
	}

	// Load camera_management.html
	if data, err := os.ReadFile(filepath.Join(templatesDir, "camera_management.html")); err != nil {
		return nil, fmt.Errorf("failed to load camera_management.html: %v", err)
	} else {
		templates.cameraManagement = data
	}

	// Load image_viewer.html
	if data, err := os.ReadFile(filepath.Join(templatesDir, "image_viewer.html")); err != nil {
		return nil, fmt.Errorf("failed to load image_viewer.html: %v", err)
	} else {
		templates.imageViewer = data
	}

	// Load alerts.html
	if data, err := os.ReadFile(filepath.Join(templatesDir, "alerts.html")); err != nil {
		return nil, fmt.Errorf("failed to load alerts.html: %v", err)
	} else {
		templates.alerts = data
	}

	log.Info("successfully loaded all HTML templates into memory")
	return templates, nil
}

// NewWebServer creates a new web server
func NewWebServer(outputDir string, port uint, rtspManager *RTSPManager) *WebServer {
	templates, err := loadHTMLTemplates()
	if err != nil {
		log.Warn(fmt.Sprintf("failed to load HTML templates: %v", err))
		// ensure non-nil to avoid nil dereference; actual fix is to include templates in runtime image
		templates = &HTMLTemplates{}
	}

	return &WebServer{
		OutputDir:   outputDir,
		Port:        port,
		Templates:   templates,
		RtspManager: rtspManager,
	}
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

	log.Info(fmt.Sprintf("starting web server on port %d", ws.Port))
	log.Info(fmt.Sprintf("access web interface at: http://localhost:%d", ws.Port))

	return http.ListenAndServe(fmt.Sprintf(":%d", ws.Port), router)
}

// handleIndex serves the main page with links to different sections
func (ws *WebServer) handleIndex(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write(ws.Templates.index)
}

// handleCameraManagement serves the camera management page
func (ws *WebServer) handleCameraManagement(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write(ws.Templates.cameraManagement)
}

// handleImages serves the image viewer page
func (ws *WebServer) handleImages(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write(ws.Templates.imageViewer)
}

func generateCameraID() string {
	// Use full UUID to ensure uniqueness, remove dashes for cleaner look
	return "cam_" + strings.ReplaceAll(uuid.New().String(), "-", "")
}

func generateInferenceServerID(modelType string) string {
	// Sanitize model type for ID (replace spaces and special chars with underscore)
	sanitizedModelType := strings.ReplaceAll(strings.ToLower(modelType), " ", "_")
	sanitizedModelType = strings.ReplaceAll(sanitizedModelType, "-", "_")

	// Use full UUID to ensure uniqueness
	// Generate ID format: inf_<model_type>_<full_uuid>
	uuidPart := strings.ReplaceAll(uuid.New().String(), "-", "")
	return fmt.Sprintf("inf_%s_%s", sanitizedModelType, uuidPart)
}

type APIResponse struct {
	Success bool        `json:"success"`
	Message string      `json:"message"`
	Data    interface{} `json:"data,omitempty"`
	Error   string      `json:"error,omitempty"`
}

// API Handlers
func (ws *WebServer) handleAPICameras(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	switch r.Method {
	case "GET":
		var cameraList []*store.CameraConfig
		store.SafeReadDataStore(func() {
			for _, camera := range store.Data.Cameras {
				cameraList = append(cameraList, camera)
			}
		})

		response := APIResponse{
			Success: true,
			Message: "Cameras retrieved successfully",
			Data:    cameraList,
		}
		json.NewEncoder(w).Encode(response)

	case "POST":
		var newCamera store.CameraConfig
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

		store.SafeUpdateDataStore(func() {
			store.Data.Cameras[newCamera.ID] = &newCamera
		})

		if err := store.SaveDataStore(); err != nil {
			log.Warn(fmt.Sprintf("failed to save data store: %v", err))
		}

		// Actually start the RTSP stream processing
		if ws.RtspManager != nil {
			if err := ws.RtspManager.StartCamera(&newCamera); err != nil {
				log.Warn(fmt.Sprintf("failed to start RTSP stream for camera %s: %v", newCamera.ID, err))
			}
		}

		// Start fall detection tasks for any bound fall detection servers
		for _, binding := range newCamera.InferenceServerBindings {
			server, serverExists := store.SafeGetInferenceServer(binding.ServerID)
			if serverExists && server.Enabled && server.ModelType == string(config.ModelTypeFall) {
				ws.startFallDetectionTask(&newCamera, server)
			}
		}

		log.Info(fmt.Sprintf("created camera: %s (%s)", newCamera.ID, newCamera.Name))

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

	camera, exists := store.SafeGetCamera(id)
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
		var updatedCamera store.CameraConfig
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

		store.SafeUpdateDataStore(func() {
			store.Data.Cameras[id] = &updatedCamera
		})

		if err := store.SaveDataStore(); err != nil {
			log.Warn(fmt.Sprintf("failed to save data store: %v", err))
		}

		log.Info(fmt.Sprintf("updated camera: %s", id))

		response := APIResponse{
			Success: true,
			Message: "Camera updated successfully",
			Data:    &updatedCamera,
		}
		json.NewEncoder(w).Encode(response)

	case "DELETE":
		// Stop RTSP stream first
		if ws.RtspManager != nil {
			if err := ws.RtspManager.StopCamera(id); err != nil {
				log.Warn(fmt.Sprintf("failed to stop RTSP stream for camera %s: %v", id, err))
			}
		}

		// Stop fall detection tasks for this camera
		for _, binding := range camera.InferenceServerBindings {
			server, serverExists := store.SafeGetInferenceServer(binding.ServerID)
			if serverExists && server.ModelType == string(config.ModelTypeFall) {
				ws.stopFallDetectionTasksForCamera(camera.ID, server.ID)
			}
		}

		store.SafeUpdateDataStore(func() {
			delete(store.Data.Cameras, id)
		})

		if err := store.SaveDataStore(); err != nil {
			log.Warn(fmt.Sprintf("failed to save data store: %v", err))
		}

		log.Info(fmt.Sprintf("deleted camera: %s", id))

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
	totalCameras := 0

	store.SafeReadDataStore(func() {
		totalCameras = len(store.Data.Cameras)
		for _, camera := range store.Data.Cameras {
			if camera.Running {
				runningCount++
			}
			if camera.Enabled {
				enabledCount++
			}
		}
	})

	response := APIResponse{
		Success: true,
		Message: "System status retrieved successfully",
		Data: map[string]interface{}{
			"manager_running":    true,
			"running_cameras":    runningCount,
			"total_cameras":      totalCameras,
			"enabled_cameras":    enabledCount,
			"disabled_cameras":   totalCameras - enabledCount,
			"persistent_storage": true,
			"data_file":          config.DataFile,
		},
	}

	json.NewEncoder(w).Encode(response)
}

func (ws *WebServer) handleAPIDebug(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	var cameraIDs []string
	totalCameras := 0
	store.SafeReadDataStore(func() {
		totalCameras = len(store.Data.Cameras)
		for id := range store.Data.Cameras {
			cameraIDs = append(cameraIDs, id)
		}
	})

	debugInfo := map[string]interface{}{
		"message":            "Debug route is working!",
		"timestamp":          time.Now().Format(time.RFC3339),
		"routes_registered":  "API routes are properly registered",
		"cors_enabled":       true,
		"total_cameras":      totalCameras,
		"camera_ids":         cameraIDs,
		"persistent_storage": true,
		"data_file_exists":   fileExists(config.DataFile),
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

// getCommandVersion gets the version string from a command like ffmpeg or ffprobe
func getCommandVersion(command string) string {
	if cmd := exec.Command(command, "-version"); cmd != nil {
		if output, err := cmd.Output(); err == nil {
			lines := strings.Split(string(output), "\n")
			if len(lines) > 0 {
				return strings.TrimSpace(lines[0])
			}
		}
	}
	return "unknown"
}

// handleAPIPing returns basic liveness and FFmpeg version info
func (ws *WebServer) handleAPIPing(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	info := map[string]interface{}{
		"timestamp":       time.Now().Format(time.RFC3339),
		"ffmpeg_version":  getCommandVersion("ffmpeg"),
		"ffprobe_version": getCommandVersion("ffprobe"),
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

func getImagesFromDirectory(dir string) ([]ImageInfo, error) {
	var images []ImageInfo

	// Check if directory exists
	if _, err := os.Stat(dir); err != nil {
		if os.IsNotExist(err) {
			return images, nil
		}
		return nil, fmt.Errorf("failed to stat dir: %w", err)
	}

	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, fmt.Errorf("failed to read directory: %v", err)
	}

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}

		filename := entry.Name()
		lowerName := strings.ToLower(filename)
		if !strings.HasSuffix(lowerName, ".jpg") && !strings.HasSuffix(lowerName, ".jpeg") {
			continue
		}

		path := filepath.Join(dir, filename)
		info, err := os.Stat(path)
		if err != nil {
			return nil, fmt.Errorf("stat %s: %w", path, err)
		}

		images = append(images, ImageInfo{
			Filename:  filename,
			Size:      info.Size(),
			CreatedAt: info.ModTime(),
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
	entries, err := os.ReadDir(ws.OutputDir)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(APIResponse{Success: false, Message: "failed to read output directory", Error: err.Error()})
		return
	}

	var servers []ImageServer
	for _, e := range entries {
		if e.IsDir() {
			id := e.Name()
			// only include servers that exist in dataStore (not deleted)
			server, exists := store.SafeGetInferenceServer(id)
			if exists && server != nil {
				name := server.Name
				if name == "" {
					name = id // fallback to id if name is empty
				}
				servers = append(servers, ImageServer{ID: id, Name: name})
			}
			// Note: directories for deleted servers are ignored but files remain on disk
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
	// TODO: hard coding
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

	dir := filepath.Join(ws.OutputDir, serverId)
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
		var serverList []*store.InferenceServer
		store.SafeReadDataStore(func() {
			for _, server := range store.Data.InferenceServers {
				serverList = append(serverList, server)
			}
		})

		response := APIResponse{
			Success: true,
			Message: "Inference servers retrieved successfully",
			Data:    serverList,
		}
		json.NewEncoder(w).Encode(response)

	case "POST":
		var newServer store.InferenceServer
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
			newServer.ModelType = string(config.ModelTypeOther)
		}

		if newServer.ID == "" {
			newServer.ID = generateInferenceServerID(newServer.ModelType)
		}

		newServer.CreatedAt = time.Now()
		newServer.UpdatedAt = time.Now()
		newServer.Enabled = true

		store.SafeUpdateDataStore(func() {
			store.Data.InferenceServers[newServer.ID] = &newServer
		})

		if err := store.SaveDataStore(); err != nil {
			log.Warn(fmt.Sprintf("failed to save data store: %v", err))
		}

		log.Info(fmt.Sprintf("created inference server: %s (%s)", newServer.ID, newServer.Name))

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

	server, exists := store.SafeGetInferenceServer(id)
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
		var updatedServer store.InferenceServer
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

		store.SafeUpdateDataStore(func() {
			store.Data.InferenceServers[id] = &updatedServer
		})

		if err := store.SaveDataStore(); err != nil {
			log.Warn(fmt.Sprintf("failed to save data store: %v", err))
		}

		log.Info(fmt.Sprintf("updated inference server: %s", id))

		response := APIResponse{
			Success: true,
			Message: "Inference server updated successfully",
			Data:    &updatedServer,
		}
		json.NewEncoder(w).Encode(response)

	case "DELETE":
		store.SafeUpdateDataStore(func() {
			for _, camera := range store.Data.Cameras {
				for i, serverBinding := range camera.InferenceServerBindings {
					if serverBinding.ServerID == id {
						camera.InferenceServerBindings = append(camera.InferenceServerBindings[:i], camera.InferenceServerBindings[i+1:]...)
						camera.UpdatedAt = time.Now()
						break
					}
				}
			}

			delete(store.Data.InferenceServers, id)
		})

		if err := store.SaveDataStore(); err != nil {
			log.Warn(fmt.Sprintf("failed to save data store: %v", err))
		}

		log.Info(fmt.Sprintf("deleted inference server: %s", id))

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
		var alertConfig *store.AlertServerConfig
		store.SafeReadDataStore(func() {
			if store.Data.AlertServer != nil {
				alertConfig = store.Data.AlertServer
			} else {
				// Return default/empty configuration
				alertConfig = &store.AlertServerConfig{
					URL:       "",
					Enabled:   false,
					UpdatedAt: time.Now(),
				}
			}
		})

		response := APIResponse{
			Success: true,
			Message: "Alert server configuration retrieved successfully",
			Data:    alertConfig,
		}
		json.NewEncoder(w).Encode(response)

	case "PUT":
		// Update alert server configuration
		var updatedConfig store.AlertServerConfig
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
		store.SafeUpdateDataStore(func() {
			store.Data.AlertServer = &updatedConfig
		})

		if err := store.SaveDataStore(); err != nil {
			log.Warn(fmt.Sprintf("failed to save data store: %v", err))
		}

		log.Info(fmt.Sprintf("updated alert server configuration: URL=%s, Enabled=%t", updatedConfig.URL, updatedConfig.Enabled))

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
	w.Write(ws.Templates.alerts)
}

// Fall Detection Task Management Functions

// startFallDetectionResultPolling starts a polling goroutine for fall detection results
func (ws *WebServer) startFallDetectionResultPolling(task *store.FallDetectionTaskState, server *store.InferenceServer,
	camera *store.CameraConfig) {
	// Create ticker for high-frequency polling (every 200ms to handle 25 FPS)
	// This ensures we don't miss results from the inference server
	task.PollTicker = time.NewTicker(200 * time.Millisecond)
	task.PollStopChan = make(chan bool, 1)

	// Start polling goroutine
	go func() {
		log.Info(fmt.Sprintf("started fall detection result polling for task %s (camera: %s, server: %s)", task.TaskID, camera.Name, server.Name))

		for {
			// Safety check for ticker before accessing it
			if task.PollTicker == nil {
				log.Info(fmt.Sprintf("ticker is nil for task %s, stopping polling goroutine", task.TaskID))
				return
			}

			select {
			case <-task.PollTicker.C:
				// Fetch fall detection results
				limit := 20
				results, err := GetFallDetectionResults(server, task.TaskID, &limit)
				if err != nil {
					log.Warn(fmt.Sprintf("failed to poll fall detection results for task %s: %v", task.TaskID, err))
					continue
				}

				if len(results) == 0 {
					continue // no results, continue polling
				}

				log.Info(fmt.Sprintf("polling found %d fall detection results for task %s", len(results), task.TaskID))

				// Process results using existing logic
				// Find camera binding for threshold check
				var binding *store.InferenceServerBinding
				for _, b := range camera.InferenceServerBindings {
					if b.ServerID == server.ID {
						binding = &b
						break
					}
				}

				if binding == nil {
					log.Warn(fmt.Sprintf("binding not found for server %s", server.ID))
					continue
				}

				ws.processFallResultsFromPolling(results, server, camera, binding)

			case <-task.PollStopChan:
				// Safely stop ticker within goroutine to avoid race condition
				if task.PollTicker != nil {
					task.PollTicker.Stop()
				}
				log.Info(fmt.Sprintf("stopped fall detection result polling for task %s", task.TaskID))
				return
			}
		}
	}()
}

// processFallResultsFromPolling processes fall detection results from polling
func (ws *WebServer) processFallResultsFromPolling(results []FallDetectionResultItem, server *store.InferenceServer,
	camera *store.CameraConfig, binding *store.InferenceServerBinding) {
	// Process each fall detection result independently and save immediately
	for _, result := range results {
		// Decode the image from backend
		var imageData []byte
		if result.Image != "" {
			decoded, err := base64.StdEncoding.DecodeString(result.Image)
			if err != nil {
				log.Warn(fmt.Sprintf("failed to decode fall detection image: %v", err))
				continue
			}
			imageData = decoded
		} else {
			log.Warn("fall detection result has empty image data")
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

		detection := common.Detection{
			Class:      "FALL_DETECTED",
			Confidence: confidence,
			X1:         int(result.Results.Location.Left),
			Y1:         int(result.Results.Location.Top),
			X2:         int(result.Results.Location.Left + result.Results.Location.Width),
			Y2:         int(result.Results.Location.Top + result.Results.Location.Height),
		}

		// Draw detection on the original image
		drawnImage, err := common.DrawDetections(imageData, []common.Detection{detection}, camera.Name, false)
		if err != nil {
			log.Warn(fmt.Sprintf("failed to draw image for fall detection: %v", err))
			continue
		}
		// debug image.
		debugImage, err := common.DrawDetections(imageData, []common.Detection{detection}, camera.Name, true)
		if err != nil {
			log.Warn(fmt.Sprintf("failed to draw debug image for fall detection: %v", err))
			continue
		}

		// Store original image copy for DEBUG mode
		var originalImageCopy []byte
		if config.GlobalDebugMode {
			originalImageCopy = make([]byte, len(imageData))
			copy(originalImageCopy, imageData)
		}

		// Create ModelResult for this SINGLE detection
		singleModelResult := map[string]*ModelResult{
			server.ModelType: {
				ModelType:          server.ModelType,
				ServerID:           binding.ServerID,
				Detections:         []common.Detection{detection},
				DisplayResultImage: drawnImage,
				DisplayDebugImage:  debugImage,
				OriginalImage:      originalImageCopy,
				Error:              nil,
			},
		}

		// Launch independent async operations for fall detection result
		modelResult := singleModelResult[server.ModelType]

		// 1. Save fall detection result (async)
		go func() {
			saveModelResult(camera.Name, modelResult, ws.RtspManager.OutputDir)
		}()

		// 2. Send fall detection alert (async)
		go func() {
			// Create image data copy for alert sending
			alertImageData := make([]byte, len(modelResult.DisplayResultImage))
			copy(alertImageData, modelResult.DisplayResultImage)

			sendDetectionAlerts(alertImageData, modelResult.Detections, camera.Name, modelResult.ModelType)
		}()

		log.Info(fmt.Sprintf("processed fall detection result: confidence=%.2f, camera=%s", confidence, camera.Name))
	}
}

// stopFallDetectionResultPolling stops the polling goroutine for a task
func (ws *WebServer) stopFallDetectionResultPolling(task *store.FallDetectionTaskState) {
	if task.PollStopChan != nil {
		task.PollStopChan <- true
		close(task.PollStopChan)
		task.PollStopChan = nil
	}
	if task.PollTicker != nil {
		task.PollTicker.Stop()
		task.PollTicker = nil
	}
}

// startFallDetectionTask starts a fall detection task for a camera with a fall detection server
func (ws *WebServer) startFallDetectionTask(camera *store.CameraConfig, server *store.InferenceServer) {
	// Check if there's already a running task for this camera-server combination
	var existingTaskFound bool
	store.SafeReadTasks(func() {
		for _, task := range store.FallDetectionTasks {
			if task.CameraID == camera.ID && task.ServerID == server.ID && task.Status == "running" {
				log.Info(fmt.Sprintf("fall detection task already running for camera %s with server %s (task: %s)", camera.ID, server.ID, task.TaskID))
				existingTaskFound = true
				return
			}
		}
	})
	if existingTaskFound {
		return
	}

	// Start the fall detection task
	taskID, err := StartFallDetection(server, camera)
	if err != nil {
		log.Warn(fmt.Sprintf("failed to start fall detection task for camera %s with server %s: %v", camera.ID, server.ID, err))
		// Create error task state
		store.SafeUpdateTasks(func() {
			store.FallDetectionTasks["error_"+camera.ID+"_"+server.ID] = &store.FallDetectionTaskState{
				TaskID:    "",
				CameraID:  camera.ID,
				ServerID:  server.ID,
				Status:    "error",
				StartedAt: time.Now(),
				UpdatedAt: time.Now(),
				ErrorMsg:  err.Error(),
			}
		})
		return
	}

	// Create successful task state using the returned taskID as key
	task := &store.FallDetectionTaskState{
		TaskID:    taskID,
		CameraID:  camera.ID,
		ServerID:  server.ID,
		Status:    "running",
		StartedAt: time.Now(),
		UpdatedAt: time.Now(),
	}
	store.SafeUpdateTasks(func() {
		store.FallDetectionTasks[taskID] = task
	})

	// Start independent result polling for this task
	ws.startFallDetectionResultPolling(task, server, camera)

	log.Info(fmt.Sprintf("started fall detection task %s with result polling for camera %s with server %s", taskID, camera.ID, server.ID))
}

// stopFallDetectionTasksForCamera stops all fall detection tasks for a specific camera-server combination
func (ws *WebServer) stopFallDetectionTasksForCamera(cameraID, serverID string) {
	var tasksToStop []string

	// Find all running tasks for this camera-server combination
	store.SafeReadTasks(func() {
		for taskID, task := range store.FallDetectionTasks {
			if task.CameraID == cameraID && task.ServerID == serverID && task.Status == "running" {
				tasksToStop = append(tasksToStop, taskID)
			}
		}
	})

	if len(tasksToStop) == 0 {
		log.Info(fmt.Sprintf("no running fall detection tasks found for camera %s with server %s", cameraID, serverID))
		return
	}

	server, serverExists := store.SafeGetInferenceServer(serverID)
	if !serverExists {
		log.Warn(fmt.Sprintf("inference server %s not found when stopping fall detection tasks", serverID))
		return
	}

	// Stop each task
	for _, taskID := range tasksToStop {
		var task *store.FallDetectionTaskState
		store.SafeReadTasks(func() {
			task = store.FallDetectionTasks[taskID]
		})

		if task != nil {
			// Stop the result polling first
			ws.stopFallDetectionResultPolling(task)

			err := StopFallDetection(server, taskID)
			if err != nil {
				log.Warn(fmt.Sprintf("failed to stop fall detection task %s: %v", taskID, err))
				// Update task state to error
				store.SafeUpdateTasks(func() {
					if t, exists := store.FallDetectionTasks[taskID]; exists {
						t.Status = "error"
						t.ErrorMsg = err.Error()
						t.UpdatedAt = time.Now()
					}
				})
			} else {
				// Update task state to stopped
				store.SafeUpdateTasks(func() {
					if t, exists := store.FallDetectionTasks[taskID]; exists {
						t.Status = "stopped"
						t.UpdatedAt = time.Now()
					}
				})
				log.Info(fmt.Sprintf("stopped fall detection task %s with polling for camera %s with server %s", taskID, cameraID, serverID))
			}
		}
	}
}

// Config Export/Import API Handlers

// handleAPIConfigExport exports the current camera configuration as JSON
func (ws *WebServer) handleAPIConfigExport(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	// Generate filename with timestamp
	timestamp := time.Now().Format("2006-01-02_15-04-05")
	filename := fmt.Sprintf("cameras_%s.json", timestamp)
	w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=%s", filename))

	// Export current dataStore as JSON using thread-safe access
	var data []byte
	var err error
	store.SafeReadDataStore(func() {
		data, err = json.MarshalIndent(store.Data, "", "  ")
	})

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
	log.Info("configuration exported successfully")
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
	body, err := io.ReadAll(r.Body)
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
	var importedData store.DataStore
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
		importedData.Cameras = make(map[string]*store.CameraConfig)
	}
	if importedData.InferenceServers == nil {
		importedData.InferenceServers = make(map[string]*store.InferenceServer)
	}

	// Stop all running cameras and fall detection tasks before import
	if ws.RtspManager != nil {
		var cameraIDs []string
		store.SafeReadDataStore(func() {
			for cameraID := range store.Data.Cameras {
				cameraIDs = append(cameraIDs, cameraID)
			}
		})
		for _, cameraID := range cameraIDs {
			ws.RtspManager.StopCamera(cameraID)
		}
	}

	// Stop all fall detection tasks
	var tasksToStop []*store.FallDetectionTaskState
	store.SafeReadTasks(func() {
		for _, task := range store.FallDetectionTasks {
			tasksToStop = append(tasksToStop, task)
		}
	})
	for _, task := range tasksToStop {
		ws.stopFallDetectionResultPolling(task)
	}
	store.SafeUpdateTasks(func() {
		store.FallDetectionTasks = make(map[string]*store.FallDetectionTaskState)
	})

	// Replace current dataStore with imported data using thread-safe access
	store.SafeUpdateDataStore(func() {
		store.Data = &importedData

		// Ensure max_threshold field is set for existing bindings without it
		for _, camera := range store.Data.Cameras {
			for i := range camera.InferenceServerBindings {
				binding := &camera.InferenceServerBindings[i]
				if binding.MaxThreshold == 0 {
					// Set default max threshold to 1.0 if not specified
					binding.MaxThreshold = 1.0
				}
			}
		}
	})

	// Save the imported configuration
	if err := store.SaveDataStore(); err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		response := APIResponse{
			Success: false,
			Message: "Failed to save imported configuration",
			Error:   err.Error(),
		}
		json.NewEncoder(w).Encode(response)
		return
	}

	// Start imported cameras using thread-safe access
	if ws.RtspManager != nil {
		var camerasToStart []*store.CameraConfig
		store.SafeReadDataStore(func() {
			for _, camera := range store.Data.Cameras {
				if camera.Enabled && camera.Running {
					camerasToStart = append(camerasToStart, camera)
				}
			}
		})

		for _, camera := range camerasToStart {
			if err := ws.RtspManager.StartCamera(camera); err != nil {
				log.Warn(fmt.Sprintf("failed to start imported camera %s: %v", camera.ID, err))
			}

			// Start fall detection tasks for fall detection servers
			for _, binding := range camera.InferenceServerBindings {
				server, serverExists := store.SafeGetInferenceServer(binding.ServerID)
				if serverExists && server.Enabled && server.ModelType == string(config.ModelTypeFall) {
					ws.startFallDetectionTask(camera, server)
				}
			}
		}
	}

	// Get counts using thread-safe access
	var camerasCount, serversCount int
	store.SafeReadDataStore(func() {
		camerasCount = len(store.Data.Cameras)
		serversCount = len(store.Data.InferenceServers)
	})

	log.Info(fmt.Sprintf("configuration imported successfully: %d cameras, %d inference servers",
		camerasCount, serversCount))

	response := APIResponse{
		Success: true,
		Message: fmt.Sprintf("Configuration imported successfully: %d cameras, %d inference servers",
			camerasCount, serversCount),
		Data: map[string]interface{}{
			"cameras_count":           camerasCount,
			"inference_servers_count": serversCount,
		},
	}
	json.NewEncoder(w).Encode(response)
}
