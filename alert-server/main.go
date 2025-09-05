package main

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/gorilla/mux"
)

// AlertRequest represents the alert request format from management platform
type AlertRequest struct {
	Image     string  `json:"image"`
	RequestID string  `json:"request_id"`
	Model     string  `json:"model"`
	CameraKKS string  `json:"camera_kks"`
	Score     float64 `json:"score"`
	X1        float64 `json:"x1"`
	Y1        float64 `json:"y1"`
	X2        float64 `json:"x2"`
	Y2        float64 `json:"y2"`
	Timestamp string  `json:"timestamp"`
}

// AlertResponse represents the response to alert requests
type AlertResponse struct {
	Success   bool   `json:"success"`
	Message   string `json:"message"`
	RequestID string `json:"request_id"`
}

// CamStreamData represents the structure of cam-stream's cameras.json
type CamStreamData struct {
	AlertServer struct {
		URL       string    `json:"url"`
		Enabled   bool      `json:"enabled"`
		UpdatedAt time.Time `json:"updated_at"`
	} `json:"alert_server"`
}

const (
	DefaultPort = 8081
	OutputDir   = "saved_requests"
	ImagesDir   = "saved_images"
)

// Global variable to store current port
var currentPort int

func main() {

	currentPort = DefaultPort

	log.Printf("Starting Alert Platform Mock Server on port %d", DefaultPort)

	// Create output directories for saved alerts and images
	if err := os.MkdirAll(OutputDir, 0755); err != nil {
		log.Fatalf("Failed to create output directory: %v", err)
	}
	if err := os.MkdirAll(ImagesDir, 0755); err != nil {
		log.Fatalf("Failed to create images directory: %v", err)
	}

	// Set up HTTP routes
	router := mux.NewRouter()

	// Add CORS middleware for web debugging
	router.Use(func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Access-Control-Allow-Origin", "*")
			w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
			w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
			if r.Method == "OPTIONS" {
				w.WriteHeader(http.StatusOK)
				return
			}

			next.ServeHTTP(w, r)
		})
	})

	// Alert endpoint - matches the API specification
	router.HandleFunc("/alert", handleAlert).Methods("POST", "OPTIONS")
	// Debug endpoints
	router.HandleFunc("/status", handleStatus).Methods("GET")
	router.HandleFunc("/alerts", handleListAlerts).Methods("GET")
	fmt.Println("server is up")
	slog.Error(http.ListenAndServe(fmt.Sprintf(":%d", DefaultPort), router).Error())
}

// handleAlert processes incoming alert requests
func handleAlert(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	log.Printf("Received alert request from %s", r.RemoteAddr)

	// Read request body
	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		log.Printf("Failed to read request body: %v", err)
		response := AlertResponse{
			Success: false,
			Message: "Failed to read request body",
		}
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(response)
		return
	}

	// Parse JSON alert request
	var alertReq AlertRequest
	if err := json.Unmarshal(body, &alertReq); err != nil {
		log.Printf("Failed to parse alert request: %v", err)
		response := AlertResponse{
			Success: false,
			Message: "Invalid JSON format",
		}
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(response)
		return
	}

	log.Printf("Alert received - Camera: %s, Model: %s, Score: %.3f, Request ID: %s",
		alertReq.CameraKKS, alertReq.Model, alertReq.Score, alertReq.RequestID)

	// Save alert to file
	if err := saveAlertToFile(alertReq); err != nil {
		log.Printf("Failed to save alert to file: %v", err)
		response := AlertResponse{
			Success:   false,
			Message:   "Failed to save alert",
			RequestID: alertReq.RequestID,
		}
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(response)
		return
	}

	// Save image if present
	var imageSaveErr error
	if alertReq.Image != "" {
		if err := saveImageToFile(alertReq); err != nil {
			log.Printf("Failed to save image to file: %v", err)
			imageSaveErr = err
		}
	}

	log.Printf("Alert saved successfully for camera %s (Request ID: %s)", alertReq.CameraKKS, alertReq.RequestID)

	// Send success response
	message := "Alert received and saved successfully"
	if imageSaveErr != nil {
		message += " (warning: image save failed)"
	}

	response := AlertResponse{
		Success:   true,
		Message:   message,
		RequestID: alertReq.RequestID,
	}
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(response)
}

// saveAlertToFile saves the alert request to a JSON file
func saveAlertToFile(alertReq AlertRequest) error {
	// Create timestamp for filename
	timestamp := time.Now().Format("20060102_150405_000")

	// Create filename with camera, model and timestamp info
	filename := fmt.Sprintf("%s_%s_%s_%s.json",
		timestamp, alertReq.CameraKKS, alertReq.Model, alertReq.RequestID)

	// Ensure filename is safe
	filename = filepath.Base(filename)
	filePath := filepath.Join(OutputDir, filename)

	// Convert alert to pretty JSON
	jsonData, err := json.MarshalIndent(alertReq, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal alert to JSON: %v", err)
	}

	// Write to file
	if err := ioutil.WriteFile(filePath, jsonData, 0644); err != nil {
		return fmt.Errorf("failed to write alert file: %v", err)
	}

	log.Printf("Alert saved to file: %s", filePath)
	return nil
}

// saveImageToFile decodes and saves the base64 image to a file
func saveImageToFile(alertReq AlertRequest) error {
	if alertReq.Image == "" {
		return nil
	}

	// Create timestamp for filename
	timestamp := time.Now().Format("20060102_150405_000")

	// Remove data URL prefix if present (e.g., "data:image/jpeg;base64,")
	imageData := alertReq.Image
	if strings.Contains(imageData, ",") {
		parts := strings.SplitN(imageData, ",", 2)
		if len(parts) == 2 {
			imageData = parts[1]
		}
	}

	// Decode base64 image
	decodedImage, err := base64.StdEncoding.DecodeString(imageData)
	if err != nil {
		return fmt.Errorf("failed to decode base64 image: %v", err)
	}

	// Determine file extension based on image magic bytes
	ext := getImageExtension(decodedImage)

	// Create filename with camera, model and timestamp info
	filename := fmt.Sprintf("%s_%s_%s_%s%s",
		timestamp, alertReq.CameraKKS, alertReq.Model, alertReq.RequestID, ext)

	// Ensure filename is safe
	filename = filepath.Base(filename)
	filePath := filepath.Join(ImagesDir, filename)

	// Write image to file
	if err := ioutil.WriteFile(filePath, decodedImage, 0644); err != nil {
		return fmt.Errorf("failed to write image file: %v", err)
	}

	log.Printf("Image saved to file: %s", filePath)
	return nil
}

// getImageExtension determines the file extension based on image magic bytes
func getImageExtension(data []byte) string {
	if len(data) < 8 {
		return ".bin"
	}

	// Check for common image formats
	switch {
	case len(data) >= 2 && data[0] == 0xFF && data[1] == 0xD8: // JPEG
		return ".jpg"
	case len(data) >= 8 && string(data[0:8]) == "\x89PNG\x0D\x0A\x1A\x0A": // PNG
		return ".png"
	case len(data) >= 6 && string(data[0:6]) == "GIF87a" || string(data[0:6]) == "GIF89a": // GIF
		return ".gif"
	case len(data) >= 4 && string(data[0:4]) == "RIFF" && len(data) >= 12 && string(data[8:12]) == "WEBP": // WEBP
		return ".webp"
	case len(data) >= 2 && data[0] == 'B' && data[1] == 'M': // BMP
		return ".bmp"
	default:
		return ".bin"
	}
}

// handleStatus returns server status and statistics
func handleStatus(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	// Count alert files
	files, err := ioutil.ReadDir(OutputDir)
	if err != nil {
		log.Printf("Failed to read output directory: %v", err)
	}

	alertCount := 0
	for _, file := range files {
		if !file.IsDir() && filepath.Ext(file.Name()) == ".json" {
			alertCount++
		}
	}

	// Count image files
	imageFiles, err := ioutil.ReadDir(ImagesDir)
	if err != nil {
		log.Printf("Failed to read images directory: %v", err)
	}

	imageCount := 0
	for _, file := range imageFiles {
		if !file.IsDir() {
			ext := strings.ToLower(filepath.Ext(file.Name()))
			if ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".gif" || ext == ".webp" || ext == ".bmp" {
				imageCount++
			}
		}
	}

	status := map[string]interface{}{
		"server_name":     "Alert Platform Mock Server",
		"status":          "running",
		"port":            currentPort,
		"output_dir":      OutputDir,
		"images_dir":      ImagesDir,
		"alerts_received": alertCount,
		"images_saved":    imageCount,
		"timestamp":       time.Now().Format(time.RFC3339),
		"endpoints": map[string]string{
			"alert":  "/alert",
			"status": "/status",
			"alerts": "/alerts",
		},
	}

	json.NewEncoder(w).Encode(status)
}

// handleListAlerts returns a list of received alerts
func handleListAlerts(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	files, err := ioutil.ReadDir(OutputDir)
	if err != nil {
		log.Printf("Failed to read output directory: %v", err)
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{
			"error": "Failed to read alert directory",
		})
		return
	}

	var alerts []map[string]interface{}
	for _, file := range files {
		if !file.IsDir() && filepath.Ext(file.Name()) == ".json" {
			alerts = append(alerts, map[string]interface{}{
				"filename": file.Name(),
				"size":     file.Size(),
				"modified": file.ModTime().Format(time.RFC3339),
			})
		}
	}

	response := map[string]interface{}{
		"total_alerts": len(alerts),
		"alerts":       alerts,
		"output_dir":   OutputDir,
		"images_dir":   ImagesDir,
	}

	json.NewEncoder(w).Encode(response)
}
