package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"path/filepath"
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

const (
	Port      = 8081
	OutputDir = "received_alerts"
)

func main() {
	log.Printf("Starting Alert Platform Mock Server on port %d", Port)

	// Create output directory for saved alerts
	if err := os.MkdirAll(OutputDir, 0755); err != nil {
		log.Fatalf("Failed to create output directory: %v", err)
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
	router.HandleFunc("/", handleIndex).Methods("GET")
	router.HandleFunc("/status", handleStatus).Methods("GET")
	router.HandleFunc("/alerts", handleListAlerts).Methods("GET")

	log.Printf("Mock server is running at: http://localhost:%d", Port)
	log.Printf("Alert endpoint: http://localhost:%d/alert", Port)
	log.Printf("Status endpoint: http://localhost:%d/status", Port)
	log.Printf("Alert list: http://localhost:%d/alerts", Port)

	// Start server
	log.Fatal(http.ListenAndServe(fmt.Sprintf(":%d", Port), router))
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

	log.Printf("Alert saved successfully for camera %s (Request ID: %s)", alertReq.CameraKKS, alertReq.RequestID)

	// Send success response
	response := AlertResponse{
		Success:   true,
		Message:   "Alert received and saved successfully",
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

// handleIndex serves a simple status page
func handleIndex(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")

	content := `
<!DOCTYPE html>
<html>
<head>
    <title>Alert Platform Mock Server</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
        h1 { color: #333; }
        .status { background: #f0f8ff; padding: 15px; border-radius: 5px; margin: 20px 0; }
        .endpoint { background: #f5f5f5; padding: 10px; margin: 10px 0; border-left: 4px solid #007cba; }
        pre { background: #f5f5f5; padding: 15px; border-radius: 3px; overflow-x: auto; }
    </style>
</head>
<body>
    <h1>Alert Platform Mock Server</h1>
    <div class="status">
        <strong>Status:</strong> Server is running and ready to receive alerts
    </div>
    
    <h2>Available Endpoints</h2>
    
    <div class="endpoint">
        <strong>POST /alert</strong> - Receive detection alerts (main endpoint)
    </div>
    
    <div class="endpoint">
        <strong>GET /status</strong> - Server status and statistics
    </div>
    
    <div class="endpoint">
        <strong>GET /alerts</strong> - List received alerts
    </div>
    
    <h2>Usage</h2>
    <p>Configure your camera's <code>platform_url</code> to:</p>
    <pre>http://localhost:8080/alert</pre>
    
    <p>Alerts will be automatically saved as JSON files in the <code>received_alerts/</code> directory.</p>
    
    <h2>Example Alert Request</h2>
    <pre>{
  "image": "base64_encoded_image",
  "request_id": "2058ed71-6b61-4f66-83e6-59a31ba87d7b",
  "model": "model-name-v1.0",
  "camera_kks": "1M2DTW345TV",
  "score": 0.95,
  "x1": 0.1,
  "y1": 0.2,
  "x2": 0.4,
  "y2": 0.6,
  "timestamp": "2024-01-01T12:00:00+08:00"
}</pre>
</body>
</html>
	`

	w.Write([]byte(content))
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

	status := map[string]interface{}{
		"server_name":     "Alert Platform Mock Server",
		"status":          "running",
		"port":            Port,
		"output_dir":      OutputDir,
		"alerts_received": alertCount,
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
	}

	json.NewEncoder(w).Encode(response)
}
