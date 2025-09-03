package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image/jpeg"
	"io/ioutil"
	"log"
	"net/http"
	"time"

	"github.com/google/uuid"
)

// InferenceClient handles communication with inference server
type InferenceClient struct {
	serverURL string
	client    *http.Client
}

// Detection represents a single object detection
type Detection struct {
	Class      string  `json:"class"`
	Confidence float64 `json:"confidence"`
	X1         int     `json:"x1"`
	Y1         int     `json:"y1"`
	X2         int     `json:"x2"`
	Y2         int     `json:"y2"`
}

// InferenceRequest represents the request to inference server
type InferenceRequest struct {
	Image     string `json:"image"`      // Base64 encoded image
	ModelType string `json:"model_type"` // Model type to use
	CameraID  string `json:"camera_id"`  // Camera ID for fall detection
}

// InferenceResponse represents the response from inference server
type InferenceResponse struct {
	LogId        string            `json:"log_id"`
	Errno        int               `json:"errno"`
	ErrMsg       string            `json:"err_msg"`
	ApiVersion   string            `json:"api_version"`
	ModelVersion string            `json:"model_version"`
	Results      []DetectionResult `json:"results"`
}

// DetectionResult represents a single detection result from Python server
type DetectionResult struct {
	Score    float64  `json:"score"`
	Location Location `json:"location"`
	Class    string   `json:"class,omitempty"` // Class name if provided by server
}

// Location represents the location of a detected object
type Location struct {
	Left   float64 `json:"left"`
	Top    float64 `json:"top"`
	Width  float64 `json:"width"`
	Height float64 `json:"height"`
}

// NewInferenceClient creates a new inference client
func NewInferenceClient(serverURL string) *InferenceClient {
	if serverURL == "" {
		serverURL = "http://localhost:8000" // Default inference server URL
	}
	return &InferenceClient{
		serverURL: serverURL,
		client: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// DetectObjects sends image to inference server and returns detections
func (ic *InferenceClient) DetectObjects(imageData []byte, modelType string) ([]Detection, error) {
	// Encode image to base64
	encodedImage := base64.StdEncoding.EncodeToString(imageData)

	// Create request
	request := InferenceRequest{
		Image:     encodedImage,
		ModelType: modelType,
	}

	requestBody, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %v", err)
	}

	// Build URL - use serverURL directly if it's already a complete URL
	url := ic.serverURL

	// Send request to inference server
	resp, err := ic.client.Post(
		url,
		"application/json",
		bytes.NewBuffer(requestBody),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to send request to inference server: %v", err)
	}
	defer resp.Body.Close()

	// Check HTTP status
	if resp.StatusCode != http.StatusOK {
		body, _ := ioutil.ReadAll(resp.Body)
		if len(body) > 0 && body[0] == '<' {
			return nil, fmt.Errorf("inference server returned HTML (status %d) - check if service is running at %s", resp.StatusCode, url)
		}
		return nil, fmt.Errorf("inference server returned status %d: %s", resp.StatusCode, string(body))
	}

	// Read response
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %v", err)
	}

	// Check if response is HTML (error page)
	if len(body) > 0 && body[0] == '<' {
		return nil, fmt.Errorf("inference server returned HTML instead of JSON - service may not be running at %s", url)
	}

	// Parse response
	var response InferenceResponse
	if err := json.Unmarshal(body, &response); err != nil {
		// Log first 200 chars of response for debugging
		preview := string(body)
		if len(preview) > 200 {
			preview = preview[:200] + "..."
		}
		return nil, fmt.Errorf("failed to parse response: %v (response preview: %s)", err, preview)
	}

	// Check if inference was successful (errno == 0 means success)
	// Note: NO_OBJECT_DETECTED is not an error, it's a valid result with no detections
	if response.Errno != 0 && response.ErrMsg != "NO_OBJECT_DETECTED" {
		return nil, fmt.Errorf("inference failed: %s (errno: %d)", response.ErrMsg, response.Errno)
	}

	// Log when no objects are detected (for debugging)
	if response.ErrMsg == "NO_OBJECT_DETECTED" {
		// TODO: slog.Debug()
		// log.Printf("No objects detected by inference server - returning empty detections")
		return []Detection{}, nil // Return empty detections, not an error
	}

	// Get actual image dimensions for coordinate conversion
	imgReader := bytes.NewReader(imageData)
	img, err := jpeg.DecodeConfig(imgReader)
	if err != nil {
		log.Printf("Warning: Failed to decode image config, using defaults: %v", err)
		img.Width = 1920
		img.Height = 1080
	}

	// Convert Python server results to our Detection format
	var detections []Detection
	for _, result := range response.Results {
		// Skip results with no score (no detection)
		if result.Score <= 0 || result.Location.Left < 0 {
			log.Printf("Skipping invalid detection with score %.3f", result.Score)
			continue
		}

		// Python server returns normalized coordinates [0,1]
		// Convert to pixel coordinates
		x1 := int(result.Location.Left * float64(img.Width))
		y1 := int(result.Location.Top * float64(img.Height))
		x2 := int((result.Location.Left + result.Location.Width) * float64(img.Width))
		y2 := int((result.Location.Top + result.Location.Height) * float64(img.Height))

		// Ensure coordinates are within image bounds
		if x1 < 0 {
			x1 = 0
		}
		if y1 < 0 {
			y1 = 0
		}
		if x2 > img.Width {
			x2 = img.Width
		}
		if y2 > img.Height {
			y2 = img.Height
		}

		// Skip invalid boxes
		if x2 <= x1 || y2 <= y1 {
			log.Printf("Skipping invalid box coordinates: (%d,%d,%d,%d)", x1, y1, x2, y2)
			continue
		}

		// Use class name from server response, or default to "detected_object"
		className := result.Class
		if className == "" {
			className = "detected_object" // Default class name
		}

		detection := Detection{
			Class:      className,
			Confidence: result.Score,
			X1:         x1,
			Y1:         y1,
			X2:         x2,
			Y2:         y2,
		}
		detections = append(detections, detection)
	}

	// Don't log here - let the threshold logic in ProcessFrameWithMultipleInference handle logging

	return detections, nil
}

// FallDetectionResponse represents the response from fall detection API
type FallDetectionResponse struct {
	LogId        string                 `json:"log_id"`
	Errno        int                    `json:"errno"`
	ErrMsg       string                 `json:"err_msg"`
	ApiVersion   string                 `json:"api_version"`
	ModelVersion string                 `json:"model_version"`
	CameraID     string                 `json:"camera_id"`
	Timestamp    float64                `json:"timestamp"`
	PersonsCount int                    `json:"persons_detected"`
	FallDetected bool                   `json:"fall_detected"`
	DebugInfo    map[string]interface{} `json:"debug_info"`
	Results      []FallDetectionResult  `json:"results"`
}

// FallDetectionResult represents a single fall detection result
type FallDetectionResult struct {
	Score     float64  `json:"score"` // 🎯 Fall confidence score (0-1)
	PersonID  string   `json:"person_id"`
	AlertType string   `json:"alert_type"`
	Location  Location `json:"location"`
}

// ProcessFrameForFallDetection processes a frame for fall detection using unified API pattern
func (ic *InferenceClient) ProcessFrameForFallDetection(imageData []byte, cameraID string) ([]Detection, error) {
	// Now use the same DetectObjects method with "fall" model type
	// This maintains the camera_id in the request for fall detection context
	encodedImage := base64.StdEncoding.EncodeToString(imageData)

	// Create unified request format
	request := InferenceRequest{
		Image:     encodedImage,
		ModelType: "fall",
		CameraID:  cameraID, // Fall detection still needs camera ID for temporal context
	}

	requestBody, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal fall detection request: %v", err)
	}

	// Use unified endpoint pattern
	url := ic.serverURL + "/fall"

	// Send request using standard format
	resp, err := ic.client.Post(
		url,
		"application/json",
		bytes.NewBuffer(requestBody),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to send request to fall detection API: %v", err)
	}
	defer resp.Body.Close()

	// Check HTTP status
	if resp.StatusCode != http.StatusOK {
		body, _ := ioutil.ReadAll(resp.Body)
		if len(body) > 0 && body[0] == '<' {
			return nil, fmt.Errorf("fall detection API returned HTML (status %d) - check if service is running at %s", resp.StatusCode, url)
		}
		return nil, fmt.Errorf("fall detection API returned status %d: %s", resp.StatusCode, string(body))
	}

	// Read response
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read fall detection response: %v", err)
	}

	// Check if response is HTML (error page)
	if len(body) > 0 && body[0] == '<' {
		return nil, fmt.Errorf("fall detection API returned HTML instead of JSON - service may not be running at %s", url)
	}

	// Parse response using unified format
	var response InferenceResponse
	if err := json.Unmarshal(body, &response); err != nil {
		// Log first 200 chars of response for debugging
		preview := string(body)
		if len(preview) > 200 {
			preview = preview[:200] + "..."
		}
		return nil, fmt.Errorf("failed to parse fall detection response: %v (response preview: %s)", err, preview)
	}

	// Check if inference was successful (errno == 0 means success)
	if response.Errno != 0 {
		return nil, fmt.Errorf("fall detection failed: %s (errno: %d)", response.ErrMsg, response.Errno)
	}

	// Get actual image dimensions for coordinate conversion
	imgReader := bytes.NewReader(imageData)
	img, err := jpeg.DecodeConfig(imgReader)
	if err != nil {
		log.Printf("Warning: Failed to decode image config, using defaults: %v", err)
		img.Width = 1920
		img.Height = 1080
	}

	// Convert unified response to Detection format
	var detections []Detection
	for _, result := range response.Results {
		// Skip results with no score (no detection)
		if result.Score <= 0 || result.Location.Left < 0 {
			log.Printf("Skipping invalid fall detection with score %.3f", result.Score)
			continue
		}

		// Convert normalized coordinates to pixels
		x1 := int(result.Location.Left * float64(img.Width))
		y1 := int(result.Location.Top * float64(img.Height))
		x2 := int((result.Location.Left + result.Location.Width) * float64(img.Width))
		y2 := int((result.Location.Top + result.Location.Height) * float64(img.Height))

		// Ensure coordinates are within image bounds
		if x1 < 0 {
			x1 = 0
		}
		if y1 < 0 {
			y1 = 0
		}
		if x2 > img.Width {
			x2 = img.Width
		}
		if y2 > img.Height {
			y2 = img.Height
		}

		// Skip invalid boxes
		if x2 <= x1 || y2 <= y1 {
			log.Printf("Skipping invalid fall detection box coordinates: (%d,%d,%d,%d)", x1, y1, x2, y2)
			continue
		}

		// Create fall detection with special class name
		className := "FALL_DETECTED"
		if result.Score >= 0.8 {
			className = "HIGH_CONFIDENCE_FALL"
		}

		detection := Detection{
			Class:      className,
			Confidence: result.Score,
			X1:         x1,
			Y1:         y1,
			X2:         x2,
			Y2:         y2,
		}
		detections = append(detections, detection)
	}

	// Only log when fall detections are found
	if len(detections) > 0 {
		log.Printf("🚨 Fall detection completed: %d alerts with confidences:", len(detections))
		for _, det := range detections {
			log.Printf("  %s: %.1f%%", det.Class, det.Confidence*100)
		}
	}

	return detections, nil
}

// ModelResult represents detection results for a specific model
type ModelResult struct {
	ModelType  string      `json:"model_type"`
	ServerID   string      `json:"server_id"`
	Detections []Detection `json:"detections"`
	ShouldSave bool        `json:"should_save"`
	Error      error       `json:"-"`
}

// ProcessFrame processes a frame with inference using new multi-server architecture
func ProcessFrameWithInference(frameData []byte, cameraConfig *CameraConfig) ([]byte, map[string]*ModelResult, error) {
	// Use new multi-server processing if inference servers are configured
	if len(cameraConfig.InferenceServerBindings) > 0 || len(cameraConfig.InferenceServers) > 0 {
		return ProcessFrameWithMultipleInference(frameData, cameraConfig)
	}

	// Fallback to legacy single server mode for backward compatibility
	if cameraConfig.ServerUrl == "" {
		processedFrame, _ := DrawDetections(frameData, []Detection{}, cameraConfig.Name)
		return processedFrame, nil, nil
	}

	// Create inference client
	client := NewInferenceClient(cameraConfig.ServerUrl)

	// Check if this is fall detection model
	if cameraConfig.ModelType == "fall" {
		processedFrame, shouldSave, err := ProcessFrameWithFallDetection(frameData, cameraConfig, client)
		if err != nil {
			return processedFrame, nil, err
		}

		// Convert to new format
		results := make(map[string]*ModelResult)
		if shouldSave {
			results["fall"] = &ModelResult{
				ModelType:  "fall",
				ServerID:   "legacy",
				Detections: []Detection{},
				ShouldSave: true,
			}
		}
		return processedFrame, results, nil
	}

	// Regular object detection for other models
	detections, err := client.DetectObjects(frameData, cameraConfig.ModelType)
	if err != nil {
		log.Printf("Warning: Inference failed for camera %s: %v", cameraConfig.ID, err)
		processedFrame, _ := DrawDetections(frameData, []Detection{}, cameraConfig.Name)
		return processedFrame, nil, nil
	}

	// Draw detections on frame
	processedFrame, err := DrawDetections(frameData, detections, cameraConfig.Name)
	if err != nil {
		log.Printf("Warning: Failed to draw detections for camera %s: %v", cameraConfig.ID, err)
		return frameData, nil, nil
	}

	// Convert to new format
	results := make(map[string]*ModelResult)
	if len(detections) > 0 {
		results[cameraConfig.ModelType] = &ModelResult{
			ModelType:  cameraConfig.ModelType,
			ServerID:   "legacy",
			Detections: detections,
			ShouldSave: true,
		}
	}

	return processedFrame, results, nil
}

// ProcessFrameWithMultipleInference processes a frame with multiple inference servers
func ProcessFrameWithMultipleInference(frameData []byte, cameraConfig *CameraConfig) ([]byte, map[string]*ModelResult, error) {
	results := make(map[string]*ModelResult)
	allDetections := []Detection{}

	// Determine server list based on configuration format
	var serverProcessList []struct {
		ServerID  string
		Threshold float64
	}

	// Use new binding format if available, otherwise fall back to legacy format
	if len(cameraConfig.InferenceServerBindings) > 0 {
		for _, binding := range cameraConfig.InferenceServerBindings {
			serverProcessList = append(serverProcessList, struct {
				ServerID  string
				Threshold float64
			}{binding.ServerID, binding.Threshold})
		}
	} else if len(cameraConfig.InferenceServers) > 0 {
		// Legacy format - use default threshold of 0.5 for backward compatibility
		for _, serverID := range cameraConfig.InferenceServers {
			serverProcessList = append(serverProcessList, struct {
				ServerID  string
				Threshold float64
			}{serverID, 0.5})
		}
	}

	// Process each inference server
	for _, serverConfig := range serverProcessList {
		server, exists := dataStore.InferenceServers[serverConfig.ServerID]
		if !exists {
			log.Printf("Warning: Inference server %s not found for camera %s", serverConfig.ServerID, cameraConfig.ID)
			continue
		}

		if !server.Enabled {
			log.Printf("Skipping disabled inference server %s for camera %s", serverConfig.ServerID, cameraConfig.ID)
			continue
		}

		// Create inference client
		client := NewInferenceClient(server.URL)

		// Process based on model type
		var detections []Detection
		var shouldSave bool
		var err error

		if server.ModelType == "fall" {
			// Handle fall detection using unified API
			detections, err = client.ProcessFrameForFallDetection(frameData, cameraConfig.ID)
			if err != nil {
				log.Printf("Warning: Fall detection failed for server %s: %v", server.Name, err)
				results[server.ModelType] = &ModelResult{
					ModelType:  server.ModelType,
					ServerID:   serverConfig.ServerID,
					Detections: []Detection{},
					ShouldSave: false,
					Error:      err,
				}
				continue
			}

			// For fall detection, save if any falls were detected (threshold not applicable for falls)
			shouldSave = len(detections) > 0
			if shouldSave {
				log.Printf("🚨 FALL DETECTED by server %s: %d alerts", server.Name, len(detections))
			}
		} else {
			// Regular object detection
			detections, err = client.DetectObjects(frameData, server.ModelType)
			if err != nil {
				log.Printf("Warning: Inference failed for server %s: %v", server.Name, err)
				results[server.ModelType] = &ModelResult{
					ModelType:  server.ModelType,
					ServerID:   serverConfig.ServerID,
					Detections: []Detection{},
					ShouldSave: false,
					Error:      err,
				}
				continue
			}

			// Apply confidence threshold - save if any detection meets threshold
			shouldSave = false
			for _, detection := range detections {
				if detection.Confidence >= serverConfig.Threshold {
					shouldSave = true
					break
				}
			}

			if shouldSave {
				log.Printf("Threshold met for server %s (threshold: %.1f%%) - saving image", server.Name, serverConfig.Threshold*100)
			}
		}

		// Store results for this model
		results[server.ModelType] = &ModelResult{
			ModelType:  server.ModelType,
			ServerID:   serverConfig.ServerID,
			Detections: detections,
			ShouldSave: shouldSave,
			Error:      nil,
		}

		// Collect all detections for drawing
		allDetections = append(allDetections, detections...)

		// Only log when detections meet threshold and will be saved
		if shouldSave && len(detections) > 0 {
			log.Printf("Camera %s, Server %s (%s): %d detections with confidences:",
				cameraConfig.ID, server.Name, server.ModelType, len(detections))
			for _, det := range detections {
				log.Printf("  %s: %.1f%%", det.Class, det.Confidence*100)
			}
		}
	}

	// Draw all detections on frame
	processedFrame, err := DrawDetections(frameData, allDetections, cameraConfig.Name)
	if err != nil {
		log.Printf("Warning: Failed to draw detections for camera %s: %v", cameraConfig.ID, err)
		return frameData, results, nil
	}

	// Only log summary when threshold-meeting detections are saved
	hasThresholdMeetingDetections := false
	for _, result := range results {
		if result.ShouldSave {
			hasThresholdMeetingDetections = true
			break
		}
	}
	if hasThresholdMeetingDetections {
		log.Printf("Camera %s: %d total detections from %d inference servers (saving images)",
			cameraConfig.ID, len(allDetections), len(serverProcessList))
	}

	return processedFrame, results, nil
}

// ProcessFrameWithFallDetection processes a frame with fall detection using unified API
func ProcessFrameWithFallDetection(frameData []byte, cameraConfig *CameraConfig, client *InferenceClient) ([]byte, bool, error) {
	// Call unified fall detection API
	detections, err := client.ProcessFrameForFallDetection(frameData, cameraConfig.ID)
	if err != nil {
		log.Printf("Warning: Fall detection failed for camera %s: %v", cameraConfig.ID, err)
		// Still draw timestamp overlay even if inference fails
		processedFrame, _ := DrawDetections(frameData, []Detection{}, cameraConfig.Name)
		return processedFrame, false, nil
	}

	// Draw fall detections on frame
	processedFrame, err := DrawDetections(frameData, detections, cameraConfig.Name)
	if err != nil {
		log.Printf("Warning: Failed to draw fall detections for camera %s: %v", cameraConfig.ID, err)
		return frameData, false, nil
	}

	// Save if any falls were detected
	shouldSave := len(detections) > 0
	if shouldSave {
		log.Printf("🚨 FALL DETECTED in camera %s: %d alerts",
			cameraConfig.ID, len(detections))
	}

	return processedFrame, shouldSave, nil
}

// Helper function to check if string contains substring
func contains(s, substr string) bool {
	return bytes.Contains([]byte(s), []byte(substr))
}

// AlertRequest represents the alert request format for management platform
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

// SendAlertIfConfigured sends detection alert to management platform using global configuration
func SendAlertIfConfigured(imageData []byte, model string, cameraName string, score float64, x1, y1, x2, y2 float64) error {
	// Check if alert system is enabled and configured globally
	if dataStore.AlertServer == nil || !dataStore.AlertServer.Enabled || dataStore.AlertServer.URL == "" {
		return nil // Alert system not enabled or not configured, silently skip
	}

	// Encode image to base64
	encodedImage := base64.StdEncoding.EncodeToString(imageData)

	// Create alert request using camera name directly as KKS
	alertReq := AlertRequest{
		Image:     encodedImage,
		RequestID: uuid.New().String(),
		Model:     model,
		CameraKKS: cameraName, // Use camera name directly as KKS encoding
		Score:     score,
		X1:        x1,
		Y1:        y1,
		X2:        x2,
		Y2:        y2,
		Timestamp: time.Now().Format("2006-01-02T15:04:05+08:00"),
	}

	// Marshal request
	requestBody, err := json.Marshal(alertReq)
	if err != nil {
		return fmt.Errorf("failed to marshal alert request: %v", err)
	}

	// Create HTTP client
	client := &http.Client{Timeout: 30 * time.Second}

	// Send request
	resp, err := client.Post(
		dataStore.AlertServer.URL,
		"application/json",
		bytes.NewBuffer(requestBody),
	)
	if err != nil {
		return fmt.Errorf("failed to send alert to platform: %v", err)
	}
	defer resp.Body.Close()

	// Check response status
	if resp.StatusCode != http.StatusOK {
		body, _ := ioutil.ReadAll(resp.Body)
		return fmt.Errorf("platform returned status %d: %s", resp.StatusCode, string(body))
	}

	log.Printf("Alert sent successfully to platform for camera %s (model: %s, score: %.3f)",
		cameraName, model, score)

	return nil
}

// SendAlert is deprecated - use SendAlertIfConfigured instead
func SendAlert(platformURL string, imageData []byte, model string, camera string, score float64, x1, y1, x2, y2 float64) error {
	return SendAlertIfConfigured(imageData, model, camera, score, x1, y1, x2, y2)
}
