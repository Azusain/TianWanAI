package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image/jpeg"
	"io/ioutil"
	"log"
	"log/slog"
	"net/http"
	"time"

	"github.com/google/uuid"
	"github.com/pkg/errors"
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
	Score float64 `json:"score"`
	// TODO: this is used by 'tshirt' service.
	DetScore *float64 `json:"det_score,omitempty"`
	ClsScore *float64 `json:"cls_score,omitempty"`
	Location Location `json:"location"`
	// Class name if provided by server
	// TODO: only smoke detection supports this field.
	Class *string `json:"label,omitempty"`
}

// Location represents the location of a detected object
type Location struct {
	Left   float64 `json:"left"`
	Top    float64 `json:"top"`
	Width  float64 `json:"width"`
	Height float64 `json:"height"`
}

// NewInferenceClient creates a new inference client
func NewInferenceClient(serverURL string) (*InferenceClient, error) {
	if serverURL == "" {
		return nil, errors.New("server url should not be empty")
	}
	return &InferenceClient{
		serverURL: serverURL,
		client: &http.Client{
			Timeout: 30 * time.Second,
		},
	}, nil
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

	if response.ErrMsg == "NO_OBJECT_DETECTED" {
		return []Detection{}, nil
	}

	// Get actual image dimensions for coordinate conversion
	imgReader := bytes.NewReader(imageData)
	img, err := jpeg.DecodeConfig(imgReader)
	if err != nil {
		slog.Warn(fmt.Sprintf("Failed to decode image config, using defaults: %v", err))
		img.Width = 1920
		img.Height = 1080
	}

	// Convert Python server results to our Detection format
	var detections []Detection
	for _, result := range response.Results {
		// Use class name from server response, or default to "unknown_class"
		className := result.Class
		if className == nil || *className == "" {
			className = new(string)
			*className = "unknown_class" // Default class name
		}

		// TODO: this is really weird.
		if modelType == "smoke" && *className == "fire" {
			slog.Warn("filter out the class `fire` from the smoke inference server.")
			continue
		}

		confidence := 0.0
		// TODO: use enumerate instead.
		validRegularScore := modelType != "tshirt" && result.Score > 0 && result.Location.Left > 0
		validTshirtScore := modelType == "tshirt" && result.DetScore != nil && result.ClsScore != nil && *result.DetScore > 0 && *result.ClsScore > 0
		if validTshirtScore {
			// TODO: better algorithm.
			// confidence = math.Exp(math.Log(*result.DetScore) + math.Log(*result.ClsScore))
			confidence = *result.ClsScore
			slog.Info(fmt.Sprintf("det: %f , cls: %f", *result.DetScore, *result.ClsScore))
		} else if validRegularScore {
			confidence = result.Score
		} else {
			slog.Warn("invalid detection result from inference server")
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

		detection := Detection{
			Class:      *className,
			Confidence: confidence,
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
	Score     float64  `json:"score"` //  Fall confidence score (0-1)
	PersonID  string   `json:"person_id"`
	AlertType string   `json:"alert_type"`
	Location  Location `json:"location"`
}

// ModelResult represents detection results for a specific model
type ModelResult struct {
	ModelType          string      `json:"model_type"`
	ServerID           string      `json:"server_id"`
	Detections         []Detection `json:"detections"`
	DisplayResultImage []byte      `json:"display_image"`
	Error              error       `json:"-"`
}

// ProcessFrameWithMultipleInference processes a frame with multiple inference servers
func ProcessFrameWithMultipleInference(frameData []byte, cameraConfig *CameraConfig) (map[string]*ModelResult, error) {
	results := make(map[string]*ModelResult)

	// Process each inference server
	for _, binding := range cameraConfig.InferenceServerBindings {
		server, exists := dataStore.InferenceServers[binding.ServerID]
		if !exists {
			slog.Warn(fmt.Sprintf("Inference server %s not found for camera %s", binding.ServerID, cameraConfig.ID))
			continue
		}
		if !server.Enabled {
			slog.Warn(fmt.Sprintf("Skipping disabled inference server %s", server.Name))
			continue
		}

		detections := processInferenceServer(frameData, server, &binding, cameraConfig.ID)
		displayedImage, err := DrawDetections(frameData, detections, cameraConfig.Name)
		if err != nil {
			slog.Warn(fmt.Sprintf("failed to write result for model %q", server.ModelType))
		}
		// Store results
		results[server.ModelType] = &ModelResult{
			ModelType:          server.ModelType,
			ServerID:           binding.ServerID,
			Detections:         detections,
			DisplayResultImage: displayedImage,
			Error:              nil,
		}
	}

	return results, nil
}

// This function never returns nil !!!
func processInferenceServer(frameData []byte, server *InferenceServer, binding *InferenceServerBinding, cameraID string) []Detection {
	client, err := NewInferenceClient(server.URL)
	if err != nil {
		slog.Warn(fmt.Sprintf("Failed to create client for server %s: %v", server.Name, err))
		return []Detection{}
	}

	detections := []Detection{}

	// Process based on model type
	// TODO: enum instead of hardcoding.
	if server.ModelType == "fall" {
		// For fall detection, get results from the task using /fall/result API
		detections = getFallDetectionResults(server, cameraID)
	} else {
		detections, err = client.DetectObjects(frameData, server.ModelType)
		if err != nil {
			slog.Warn(fmt.Sprintf("inference failed for server %s: %v", server.Name, err))
			return detections
		}
	}

	// Check if any detection meets threshold
	retDetections := []Detection{}
	for _, detection := range detections {
		if detection.Confidence >= binding.Threshold {
			retDetections = append(retDetections, detection)
		}
	}

	return retDetections
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

// getFallDetectionResults gets fall detection results from /fall/result API
func getFallDetectionResults(server *InferenceServer, cameraID string) []Detection {
	// Find the running task for this camera and server
	var taskID string
	for id, task := range fallDetectionTasks {
		if task.CameraID == cameraID && task.ServerID == server.ID && task.Status == "running" {
			taskID = id
			break
		}
	}

	if taskID == "" {
		// No running task found, return empty detections
		return []Detection{}
	}

	// Fetch results from /fall/result API
	limit := 20 // Get up to 5 latest results
	results, err := GetFallDetectionResults(server, taskID, &limit)
	if err != nil {
		slog.Warn(fmt.Sprintf("Failed to get fall detection results for task %s: %v", taskID, err))
		return []Detection{}
	}

	// Convert results to Detection format
	var detections []Detection
	for _, result := range results {
		// Extract detection information from tianwan results
		// tianwan returns pixel coordinates that need to be used directly
		// Note: tianwan returns confidence as percentage (0-100), normalize to (0-1)
		confidence := result.Results.Score
		if confidence > 1.0 {
			// If confidence > 1, assume it's in percentage format, convert to decimal
			confidence = confidence / 100.0
		}

		detection := Detection{
			Class:      "FALL_DETECTED",
			Confidence: confidence,
			X1:         int(result.Results.Location.Left),
			Y1:         int(result.Results.Location.Top),
			X2:         int(result.Results.Location.Left + result.Results.Location.Width),
			Y2:         int(result.Results.Location.Top + result.Results.Location.Height),
		}

		detections = append(detections, detection)
	}

	if len(detections) > 0 {
		log.Printf("Got %d fall detection results for camera %s from task %s", len(detections), cameraID, taskID)
	}

	return detections
}

// SendAlert is deprecated - use SendAlertIfConfigured instead
func SendAlert(platformURL string, imageData []byte, model string, camera string, score float64, x1, y1, x2, y2 float64) error {
	return SendAlertIfConfigured(imageData, model, camera, score, x1, y1, x2, y2)
}
