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

	// Log response for debugging
	log.Printf("Inference server response: %s", string(body))

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
	if response.Errno != 0 {
		return nil, fmt.Errorf("inference failed: %s (errno: %d)", response.ErrMsg, response.Errno)
	}

	// Get actual image dimensions for coordinate conversion
	imgReader := bytes.NewReader(imageData)
	img, err := jpeg.DecodeConfig(imgReader)
	if err != nil {
		log.Printf("Warning: Failed to decode image config, using defaults: %v", err)
		img.Width = 1920
		img.Height = 1080
	}
	log.Printf("Image dimensions: %dx%d", img.Width, img.Height)

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

		// Debug: log coordinate conversion
		log.Printf("Converting coordinates: Left=%.3f, Top=%.3f, Width=%.3f, Height=%.3f -> (%d,%d,%d,%d)",
			result.Location.Left, result.Location.Top, result.Location.Width, result.Location.Height,
			x1, y1, x2, y2)

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

	log.Printf("Detected %d real objects from inference server", len(detections))

	return detections, nil
}

// FallDetectionResponse represents the response from fall detection API
type FallDetectionResponse struct {
	LogId         string                 `json:"log_id"`
	Errno         int                    `json:"errno"`
	ErrMsg        string                 `json:"err_msg"`
	ApiVersion    string                 `json:"api_version"`
	ModelVersion  string                 `json:"model_version"`
	CameraID      string                 `json:"camera_id"`
	Timestamp     float64                `json:"timestamp"`
	PersonsCount  int                    `json:"persons_detected"`
	FallDetected  bool                   `json:"fall_detected"`
	DebugInfo     map[string]interface{} `json:"debug_info"`
	Results       []FallDetectionResult  `json:"results"`
}

// FallDetectionResult represents a single fall detection result
type FallDetectionResult struct {
	Score     float64  `json:"score"` // 🎯 Fall confidence score (0-1)
	PersonID  string   `json:"person_id"`
	AlertType string   `json:"alert_type"`
	Location  Location `json:"location"`
}

// ProcessFrameForFallDetection processes a frame specifically for fall detection
func (ic *InferenceClient) ProcessFrameForFallDetection(imageData []byte, cameraID string) (*FallDetectionResponse, error) {
	// Encode image to base64
	encodedImage := base64.StdEncoding.EncodeToString(imageData)

	// Create request for fall detection
	request := map[string]interface{}{
		"image":     encodedImage,
		"camera_id": cameraID,
	}

	requestBody, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal fall detection request: %v", err)
	}

	// Build URL for fall detection endpoint
	url := ic.serverURL + "/fall"

	// Send request to fall detection API
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
		return nil, fmt.Errorf("fall detection API returned status %d: %s", resp.StatusCode, string(body))
	}

	// Read response
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read fall detection response: %v", err)
	}

	// Parse response
	var response FallDetectionResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("failed to parse fall detection response: %v", err)
	}

	// Check if inference was successful
	if response.Errno != 0 {
		return nil, fmt.Errorf("fall detection failed: %s (errno: %d)", response.ErrMsg, response.Errno)
	}

	return &response, nil
}

// ProcessFrame processes a frame with inference
func ProcessFrameWithInference(frameData []byte, cameraConfig *CameraConfig) ([]byte, bool, error) {
	// Skip inference if no server URL configured
	if cameraConfig.ServerUrl == "" {
		processedFrame, _ := DrawDetections(frameData, []Detection{}, cameraConfig.Name)
		return processedFrame, false, nil // No detections, don't save
	}

	// Create inference client
	client := NewInferenceClient(cameraConfig.ServerUrl)

	// Check if this is fall detection model
	if cameraConfig.ModelType == "fall" {
		return ProcessFrameWithFallDetection(frameData, cameraConfig, client)
	}

	// Regular object detection for other models
	detections, err := client.DetectObjects(frameData, cameraConfig.ModelType)
	if err != nil {
		log.Printf("Warning: Inference failed for camera %s: %v", cameraConfig.ID, err)
		// Still draw timestamp overlay even if inference fails
		processedFrame, _ := DrawDetections(frameData, []Detection{}, cameraConfig.Name)
		return processedFrame, false, nil // No detections, don't save
	}

	// Draw detections on frame
	processedFrame, err := DrawDetections(frameData, detections, cameraConfig.Name)
	if err != nil {
		log.Printf("Warning: Failed to draw detections for camera %s: %v", cameraConfig.ID, err)
		return frameData, false, nil // Return original frame on error
	}

	// Save if inference was successful and returned any detections
	shouldSave := len(detections) > 0
	if shouldSave {
		log.Printf("Camera %s: Inference successful with %d detections, will save frame", cameraConfig.ID, len(detections))
	} else {
		log.Printf("Camera %s: No detections from inference server, skipping save", cameraConfig.ID)
	}

	return processedFrame, shouldSave, nil
}

// ProcessFrameWithFallDetection processes a frame with fall detection
func ProcessFrameWithFallDetection(frameData []byte, cameraConfig *CameraConfig, client *InferenceClient) ([]byte, bool, error) {
	// Call fall detection API
	fallResponse, err := client.ProcessFrameForFallDetection(frameData, cameraConfig.ID)
	if err != nil {
		log.Printf("Warning: Fall detection failed for camera %s: %v", cameraConfig.ID, err)
		// Still draw timestamp overlay even if inference fails
		processedFrame, _ := DrawDetections(frameData, []Detection{}, cameraConfig.Name)
		return processedFrame, false, nil
	}

	// Convert fall detection results to Detection format for drawing
	var detections []Detection
	for _, result := range fallResponse.Results {
		// Get actual image dimensions for coordinate conversion
		imgReader := bytes.NewReader(frameData)
		img, err := jpeg.DecodeConfig(imgReader)
		if err != nil {
			log.Printf("Warning: Failed to decode image config: %v", err)
			img.Width = 1920
			img.Height = 1080
		}

		// Convert normalized coordinates to pixels
		x1 := int(result.Location.Left * float64(img.Width))
		y1 := int(result.Location.Top * float64(img.Height))
		x2 := int((result.Location.Left + result.Location.Width) * float64(img.Width))
		y2 := int((result.Location.Top + result.Location.Height) * float64(img.Height))

		// Create detection with fall-specific class name
		className := fmt.Sprintf("FALL_ALERT_%s", result.PersonID)
		if result.Score >= 0.8 {
			className = fmt.Sprintf("HIGH_FALL_%s", result.PersonID)
		}

		detection := Detection{
			Class:      className,
			Confidence: result.Score, // 🎯 This is the fall confidence!
			X1:         x1,
			Y1:         y1,
			X2:         x2,
			Y2:         y2,
		}
		detections = append(detections, detection)
	}

	// Draw fall detections on frame
	processedFrame, err := DrawDetections(frameData, detections, cameraConfig.Name)
	if err != nil {
		log.Printf("Warning: Failed to draw fall detections for camera %s: %v", cameraConfig.ID, err)
		return frameData, false, nil
	}

	// Save if fall was detected
	shouldSave := fallResponse.FallDetected
	if shouldSave {
		log.Printf("🚨 FALL DETECTED in camera %s: %d alerts", 
			cameraConfig.ID, len(fallResponse.Results))
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

// SendAlert sends detection alert to management platform
func SendAlert(platformURL string, imageData []byte, model string, camera string, score float64, x1, y1, x2, y2 float64) error {
	if platformURL == "" {
		return fmt.Errorf("platform URL not configured")
	}

	// Encode image to base64
	encodedImage := base64.StdEncoding.EncodeToString(imageData)

	// Create alert request
	alertReq := AlertRequest{
		Image:     encodedImage,
		RequestID: uuid.New().String(),
		Model:     model,
		CameraKKS: camera,
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
		platformURL,
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
		camera, model, score)

	return nil
}
