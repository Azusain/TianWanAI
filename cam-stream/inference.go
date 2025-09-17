package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image/jpeg"
	"io/ioutil"
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
	if response.Errno != 0 && response.Errno != -4 {
		return nil, fmt.Errorf("inference failed: %s (errno: %d)", response.ErrMsg, response.Errno)
	}
	// NO_OBJECT_DETECTED
	if response.Errno == -4 {
		return []Detection{}, nil
	}

	// Get actual image dimensions for coordinate conversion
	imgReader := bytes.NewReader(imageData)
	img, err := jpeg.DecodeConfig(imgReader)
	if err != nil {
		Warn(fmt.Sprintf("failed to decode image config, using defaults: %v", err))
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
			Warn("filter out the class `fire` from the smokexfire inference server.")
			continue
		}
		if modelType == "fire" && *className == "smoke" {
			Warn("filter out the class `smoke` from the smokexfire inference server.")
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
			Info(fmt.Sprintf("det: %f , cls: %f", *result.DetScore, *result.ClsScore))
		} else if validRegularScore {
			confidence = result.Score
		} else {
			// Skip invalid detection result silently
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
			Warn(fmt.Sprintf("skipping invalid box coordinates: (%d,%d,%d,%d)", x1, y1, x2, y2))
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

// ModelResult represents detection results for a specific model
type ModelResult struct {
	ModelType  string      `json:"model_type"`
	ServerID   string      `json:"server_id"`
	Detections []Detection `json:"detections"`
	// image sent to their platform.
	DisplayResultImage []byte `json:"display_image"`
	// used for displayed on debug platform.
	DisplayDebugImage []byte `json:"debug_img"`
	OriginalImage     []byte `json:"-"` // Original image without detection boxes (for DEBUG mode)
	Error             error  `json:"-"`
}

// ProcessFrameWithMultipleInference processes a frame with multiple inference servers concurrently
func ProcessFrameWithMultipleInference(frameData []byte, cameraConfig *CameraConfig, manager *RTSPManager) (map[string]*ModelResult, error) {
	results := make(map[string]*ModelResult)

	// Prepare channel for collecting results
	type inferenceResult struct {
		modelType string
		result    *ModelResult
	}
	resultChan := make(chan inferenceResult, len(cameraConfig.InferenceServerBindings))
	var goroutineCount int

	// Launch goroutines for each inference server
	for _, binding := range cameraConfig.InferenceServerBindings {
		server, exists := dataStore.InferenceServers[binding.ServerID]
		if !exists {
			Warn(fmt.Sprintf("inference server %s not found for camera %s", binding.ServerID, cameraConfig.ID))
			continue
		}
		if !server.Enabled {
			Warn(fmt.Sprintf("skipping disabled inference server %s", server.Name))
			continue
		}

		// For fall detection, skip frame-based processing since we now use active polling
		// The polling mechanism handles result retrieval independently
		if server.ModelType == "fall" {
			// Skip - results are handled by polling goroutine in webserver
			continue
		}

		// Other models: launch concurrent inference
		goroutineCount++
		go func(s *InferenceServer, b InferenceServerBinding) {
			// Create frame data copy for this goroutine
			// TODO: shit codes.
			frameDataCopy := make([]byte, len(frameData))
			copy(frameDataCopy, frameData)
			frameDataCopy2 := make([]byte, len(frameData))
			copy(frameDataCopy2, frameData)

			// Process inference
			detections := processInferenceServer(frameDataCopy, s, &b, cameraConfig.ID)

			// Draw detections on image copy (without confidence labels, without camera name overlay)
			displayedImage, err := DrawDetectionsWithServerInfo(frameDataCopy, detections, cameraConfig.Name, false, s.Name)
			if err != nil {
				Warn(fmt.Sprintf("failed to draw results for model %q: %v", s.ModelType, err))
			}
			// with confidence labels and server info
			debugImage, err := DrawDetectionsWithServerInfo(frameDataCopy2, detections, cameraConfig.Name, true, s.Name)
			if err != nil {
				Warn(fmt.Sprintf("failed to draw debug image for model %q: %v", s.ModelType, err))
			}

			// Store original image copy for DEBUG mode
			var originalImageCopy []byte
			if globalDebugMode {
				originalImageCopy = make([]byte, len(frameDataCopy))
				copy(originalImageCopy, frameDataCopy)
			}

			modelResult := &ModelResult{
				ModelType:          s.ModelType,
				ServerID:           b.ServerID,
				Detections:         detections,
				DisplayResultImage: displayedImage,
				DisplayDebugImage:  debugImage,
				OriginalImage:      originalImageCopy,
				Error:              nil,
			}

			resultChan <- inferenceResult{modelType: s.ModelType, result: modelResult}
		}(server, binding)
	}

	// Collect all results
	// TODO: Stupid sync.
	for i := 0; i < goroutineCount; i++ {
		result := <-resultChan
		if result.result != nil {
			results[result.modelType] = result.result
		}
	}

	return results, nil
}

// This function never returns nil !!!
func processInferenceServer(frameData []byte, server *InferenceServer, binding *InferenceServerBinding, cameraID string) []Detection {
	client, err := NewInferenceClient(server.URL)
	if err != nil {
		Warn(fmt.Sprintf("failed to create client for server %s: %v", server.Name, err))
		return []Detection{}
	}

	detections := []Detection{}

	// Process based on model type
	// TODO: enum instead of hardcoding.
	if server.ModelType == "fall" {
		// Fall detection is handled separately in ProcessFrameWithMultipleInference
		return []Detection{}
	} else {
		detections, err = client.DetectObjects(frameData, server.ModelType)
		if err != nil {
			Warn(fmt.Sprintf("inference failed for server %s: %v", server.Name, err))
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
// getClassIndexFromModelType maps modelType to YOLO class index
// This function is used to generate YOLO format labels in DEBUG mode
func getClassIndexFromModelType(modelType string) int {
	// class index mapping for YOLO dataset labels
	// based on the user's requirement: other, gesture, ponding, mouse, tshirt, cigar
	classMap := map[string]int{
		"other":   0,
		"gesture": 1,
		"ponding": 2,
		"smoke":   3,
		"mouse":   4,
		"tshirt":  5,
		"cigar":   6,
		"helmet":  7,
	}

	if classIndex, exists := classMap[modelType]; exists {
		return classIndex
	}

	// default to 'other' category if model type not found
	return 0
}

func SendAlertIfConfigured(imageData []byte, modelType, cameraName string, score, x1, y1, x2, y2 float64) error {
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
		Model:     modelType,
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

	Info(fmt.Sprintf("alert sent successfully to platform for camera %s (model: %s, score: %.3f)",
		cameraName, modelType, score))

	return nil
}
