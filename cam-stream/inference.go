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

		detection := Detection{
			Class:      "gesture", // Based on your endpoint
			Confidence: result.Score,
			X1:         x1,
			Y1:         y1,
			X2:         x2,
			Y2:         y2,
		}
		detections = append(detections, detection)
	}

	log.Printf("Detected %d objects from inference server", len(detections))

	// FOR TESTING: Add a fake detection box if no detections found
	if len(detections) == 0 {
		// Use actual image dimensions for test box
		testDetection := Detection{
			Class:      "test_gesture",
			Confidence: 0.99,
			X1:         img.Width / 4,      // 25% from left
			Y1:         img.Height / 4,     // 25% from top
			X2:         3 * img.Width / 4,  // 75% from left
			Y2:         3 * img.Height / 4, // 75% from top
		}
		detections = append(detections, testDetection)
		log.Printf("Added test detection box for debugging at (%d,%d,%d,%d) on %dx%d image",
			testDetection.X1, testDetection.Y1, testDetection.X2, testDetection.Y2, img.Width, img.Height)
	}

	return detections, nil
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

	// Detect objects
	detections, err := client.DetectObjects(frameData, cameraConfig.ModelType)
	if err != nil {
		log.Printf("Warning: Inference failed for camera %s: %v", cameraConfig.ID, err)
		// Still draw timestamp overlay even if inference fails
		processedFrame, _ := DrawDetections(frameData, []Detection{}, cameraConfig.Name)
		return processedFrame, false, nil // No detections, don't save
	}

	// Filter out test detections for saving decision
	realDetections := make([]Detection, 0)
	for _, det := range detections {
		if det.Class != "test_gesture" {
			realDetections = append(realDetections, det)
		}
	}

	// Draw detections on frame (including test detections for display)
	processedFrame, err := DrawDetections(frameData, detections, cameraConfig.Name)
	if err != nil {
		log.Printf("Warning: Failed to draw detections for camera %s: %v", cameraConfig.ID, err)
		return frameData, false, nil // Return original frame on error
	}

	// Only save if there are real detections (not test)
	hasRealDetections := len(realDetections) > 0
	if hasRealDetections {
		log.Printf("Camera %s: Found %d real detections, will save frame", cameraConfig.ID, len(realDetections))
	} else {
		log.Printf("Camera %s: No real detections found, skipping save", cameraConfig.ID)
	}

	return processedFrame, hasRealDetections, nil
}

// Helper function to check if string contains substring
func contains(s, substr string) bool {
	return bytes.Contains([]byte(s), []byte(substr))
}
