package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
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
	Success    bool        `json:"success"`
	Detections []Detection `json:"detections"`
	Message    string      `json:"message"`
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
	if url[len(url)-1] != '/' && !contains(url, "/detect") && !contains(url, "/gesture") {
		url = url + "/detect"
	}

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

	if !response.Success {
		return nil, fmt.Errorf("inference failed: %s", response.Message)
	}

	log.Printf("Detected %d objects", len(response.Detections))
	return response.Detections, nil
}

// ProcessFrame processes a frame with inference
func ProcessFrameWithInference(frameData []byte, cameraConfig *CameraConfig) ([]byte, error) {
	// Skip inference if no server URL configured
	if cameraConfig.ServerUrl == "" {
		return frameData, nil
	}

	// Create inference client
	client := NewInferenceClient(cameraConfig.ServerUrl)

	// Detect objects
	detections, err := client.DetectObjects(frameData, cameraConfig.ModelType)
	if err != nil {
		log.Printf("Warning: Inference failed for camera %s: %v", cameraConfig.ID, err)
		// Still draw timestamp overlay even if inference fails
		processedFrame, _ := DrawDetections(frameData, []Detection{}, cameraConfig.Name)
		return processedFrame, nil
	}

	// Draw detections on frame
	processedFrame, err := DrawDetections(frameData, detections, cameraConfig.Name)
	if err != nil {
		log.Printf("Warning: Failed to draw detections for camera %s: %v", cameraConfig.ID, err)
		return frameData, nil // Return original frame on error
	}

	return processedFrame, nil
}

// Helper function to check if string contains substring
func contains(s, substr string) bool {
	return bytes.Contains([]byte(s), []byte(substr))
}
