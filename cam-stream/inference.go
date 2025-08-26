package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/jpeg"
	"net/http"
	"time"
)

// InferenceRequest inference request structure
type InferenceRequest struct {
	Image    string `json:"image"`
	Filename string `json:"filename"`
}

// InferenceResponse inference response structure
type InferenceResponse struct {
	Errno   int                      `json:"errno"`
	ErrMsg  string                   `json:"err_msg,omitempty"`
	Results []InferenceDetectionResult `json:"results"`
}

// InferenceDetectionResult detection result
type InferenceDetectionResult struct {
	Score    float64                   `json:"score"`
	Location InferenceDetectionLocation `json:"location"`
	Class    string                    `json:"class,omitempty"` // optional classification info
}

// InferenceDetectionLocation detection location (normalized coordinates)
type InferenceDetectionLocation struct {
	Left   float64 `json:"left"`
	Top    float64 `json:"top"`
	Width  float64 `json:"width"`
	Height float64 `json:"height"`
}

// InferenceClient AI inference client
type InferenceClient struct {
	serverURL  string
	httpClient *http.Client
}

// NewInferenceClient creates inference client
func NewInferenceClient(serverURL string) *InferenceClient {
	return &InferenceClient{
		serverURL: serverURL,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// imageToBase64 converts image to base64 string
func (c *InferenceClient) imageToBase64(img image.Image) (string, error) {
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: 85}); err != nil {
		return "", fmt.Errorf("failed to encode image to JPEG: %v", err)
	}
	
	encoded := base64.StdEncoding.EncodeToString(buf.Bytes())
	return encoded, nil
}

// InferImage performs inference on image
func (c *InferenceClient) InferImage(img image.Image, filename string) (*InferenceResponse, error) {
	// convert image to base64
	base64Image, err := c.imageToBase64(img)
	if err != nil {
		return nil, fmt.Errorf("failed to convert image to base64: %v", err)
	}

	// create request
	request := InferenceRequest{
		Image:    base64Image,
		Filename: filename,
	}

	// serialize request
	requestData, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %v", err)
	}

	// send HTTP request
	resp, err := c.httpClient.Post(c.serverURL, "application/json", bytes.NewReader(requestData))
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %v", err)
	}
	defer resp.Body.Close()

	// check HTTP status code
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("server returned error status: %d", resp.StatusCode)
	}

	// parse response
	var response InferenceResponse
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return nil, fmt.Errorf("failed to decode response: %v", err)
	}

	// check business error
	if response.Errno != 0 {
		return nil, fmt.Errorf("inference error: %s (code: %d)", response.ErrMsg, response.Errno)
	}

	return &response, nil
}

// TestConnection tests connection to server
func (c *InferenceClient) TestConnection() error {
	resp, err := c.httpClient.Get(c.serverURL + "/health")
	if err != nil {
		return fmt.Errorf("failed to connect to server: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		// server exists but no health endpoint, this is normal
		return nil
	}

	return nil
}
