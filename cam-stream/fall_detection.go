package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// FallDetectionStartRequest represents the request to start fall detection
type FallDetectionStartRequest struct {
	RTSPAddress string `json:"rtsp_address"`
}

// FallDetectionStartResponse represents the response from starting fall detection
type FallDetectionStartResponse struct {
	ErrMsg string `json:"err_msg"`
	TaskID string `json:"task_id"`
}

// FallDetectionStopRequest represents the request to stop fall detection
type FallDetectionStopRequest struct {
	TaskID string `json:"task_id"`
}

// FallDetectionResultRequest represents the request to get fall detection results
type FallDetectionResultRequest struct {
	TaskID string `json:"task_id"`
	Limit  *int   `json:"limit,omitempty"`
}

// FallDetectionResultItem represents a single fall detection result from tianwan API
type FallDetectionResultItem struct {
	Image   string                   `json:"image"`   // Base64 encoded image
	Results FallDetectionResultData `json:"results"` // Detection results
}

// FallDetectionResultData represents the detection data structure
type FallDetectionResultData struct {
	Score    float64                     `json:"score"`    // Confidence score
	Location FallDetectionResultLocation `json:"location"` // Bounding box location
}

// FallDetectionResultLocation represents the bounding box location
type FallDetectionResultLocation struct {
	Left   float64 `json:"left"`   // X coordinate (pixel)
	Top    float64 `json:"top"`    // Y coordinate (pixel)
	Width  float64 `json:"width"`  // Width (pixel)
	Height float64 `json:"height"` // Height (pixel)
}

// StartFallDetection starts fall detection for a camera using the specified inference server
func StartFallDetection(server *InferenceServer, camera *CameraConfig) (string, error) {
	client := &http.Client{Timeout: 30 * time.Second}

	// Send start request to tianwan service
	startReq := FallDetectionStartRequest{
		RTSPAddress: camera.RTSPUrl,
	}

	reqBody, err := json.Marshal(startReq)
	if err != nil {
		return "", fmt.Errorf("failed to marshal start request: %v", err)
	}

	url := fmt.Sprintf("%s/start", server.URL)
	resp, err := client.Post(url, "application/json", bytes.NewBuffer(reqBody))
	if err != nil {
		return "", fmt.Errorf("failed to send start request: %v", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response: %v", err)
	}

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("tianwan service returned status %d: %s", resp.StatusCode, string(body))
	}

	var startResp FallDetectionStartResponse
	if err := json.Unmarshal(body, &startResp); err != nil {
		return "", fmt.Errorf("failed to parse response: %v", err)
	}

	if startResp.ErrMsg != "success" {
		return "", fmt.Errorf("tianwan service error: %s", startResp.ErrMsg)
	}

	AsyncInfo(fmt.Sprintf("started fall detection for camera: camera_name=%s camera_id=%s task_id=%s", camera.Name, camera.ID, startResp.TaskID))
	return startResp.TaskID, nil
}

// StopFallDetection stops a fall detection task
func StopFallDetection(server *InferenceServer, taskID string) error {
	client := &http.Client{Timeout: 30 * time.Second}

	// Send stop request to tianwan service
	stopReq := FallDetectionStopRequest{
		TaskID: taskID,
	}

	reqBody, err := json.Marshal(stopReq)
	if err != nil {
		return fmt.Errorf("failed to marshal stop request: %v", err)
	}

	url := fmt.Sprintf("%s/stop", server.URL)
	resp, err := client.Post(url, "application/json", bytes.NewBuffer(reqBody))
	if err != nil {
		return fmt.Errorf("failed to send stop request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("tianwan service returned status %d: %s", resp.StatusCode, string(body))
	}

	AsyncInfo(fmt.Sprintf("stopped fall detection: task_id=%s", taskID))
	return nil
}

// GetFallDetectionResults retrieves fall detection results from tianwan service
func GetFallDetectionResults(server *InferenceServer, taskID string, limit *int) ([]FallDetectionResultItem, error) {
	client := &http.Client{Timeout: 30 * time.Second}

	// Send result request to tianwan service
	resultReq := FallDetectionResultRequest{
		TaskID: taskID,
		Limit:  limit,
	}

	reqBody, err := json.Marshal(resultReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal result request: %v", err)
	}

	url := fmt.Sprintf("%s/result", server.URL)
	resp, err := client.Post(url, "application/json", bytes.NewBuffer(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to send result request: %v", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %v", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("tianwan service returned status %d: %s", resp.StatusCode, string(body))
	}

	// Parse as object with "results" field - now unified API format
	var response struct {
		Results []FallDetectionResultItem `json:"results"`
		Error   string                    `json:"error,omitempty"`
	}
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("failed to parse tianwan /result response: %v, body: %s", err, string(body))
	}

	if response.Error != "" {
		return nil, fmt.Errorf("tianwan service error: %s", response.Error)
	}

	if len(response.Results) > 0 {
	AsyncInfo(fmt.Sprintf("got fall detection results: count=%d task_id=%s", len(response.Results), taskID))
	}

	return response.Results, nil
}
