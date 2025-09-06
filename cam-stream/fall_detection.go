package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"time"
)

// FallDetectionStartRequest represents the request to start fall detection
type FallDetectionStartRequest struct {
	RTSPAddress string `json:"rtsp_address"`
	Device      string `json:"device"`
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

// FallDetectionResult represents a single fall detection result
type FallDetectionResultItem struct {
	Image   string      `json:"image"`   // Base64 encoded image
	Results interface{} `json:"results"` // Detection results
}

// StartFallDetection starts fall detection for a camera using the specified inference server
func StartFallDetection(server *InferenceServer, camera *CameraConfig) (string, error) {
	client := &http.Client{Timeout: 30 * time.Second}

	// Send start request to tianwan service
	startReq := FallDetectionStartRequest{
		RTSPAddress: camera.RTSPUrl,
		Device:      "cuda", // Default to CUDA, could be configurable
	}

	reqBody, err := json.Marshal(startReq)
	if err != nil {
		return "", fmt.Errorf("failed to marshal start request: %v", err)
	}

	url := fmt.Sprintf("%s/fall/start", server.URL)
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

	slog.Info("started fall detection for camera", "camera_name", camera.Name, "camera_id", camera.ID, "task_id", startResp.TaskID)
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

	url := fmt.Sprintf("%s/fall/stop", server.URL)
	resp, err := client.Post(url, "application/json", bytes.NewBuffer(reqBody))
	if err != nil {
		return fmt.Errorf("failed to send stop request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("tianwan service returned status %d: %s", resp.StatusCode, string(body))
	}

	slog.Info("stopped fall detection", "task_id", taskID)
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

	url := fmt.Sprintf("%s/fall/result", server.URL)
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

	var results []FallDetectionResultItem
	if err := json.Unmarshal(body, &results); err != nil {
		return nil, fmt.Errorf("failed to parse results: %v", err)
	}

	return results, nil
}
