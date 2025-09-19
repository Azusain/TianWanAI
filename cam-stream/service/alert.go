package service

import (
	"bytes"
	"cam-stream/common"
	"cam-stream/common/log"
	"cam-stream/common/store"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image/jpeg"
	"io"
	"net/http"
	"time"

	"github.com/google/uuid"
)

// TODO: move these codes to alert.go.
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
func SendAlertIfConfigured(imageData []byte, modelType, cameraName string, score, x1, y1, x2, y2 float64) error {
	// Check if alert system is enabled and configured globally using thread-safe access
	var alertServerURL string
	var alertEnabled bool
	store.SafeReadDataStore(func() {
		// TODO: this callback is not elegant.
		// It should be with args.
		if store.Data.AlertServer != nil {
			alertEnabled = store.Data.AlertServer.Enabled
			alertServerURL = store.Data.AlertServer.URL
		}
	})

	if !alertEnabled || alertServerURL == "" {
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

	// Create HTTP client (Connection: Close)
	// TODO: this can be optimized by using keep-alive and reusing a global client.
	// But I am a LAZY BONE.
	client := &http.Client{
		Timeout: time.Duration(DefaultHttpTimeoutSecs) * time.Second,
		Transport: &http.Transport{
			DisableKeepAlives: true,
			MaxIdleConns:      0,
			IdleConnTimeout:   0,
		},
	}

	// Send request
	req, err := http.NewRequest("POST", alertServerURL, bytes.NewBuffer(requestBody))
	if err != nil {
		return fmt.Errorf("failed to create alert request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Connection", "close")

	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to send alert to platform: %v", err)
	}
	defer resp.Body.Close()

	// Check response status
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("platform returned status %d: %s", resp.StatusCode, string(body))
	}

	log.Info(fmt.Sprintf("alert sent successfully to platform for camera %s (model: %s, score: %.3f)",
		cameraName, modelType, score))

	return nil
}

// sendDetectionAlerts sends alerts for all detections in the result
func sendDetectionAlerts(imageData []byte, detections []common.Detection, cameraName, modelType string) {
	// Get the real size of the image
	img, err := jpeg.DecodeConfig(bytes.NewReader(imageData))
	if err != nil {
		log.Warn(fmt.Sprintf("failed to decode image config for alerts: %v", err))
		return
	}

	for _, detection := range detections {
		// Normalize coordinates
		x1 := float64(detection.X1) / float64(img.Width)
		y1 := float64(detection.Y1) / float64(img.Height)
		x2 := float64(detection.X2) / float64(img.Width)
		y2 := float64(detection.Y2) / float64(img.Height)

		if err := SendAlertIfConfigured(imageData, modelType, cameraName, detection.Confidence, x1, y1, x2, y2); err != nil {
			log.Warn(fmt.Sprintf("failed to send alert for detection %s: %v", detection.Class, err))
		} else {
			log.Info(fmt.Sprintf("sent alert for detection %s (confidence: %.3f) from camera %s", detection.Class, detection.Confidence, cameraName))
		}
	}
}
