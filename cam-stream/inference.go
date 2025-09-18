package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image/jpeg"
	"io"
	"io/ioutil"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/pkg/errors"
)

const (
	// TODO: make it configurable.
	DefaultHttpTimeoutSecs int = 15
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

func NewInferenceClient(serverUrl string) (*InferenceClient, error) {
	if serverUrl == "" {
		return nil, errors.New("server url should not be empty")
	}
	// TODO: this may waste a lot of time.
	// But it will be replaced by gRPC.
	transport := &http.Transport{
		DisableKeepAlives: true,
		MaxIdleConns:      0,
		IdleConnTimeout:   0,
	}
	return &InferenceClient{
		serverURL: serverUrl,
		client: &http.Client{
			Transport: transport,
			Timeout:   time.Duration(DefaultHttpTimeoutSecs) * time.Second,
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
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(requestBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Connection", "close")

	resp, err := ic.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request to inference server: %v", err)
	}
	defer resp.Body.Close()

	// Check HTTP status
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		if len(body) > 0 && body[0] == '<' {
			return nil, fmt.Errorf("inference server returned HTML (status %d) - check if service is running at %s", resp.StatusCode, url)
		}
		return nil, fmt.Errorf("inference server returned status %d: %s", resp.StatusCode, string(body))
	}

	// Read response
	body, err := io.ReadAll(resp.Body)
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
		if modelType == string(ModelTypeSmoke) && *className == string(ModelTypeFire) {
			Warn("filtering out fire class from smoke detection server")
			continue
		}
		if modelType == string(ModelTypeFire) && *className == string(ModelTypeSmoke) {
			Warn("filtering out smoke class from fire detection server")
			continue
		}

		confidence := 0.0
		// TODO: use enumerate instead.
		validRegularScore := modelType != string(ModelTypeTshirt) && result.Score > 0 && result.Location.Left > 0
		validTshirtScore := modelType == string(ModelTypeTshirt) && result.DetScore != nil && result.ClsScore != nil && *result.DetScore > 0 && *result.ClsScore > 0
		if validTshirtScore {
			// Use classification score for tshirt detection confidence
			confidence = *result.ClsScore
			Info(fmt.Sprintf("tshirt detection scores - det: %.3f, cls: %.3f", *result.DetScore, *result.ClsScore))
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

// ProcessFrameWithAsyncInference processes a frame with multiple inference servers asynchronously
// This function launches each inference server processing in its own goroutine and returns immediately
// Each goroutine handles the entire pipeline: inference -> draw -> save -> alert
func ProcessFrameWithAsyncInference(frameData []byte, cameraConfig *CameraConfig, outputDir string) {
	// Launch independent goroutines for each inference server
	for _, binding := range cameraConfig.InferenceServerBindings {
		// Use thread-safe access to get server information
		server, exists := safeGetInferenceServer(binding.ServerID)
		if !exists {
			Warn(fmt.Sprintf("inference server %s not found for camera %s", binding.ServerID, cameraConfig.ID))
			continue
		}
		if !server.Enabled {
			Warn(fmt.Sprintf("skipping disabled inference server %s", server.Name))
			continue
		}

		// For fall detection, skip frame-based processing since we now use active polling
		if server.ModelType == string(ModelTypeFall) {
			continue
		}

		// Launch independent async processing for each server
		go processInferenceServerAsync(frameData, server, &binding, cameraConfig, outputDir)
	}
}

// processInferenceServerAsync handles the complete pipeline for a single inference server asynchronously
func processInferenceServerAsync(frameData []byte, server *InferenceServer, binding *InferenceServerBinding, cameraConfig *CameraConfig, outputDir string) {
	// Create frame data copies for this goroutine to avoid race conditions
	frameDataCopy := make([]byte, len(frameData))
	copy(frameDataCopy, frameData)
	frameDataCopy2 := make([]byte, len(frameData))
	copy(frameDataCopy2, frameData)

	detections := getResultFromInferenceServer(frameDataCopy, server, binding, cameraConfig.ID)
	if len(detections) == 0 {
		return
	}

	// Draw detections on image copy (without confidence labels)
	displayedImage, err := DrawDetectionsWithServerInfo(frameDataCopy, detections, cameraConfig.Name, false, server.Name)
	if err != nil {
		Warn(fmt.Sprintf("failed to draw results for model %q: %v", server.ModelType, err))
		return
	}
	// Draw debug image with confidence labels and server info
	// TODO: temporarily controlled by `globalDebugMode`.
	debugImage, err := DrawDetectionsWithServerInfo(frameDataCopy2, detections, cameraConfig.Name, globalDebugMode, server.Name)
	if err != nil {
		Warn(fmt.Sprintf("failed to draw debug image for model %q: %v", server.ModelType, err))
		return
	}

	var originalImageCopy []byte
	if globalDebugMode {
		originalImageCopy = make([]byte, len(frameDataCopy))
		copy(originalImageCopy, frameDataCopy)
	}

	modelResult := &ModelResult{
		ModelType:          server.ModelType,
		ServerID:           binding.ServerID,
		Detections:         detections,
		DisplayResultImage: displayedImage,
		DisplayDebugImage:  debugImage,
		OriginalImage:      originalImageCopy,
		Error:              nil,
	}

	// save result and send alerts at the same time.
	go func() {
		saveModelResult(cameraConfig.Name, modelResult, outputDir)
		alertImageData := make([]byte, len(modelResult.DisplayResultImage))
		copy(alertImageData, modelResult.DisplayResultImage)
		sendDetectionAlerts(alertImageData, modelResult.Detections, cameraConfig.Name, modelResult.ModelType)
	}()

}

// saveModelResult saves a single model result to file
func saveModelResult(cameraName string, result *ModelResult, outputDir string) {
	// For fall detection, ensure exactly one detection
	if result.ModelType == string(ModelTypeFall) && len(result.Detections) != 1 {
		Warn(fmt.Sprintf("fall detection ModelResult should contain exactly one detection, got %d detections, skipping", len(result.Detections)))
		return
	}

	// Generate filename and paths
	timestamp := time.Now().Format("20060102_150405")
	filename := fmt.Sprintf("%s_%s_detection.jpg", timestamp, result.ModelType)
	serverDir := fmt.Sprintf("%s/%s", outputDir, result.ServerID)

	if err := os.MkdirAll(serverDir, 0755); err != nil {
		Warn(fmt.Sprintf("failed to create directory for server %s: %v", result.ServerID, err))
		return
	}

	filePath := fmt.Sprintf("%s/%s", serverDir, filename)
	if err := os.WriteFile(filePath, result.DisplayDebugImage, 0644); err != nil {
		Warn(fmt.Sprintf("failed to save detection image for model %s: %v", result.ModelType, err))
		return
	}

	Info(fmt.Sprintf("saved detection image for camera %s, model %s to %s (detections: %d)",
		cameraName, result.ModelType, filePath, len(result.Detections)))

	// Save debug data if enabled
	saveDebugDataAsync(result, filename, outputDir)
}

// saveDebugDataAsync saves original image and YOLO labels for DEBUG mode
func saveDebugDataAsync(result *ModelResult, filename, outputDir string) {
	if !globalDebugMode || result.OriginalImage == nil {
		return
	}

	// Create debug server directory
	debugServerDir := fmt.Sprintf("%s/%s", DebugDir, result.ServerID)
	if err := os.MkdirAll(debugServerDir, 0755); err != nil {
		Warn(fmt.Sprintf("failed to create debug directory for server %s: %v", result.ServerID, err))
		return
	}

	// Save original image
	debugFilePath := fmt.Sprintf("%s/%s", debugServerDir, filename)
	if err := os.WriteFile(debugFilePath, result.OriginalImage, 0644); err != nil {
		Warn(fmt.Sprintf("failed to save debug original image: %v", err))
		return
	}

	Info(fmt.Sprintf("saved debug original image to %s", debugFilePath))

	// Save YOLO format labels if there are detections
	if len(result.Detections) > 0 {
		saveYoloLabelsAsync(result, debugServerDir, filename)
	}
}

// saveYoloLabelsAsync saves YOLO format label file
func saveYoloLabelsAsync(result *ModelResult, debugDir, filename string) {
	// Get image dimensions for normalization
	imgCfg, err := jpeg.DecodeConfig(bytes.NewReader(result.OriginalImage))
	if err != nil {
		Warn(fmt.Sprintf("failed to decode image config for yolo label: %v", err))
		return
	}

	w := float64(imgCfg.Width)
	h := float64(imgCfg.Height)
	classIdx := getClassIndexFromModelType(result.ModelType)

	// Build YOLO format lines
	lines := make([]string, 0, len(result.Detections))
	for _, det := range result.Detections {
		// Convert pixel coordinates to normalized YOLO format
		cx := (float64(det.X1) + float64(det.X2)) / 2.0 / w
		cy := (float64(det.Y1) + float64(det.Y2)) / 2.0 / h
		bw := (float64(det.X2) - float64(det.X1)) / w
		bh := (float64(det.Y2) - float64(det.Y1)) / h

		// Clamp values between 0 and 1
		if cx < 0 {
			cx = 0
		} else if cx > 1 {
			cx = 1
		}
		if cy < 0 {
			cy = 0
		} else if cy > 1 {
			cy = 1
		}
		if bw < 0 {
			bw = 0
		} else if bw > 1 {
			bw = 1
		}
		if bh < 0 {
			bh = 0
		} else if bh > 1 {
			bh = 1
		}

		lines = append(lines, fmt.Sprintf("%d %.6f %.6f %.6f %.6f", classIdx, cx, cy, bw, bh))
	}

	// Save label file
	baseName := strings.TrimSuffix(filename, ".jpg")
	labelPath := fmt.Sprintf("%s/%s.txt", debugDir, baseName)
	labelContent := strings.Join(lines, "\n")

	if err := os.WriteFile(labelPath, []byte(labelContent), 0644); err != nil {
		Warn(fmt.Sprintf("failed to save yolo label: %v", err))
	} else {
		Info(fmt.Sprintf("saved yolo label to %s", labelPath))
	}
}

// This function never returns nil !!!
func getResultFromInferenceServer(frameData []byte, server *InferenceServer, binding *InferenceServerBinding, cameraID string) []Detection {
	client, err := NewInferenceClient(server.URL)
	if err != nil {
		Warn(fmt.Sprintf("failed to create client for server %s: %v", server.Name, err))
		return []Detection{}
	}
	detections := []Detection{}

	// Process based on model type
	if server.ModelType == string(ModelTypeFall) {
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

// ModelType represents supported model types
type ModelType string

// Supported model types as constants
const (
	ModelTypeOther      ModelType = "other"
	ModelTypeGesture    ModelType = "gesture"
	ModelTypePonding    ModelType = "ponding"
	ModelTypeSmoke      ModelType = "smoke"
	ModelTypeMouse      ModelType = "mouse"
	ModelTypeTshirt     ModelType = "tshirt"
	ModelTypeCigar      ModelType = "cigar"
	ModelTypeHelmet     ModelType = "helmet"
	ModelTypeFire       ModelType = "fire"
	ModelTypeFall       ModelType = "fall"
	ModelTypeSafetybelt ModelType = "safetybelt"
)

// getClassIndexFromModelType maps modelType to YOLO class index
// This function is used to generate YOLO format labels in DEBUG mode
func getClassIndexFromModelType(modelType string) int {
	classMap := map[string]int{
		"other":      0,
		"gesture":    1,
		"ponding":    2,
		"smoke":      3,
		"mouse":      4,
		"tshirt":     5,
		"cigar":      6,
		"helmet":     7,
		"fire":       8,
		"fall":       9,
		"safetybelt": 10,
	}
	if classIndex, exists := classMap[modelType]; exists {
		return classIndex
	}
	return classMap["other"]
}

// SendAlertIfConfigured sends detection alert to management platform using global configuration
func SendAlertIfConfigured(imageData []byte, modelType, cameraName string, score, x1, y1, x2, y2 float64) error {
	// Check if alert system is enabled and configured globally using thread-safe access
	var alertServerURL string
	var alertEnabled bool
	safeReadDataStore(func() {
		if dataStore.AlertServer != nil {
			alertEnabled = dataStore.AlertServer.Enabled
			alertServerURL = dataStore.AlertServer.URL
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
		body, _ := ioutil.ReadAll(resp.Body)
		return fmt.Errorf("platform returned status %d: %s", resp.StatusCode, string(body))
	}

	Info(fmt.Sprintf("alert sent successfully to platform for camera %s (model: %s, score: %.3f)",
		cameraName, modelType, score))

	return nil
}

// sendDetectionAlerts sends alerts for all detections in the result
func sendDetectionAlerts(imageData []byte, detections []Detection, cameraName, modelType string) {
	// Get the real size of the image
	img, err := jpeg.DecodeConfig(bytes.NewReader(imageData))
	if err != nil {
		Warn(fmt.Sprintf("failed to decode image config for alerts: %v", err))
		return
	}

	for _, detection := range detections {
		// Normalize coordinates
		x1 := float64(detection.X1) / float64(img.Width)
		y1 := float64(detection.Y1) / float64(img.Height)
		x2 := float64(detection.X2) / float64(img.Width)
		y2 := float64(detection.Y2) / float64(img.Height)

		if err := SendAlertIfConfigured(imageData, modelType, cameraName, detection.Confidence, x1, y1, x2, y2); err != nil {
			Warn(fmt.Sprintf("failed to send alert for detection %s: %v", detection.Class, err))
		} else {
			Info(fmt.Sprintf("sent alert for detection %s (confidence: %.3f) from camera %s", detection.Class, detection.Confidence, cameraName))
		}
	}
}
