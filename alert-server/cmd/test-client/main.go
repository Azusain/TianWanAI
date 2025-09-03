package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"time"
)

func main() {
	// Create a test alert request
	testAlert := map[string]interface{}{
		"image":      "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0a", // Sample base64 image
		"request_id": "test-12345",
		"model":      "yolo-v8",
		"camera_kks": "TEST-CAM-001",
		"score":      0.85,
		"x1":         0.1,
		"y1":         0.2,
		"x2":         0.4,
		"y2":         0.6,
		"timestamp":  time.Now().Format("2006-01-02T15:04:05+08:00"),
	}

	// Convert to JSON
	jsonData, err := json.Marshal(testAlert)
	if err != nil {
		fmt.Printf("Failed to marshal test alert: %v\n", err)
		return
	}

	// Send POST request to alert server
	resp, err := http.Post("http://localhost:8080/alert", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		fmt.Printf("Failed to send alert request: %v\n", err)
		fmt.Println("Make sure the alert server is running: go run main.go")
		return
	}
	defer resp.Body.Close()

	// Read response
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		fmt.Printf("Failed to read response: %v\n", err)
		return
	}

	fmt.Printf("Response Status: %d\n", resp.StatusCode)
	fmt.Printf("Response Body: %s\n", string(body))

	if resp.StatusCode == 200 {
		fmt.Println("\n✅ Test alert sent successfully!")
		fmt.Println("Check the received_alerts/ directory for the saved JSON file.")
	} else {
		fmt.Println("\n❌ Test alert failed!")
	}
}
