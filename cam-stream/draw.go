package main

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"time"
)

// DrawDetections draws detection boxes on the image
func DrawDetections(imageData []byte, detections []Detection, cameraName string) ([]byte, error) {
	// Decode JPEG
	img, err := jpeg.Decode(bytes.NewReader(imageData))
	if err != nil {
		return nil, fmt.Errorf("failed to decode JPEG: %v", err)
	}

	// Convert to RGBA for drawing
	bounds := img.Bounds()
	rgbaImg := image.NewRGBA(bounds)
	draw.Draw(rgbaImg, bounds, img, bounds.Min, draw.Src)

	// Draw detection boxes
	for _, det := range detections {
		drawDetectionBox(rgbaImg, det)
	}

	// Add timestamp and camera name overlay
	addOverlay(rgbaImg, cameraName, len(detections))

	// Encode back to JPEG
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, rgbaImg, &jpeg.Options{Quality: 90}); err != nil {
		return nil, fmt.Errorf("failed to encode JPEG: %v", err)
	}

	return buf.Bytes(), nil
}

// drawDetectionBox draws a single detection box with label
func drawDetectionBox(img *image.RGBA, det Detection) {
	// Define colors for different classes
	boxColor := getClassColor(det.Class)
	
	// Draw box border (3 pixels wide for visibility)
	drawThickRectangle(img, det.X1, det.Y1, det.X2, det.Y2, boxColor, 3)

	// Draw label background
	labelHeight := 20
	labelBg := color.RGBA{boxColor.R/2, boxColor.G/2, boxColor.B/2, 200} // Darker version with transparency
	for y := det.Y1 - labelHeight; y < det.Y1; y++ {
		if y < 0 {
			continue
		}
		for x := det.X1; x < det.X2 && x < img.Bounds().Max.X; x++ {
			if x < 0 {
				continue
			}
			img.Set(x, y, labelBg)
		}
	}

	// Draw label text (simple pixel representation)
	label := fmt.Sprintf("%s %.2f", det.Class, det.Confidence)
	drawText(img, det.X1+5, det.Y1-15, label, color.RGBA{255, 255, 255, 255})
}

// drawThickRectangle draws a rectangle with specified thickness
func drawThickRectangle(img *image.RGBA, x1, y1, x2, y2 int, col color.RGBA, thickness int) {
	// Top and bottom borders
	for t := 0; t < thickness; t++ {
		for x := x1; x <= x2; x++ {
			if x >= 0 && x < img.Bounds().Max.X {
				// Top
				if y1+t >= 0 && y1+t < img.Bounds().Max.Y {
					img.Set(x, y1+t, col)
				}
				// Bottom
				if y2-t >= 0 && y2-t < img.Bounds().Max.Y {
					img.Set(x, y2-t, col)
				}
			}
		}
	}

	// Left and right borders
	for t := 0; t < thickness; t++ {
		for y := y1; y <= y2; y++ {
			if y >= 0 && y < img.Bounds().Max.Y {
				// Left
				if x1+t >= 0 && x1+t < img.Bounds().Max.X {
					img.Set(x1+t, y, col)
				}
				// Right
				if x2-t >= 0 && x2-t < img.Bounds().Max.X {
					img.Set(x2-t, y, col)
				}
			}
		}
	}
}

// getClassColor returns a color for a detection class
func getClassColor(class string) color.RGBA {
	// Define colors for common classes
	colors := map[string]color.RGBA{
		"person":     {255, 0, 0, 255},     // Red
		"car":        {0, 255, 0, 255},     // Green
		"truck":      {0, 200, 0, 255},     // Dark Green
		"bus":        {0, 150, 0, 255},     // Darker Green
		"bicycle":    {255, 255, 0, 255},   // Yellow
		"motorcycle": {255, 200, 0, 255},   // Orange-Yellow
		"dog":        {255, 0, 255, 255},   // Magenta
		"cat":        {200, 0, 200, 255},   // Purple
		"chair":      {0, 255, 255, 255},   // Cyan
		"bottle":     {0, 200, 255, 255},   // Light Blue
		"cell phone": {128, 128, 255, 255}, // Light Purple
	}

	if col, exists := colors[class]; exists {
		return col
	}

	// Default color for unknown classes
	return color.RGBA{255, 165, 0, 255} // Orange
}

// addOverlay adds timestamp and detection count overlay
func addOverlay(img *image.RGBA, cameraName string, detectionCount int) {
	bounds := img.Bounds()
	width := bounds.Max.X
	height := bounds.Max.Y

	// Calculate overlay size (1/15 of image height as requested)
	overlayHeight := height / 15
	if overlayHeight < 30 {
		overlayHeight = 30 // Minimum height
	}
	if overlayHeight > 100 {
		overlayHeight = 100 // Maximum height
	}

	// Draw semi-transparent black background at the top
	bgColor := color.RGBA{0, 0, 0, 180}
	for y := 0; y < overlayHeight; y++ {
		for x := 0; x < width; x++ {
			img.Set(x, y, bgColor)
		}
	}

	// Add timestamp
	now := time.Now()
	timestamp := now.Format("2006-01-02 15:04:05")
	weekday := getChineseWeekday(now.Weekday())
	fullTimestamp := fmt.Sprintf("%s %s", timestamp, weekday)
	
	// Draw timestamp text (positioned at 10% from left)
	textX := width / 10
	textY := overlayHeight / 3
	drawText(img, textX, textY, fullTimestamp, color.RGBA{255, 255, 255, 255})

	// Add camera name (positioned at center)
	cameraTextX := width / 2 - len(cameraName)*3
	drawText(img, cameraTextX, textY, cameraName, color.RGBA{255, 255, 0, 255})

	// Add detection count (positioned at 80% from left)
	detectionText := fmt.Sprintf("Objects: %d", detectionCount)
	detectionTextX := width * 8 / 10
	drawText(img, detectionTextX, textY, detectionText, color.RGBA{0, 255, 0, 255})
}

// getChineseWeekday returns Chinese weekday name
func getChineseWeekday(weekday time.Weekday) string {
	weekdays := map[time.Weekday]string{
		time.Sunday:    "星期日",
		time.Monday:    "星期一",
		time.Tuesday:   "星期二",
		time.Wednesday: "星期三",
		time.Thursday:  "星期四",
		time.Friday:    "星期五",
		time.Saturday:  "星期六",
	}
	return weekdays[weekday]
}

// drawText draws simple text representation (basic ASCII only for now)
func drawText(img *image.RGBA, x, y int, text string, col color.RGBA) {
	// This is a very basic text drawing function
	// For proper text rendering with Chinese support, you'd need a font library
	// For now, we just draw simple lines to represent text presence
	
	textWidth := len(text) * 6
	textHeight := 10
	
	// Ensure text doesn't go out of bounds
	if x < 0 {
		x = 0
	}
	if y < 0 {
		y = 0
	}
	if x + textWidth > img.Bounds().Max.X {
		textWidth = img.Bounds().Max.X - x
	}
	
	// Draw a simple line to represent text
	for i := 0; i < textWidth && x+i < img.Bounds().Max.X; i++ {
		if y >= 0 && y < img.Bounds().Max.Y {
			img.Set(x+i, y, col)
		}
		if y+1 >= 0 && y+1 < img.Bounds().Max.Y {
			img.Set(x+i, y+1, col)
		}
	}
	
	// For Chinese text, draw a slightly different pattern
	if containsChinese(text) {
		for i := 0; i < textWidth && x+i < img.Bounds().Max.X; i += 3 {
			for j := 0; j < textHeight && y+j < img.Bounds().Max.Y; j += 2 {
				if x+i >= 0 && y+j >= 0 {
					img.Set(x+i, y+j, col)
				}
			}
		}
	}
}

// containsChinese checks if string contains Chinese characters
func containsChinese(s string) bool {
	for _, r := range s {
		if r >= 0x4E00 && r <= 0x9FFF {
			return true
		}
	}
	return false
}
