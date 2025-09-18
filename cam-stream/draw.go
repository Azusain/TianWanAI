package main

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"

	"github.com/fogleman/gg"
)

// DrawDetections draws detection boxes on the image
func DrawDetections(imageData []byte, detections []Detection, cameraName string, saveConfidenceLabel bool) ([]byte, error) {
	return DrawDetectionsWithServerInfo(imageData, detections, cameraName, saveConfidenceLabel, "")
}

// DrawDetectionsWithServerInfo draws detection boxes on the image with server info in confidence labels
func DrawDetectionsWithServerInfo(imageData []byte, detections []Detection, cameraName string, saveConfidenceLabel bool, serverID string) ([]byte, error) {
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
		boxColor := getClassColor(det.Class)
		drawThickRectangle(rgbaImg, det.X1, det.Y1, det.X2, det.Y2, boxColor, 3)
		if !saveConfidenceLabel {
			continue
		}
		drawConfidenceLabelWithServerInfo(rgbaImg, det, serverID)
	}

	// Encode back to JPEG
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, rgbaImg, &jpeg.Options{Quality: 90}); err != nil {
		return nil, fmt.Errorf("failed to encode JPEG: %v", err)
	}

	return buf.Bytes(), nil
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

// drawConfidenceLabelWithServerInfo draws confidence score label with server info
func drawConfidenceLabelWithServerInfo(img *image.RGBA, det Detection, serverID string) {
	// Format confidence as percentage with server info
	var confidenceText string
	if serverID != "" {
		confidenceText = fmt.Sprintf("%.1f%% [%s]", det.Confidence*100, serverID)
	} else {
		confidenceText = fmt.Sprintf("%.1f%%", det.Confidence*100)
	}

	// Position label above the detection box with some padding
	labelX := det.X1 + 5
	labelY := det.Y1 - 5

	// If the label would be outside the image bounds, position it inside the box
	if labelY <= 20 {
		labelY = det.Y1 + 25
	}

	// Draw confidence label using gg library for better text rendering
	ctx := gg.NewContextForRGBA(img)

	// Try to load font for confidence labels
	fontSize := 16.0
	fontPath := "./assets/fonts/msyh.ttc"

	if err := ctx.LoadFontFace(fontPath, fontSize); err != nil {
		// Fall back to manual text drawing if font loading fails
		drawManualConfidenceText(img, labelX, labelY, confidenceText)
		return
	}

	// Use gg library for high-quality text rendering
	// Draw background rectangle for better readability
	textWidth, textHeight := ctx.MeasureString(confidenceText)
	padding := 4.0

	// Draw semi-transparent background
	ctx.SetColor(color.RGBA{0, 0, 0, 180})
	ctx.DrawRectangle(float64(labelX)-padding, float64(labelY)-textHeight-padding, textWidth+2*padding, textHeight+2*padding)
	ctx.Fill()

	// Draw confidence text in bright yellow/green
	ctx.SetColor(color.RGBA{255, 255, 0, 255})
	ctx.DrawString(confidenceText, float64(labelX), float64(labelY))

}

// drawManualConfidenceText draws confidence text manually when font loading fails
func drawManualConfidenceText(img *image.RGBA, x, y int, text string) {
	charWidth := 8
	charHeight := 12
	padding := 2

	// Calculate background size
	textWidth := len(text) * charWidth

	// Draw background rectangle for better visibility
	bgColor := color.RGBA{0, 0, 0, 180}
	for dy := -padding; dy < charHeight+padding; dy++ {
		for dx := -padding; dx < textWidth+padding; dx++ {
			px := x + dx
			py := y + dy
			if px >= 0 && py >= 0 && px < img.Bounds().Max.X && py < img.Bounds().Max.Y {
				img.Set(px, py, bgColor)
			}
		}
	}

	// Draw text in bright yellow
	textColor := color.RGBA{255, 255, 0, 255}
	for i := range text {
		charX := x + i*charWidth
		if charX >= img.Bounds().Max.X {
			break
		}

		// Draw character as small rectangles
		for dy := 2; dy < charHeight-2; dy++ {
			for dx := 1; dx < charWidth-1; dx++ {
				px := charX + dx
				py := y + dy
				if px >= 0 && py >= 0 && px < img.Bounds().Max.X && py < img.Bounds().Max.Y {
					img.Set(px, py, textColor)
				}
			}
		}
	}
}

// TODO: support differences between classes.
// getClassColor returns a color for a detection class
func getClassColor(_ string) color.RGBA {

	// Default color for unknown classes
	return color.RGBA{0, 255, 0, 255}
}
