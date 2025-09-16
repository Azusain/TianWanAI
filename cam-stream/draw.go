package main

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"time"

	"github.com/fogleman/gg"
)

// DrawDetections draws detection boxes on the image
func DrawDetections(imageData []byte, detections []Detection, cameraName string, saveConfidenceLabel bool) ([]byte, error) {
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
		boxColor := getClassColor(det.Class)
		drawThickRectangle(rgbaImg, det.X1, det.Y1, det.X2, det.Y2, boxColor, 3)
		if !saveConfidenceLabel {
			continue
		}
		drawConfidenceLabel(rgbaImg, det)
	}

	// Add timestamp and camera name overlay with clear text
	addClearOverlay(rgbaImg, cameraName, len(detections))

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

// drawConfidenceLabel draws confidence score label near the detection box
func drawConfidenceLabel(img *image.RGBA, det Detection) {
	// Format confidence as percentage with one decimal place
	confidenceText := fmt.Sprintf("%.1f%%", det.Confidence*100)

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

// addClearOverlay adds timestamp and camera name with gg library for crisp text
func addClearOverlay(img *image.RGBA, cameraName string, detectionCount int) {
	bounds := img.Bounds()
	width := bounds.Max.X
	height := bounds.Max.Y

	// Create gg context from the RGBA image
	ctx := gg.NewContextForRGBA(img)

	// Set reasonable font size for better visibility
	fontSize := 32.0

	// Load Microsoft YaHei font from project fonts directory
	// TODO: hard coding.
	fontPath := "./assets/fonts/msyh.ttc"

	// fontLoaded := false
	if err := ctx.LoadFontFace(fontPath, fontSize); err != nil {
		// Font loading failed, will use fallback
		Error(fmt.Sprintf("failed to load font from %s", fontPath))
		return
	}
	// Right bottom corner - Camera name (yellow text with black outline)
	// Make sure we have a camera name to display
	if cameraName == "" {
		cameraName = "未命名摄像头" // Default Chinese name if empty
	}

	textWidth, _ := ctx.MeasureString(cameraName)

	cameraX := float64(width) - textWidth - 30
	cameraY := float64(height) - 30 // Simpler positioning

	drawTextWithOutline(ctx, cameraName, cameraX, cameraY, color.RGBA{255, 255, 255, 255}, color.RGBA{0, 0, 0, 255})
}

// drawTextWithOutline draws text with an outline for better visibility
func drawTextWithOutline(ctx *gg.Context, text string, x, y float64, textColor, outlineColor color.RGBA) {
	// Draw outline by drawing the text multiple times in offset positions
	offsets := []struct{ dx, dy float64 }{
		{-1, -1}, {-1, 0}, {-1, 1},
		{0, -1}, {0, 1},
		{1, -1}, {1, 0}, {1, 1},
	}

	// Draw outline
	ctx.SetColor(outlineColor)
	for _, offset := range offsets {
		ctx.DrawStringAnchored(text, x+offset.dx, y+offset.dy, 0, 0)
	}

	// Draw main text
	ctx.SetColor(textColor)
	ctx.DrawStringAnchored(text, x, y, 0, 0)
}

// addLargeTextOverlay draws large text overlay manually when font loading fails
func addLargeTextOverlay(img *image.RGBA, timestamp string, cameraName string, _ int) {
	bounds := img.Bounds()
	width := bounds.Max.X
	height := bounds.Max.Y

	// Use reasonable character sizes for manual drawing
	charWidth := 15  // Reasonable size characters
	charHeight := 25 // Reasonable height characters
	padding := 6     // Smaller padding

	// Left top corner - timestamp (white text)
	// TEMPORARILY NO NEEDED.
	// timestampX := 30
	// timestampY := 60
	// drawLargeText(img, timestampX, timestampY, timestamp, color.RGBA{255, 255, 255, 255}, charWidth, charHeight, padding)

	// Right bottom corner - camera name (yellow text)
	if cameraName == "" {
		cameraName = "未命名摄像头" // Default Chinese name if empty
	}

	cameraTextWidth := len(cameraName) * charWidth
	cameraX := width - cameraTextWidth - 40
	cameraY := height - charHeight - 40
	drawLargeText(img, cameraX, cameraY, cameraName, color.RGBA{255, 255, 0, 255}, charWidth, charHeight, padding)

}

// drawLargeText draws large text manually with specified character size
func drawLargeText(img *image.RGBA, x, y int, text string, textColor color.RGBA, charWidth, charHeight, padding int) {
	textWidth := len(text) * charWidth

	// Draw solid black background for contrast
	bgColor := color.RGBA{0, 0, 0, 220}
	for dy := -padding; dy < charHeight+padding; dy++ {
		for dx := -padding; dx < textWidth+padding; dx++ {
			px := x + dx
			py := y + dy
			if px >= 0 && py >= 0 && px < img.Bounds().Max.X && py < img.Bounds().Max.Y {
				img.Set(px, py, bgColor)
			}
		}
	}

	// Draw each character as a large filled rectangle
	for i := range text {
		charX := x + i*charWidth

		if charX >= img.Bounds().Max.X {
			break
		}

		// Draw character as a solid block with some internal structure
		for dy := 3; dy < charHeight-3; dy++ {
			for dx := 2; dx < charWidth-2; dx++ {
				px := charX + dx
				py := y + dy

				if px >= 0 && py >= 0 && px < img.Bounds().Max.X && py < img.Bounds().Max.Y {
					// Draw solid pixels for maximum visibility
					img.Set(px, py, textColor)
				}
			}
		}
	}
}
