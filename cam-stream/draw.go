package main

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"log"
	"time"

	"github.com/fogleman/gg"
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

	// Add timestamp and camera name overlay with clear text
	addClearOverlay(rgbaImg, cameraName, len(detections))

	// Encode back to JPEG
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, rgbaImg, &jpeg.Options{Quality: 90}); err != nil {
		return nil, fmt.Errorf("failed to encode JPEG: %v", err)
	}

	return buf.Bytes(), nil
}

// drawDetectionBox draws a single detection box without label
func drawDetectionBox(img *image.RGBA, det Detection) {
	// Debug: log detection details
	fmt.Printf("Drawing detection: Class=%s, Confidence=%.2f, Coords=(%d,%d,%d,%d)\n", det.Class, det.Confidence, det.X1, det.Y1, det.X2, det.Y2)

	// Define colors for different classes
	boxColor := getClassColor(det.Class)

	// Draw box border (3 pixels wide for visibility)
	drawThickRectangle(img, det.X1, det.Y1, det.X2, det.Y2, boxColor, 3)

	// No label drawing - just the detection box
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

// addOverlay adds timestamp and camera name overlay
func addOverlay(img *image.RGBA, cameraName string, detectionCount int) {
	bounds := img.Bounds()
	width := bounds.Max.X
	height := bounds.Max.Y

	// Get current time
	now := time.Now()
	timestamp := now.Format("2006-01-02 15:04:05")
	weekday := getChineseWeekday(now.Weekday())
	fullTimestamp := fmt.Sprintf("%s %s", timestamp, weekday)

	// Left top corner - Date and time
	drawTextWithBackground(img, 10, 25, fullTimestamp, color.RGBA{255, 255, 255, 255})

	// Right bottom corner - Camera name
	cameraTextWidth := len(cameraName) * 8 // Approximate width
	cameraX := width - cameraTextWidth - 10
	cameraY := height - 15
	drawTextWithBackground(img, cameraX, cameraY, cameraName, color.RGBA{255, 255, 0, 255})
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
	// Simple text drawing - just draw a small rectangle to represent text
	// This avoids the ugly line pattern from before

	textWidth := len(text) * 8
	textHeight := 12

	// Ensure text doesn't go out of bounds
	if x < 0 {
		x = 0
	}
	if y < 0 {
		y = 0
	}
	if x+textWidth > img.Bounds().Max.X {
		textWidth = img.Bounds().Max.X - x
	}

	// Draw a simple filled rectangle to represent text
	for dy := 0; dy < textHeight && y+dy < img.Bounds().Max.Y; dy++ {
		for dx := 0; dx < textWidth && x+dx < img.Bounds().Max.X; dx++ {
			if x+dx >= 0 && y+dy >= 0 {
				// Make it semi-transparent by only drawing every few pixels
				if dx%3 == 0 && dy%2 == 0 {
					img.Set(x+dx, y+dy, col)
				}
			}
		}
	}
}

// drawTextWithBackground draws text with a semi-transparent background
func drawTextWithBackground(img *image.RGBA, x, y int, text string, textColor color.RGBA) {
	// Calculate text dimensions (approximate)
	textWidth := len(text) * 7 // 7 pixels per character
	textHeight := 14           // Font height

	// Draw semi-transparent background
	bgColor := color.RGBA{0, 0, 0, 180} // Semi-transparent black
	padding := 4
	for dy := -padding; dy < textHeight+padding; dy++ {
		for dx := -padding; dx < textWidth+padding; dx++ {
			px := x + dx
			py := y + dy
			if px >= 0 && py >= 0 && px < img.Bounds().Max.X && py < img.Bounds().Max.Y {
				img.Set(px, py, bgColor)
			}
		}
	}

	// Draw text using dense pixel pattern for better readability
	drawDenseText(img, x, y, text, textColor)
}

// drawDenseText draws text with a denser pixel pattern for better visibility
func drawDenseText(img *image.RGBA, x, y int, text string, col color.RGBA) {
	// Use a denser pattern for better text visibility
	charWidth := 7
	charHeight := 12

	for i, char := range text {
		charX := x + i*charWidth

		// Skip characters that are out of bounds
		if charX >= img.Bounds().Max.X {
			break
		}

		// Draw character as a filled rectangle with better pattern
		for dy := 0; dy < charHeight; dy++ {
			for dx := 0; dx < charWidth-1; dx++ { // -1 for spacing
				px := charX + dx
				py := y + dy

				if px >= 0 && py >= 0 && px < img.Bounds().Max.X && py < img.Bounds().Max.Y {
					// Create a readable pattern based on character and position
					if shouldDrawPixel(char, dx, dy, charWidth, charHeight) {
						img.Set(px, py, col)
					}
				}
			}
		}
	}
}

// shouldDrawPixel determines if a pixel should be drawn for a character
func shouldDrawPixel(char rune, dx, dy, charWidth, charHeight int) bool {
	// Create basic patterns for better readability
	// This creates a more solid appearance than the previous sparse pattern

	// Draw solid borders for most characters
	if dy == 0 || dy == charHeight-1 || dx == 0 || dx == charWidth-2 {
		return true
	}

	// Fill some interior based on character position
	if (dx+dy)%2 == 0 {
		return true
	}

	return false
}

// drawBetterText draws clearer text with solid background
func drawBetterText(img *image.RGBA, x, y int, text string, textColor color.RGBA) {
	charWidth := 6
	charHeight := 10
	padding := 3

	textWidth := len(text) * charWidth
	textHeight := charHeight

	// Draw solid background
	bgColor := color.RGBA{0, 0, 0, 200} // More opaque black
	for dy := -padding; dy < textHeight+padding; dy++ {
		for dx := -padding; dx < textWidth+padding; dx++ {
			px := x + dx
			py := y + dy
			if px >= 0 && py >= 0 && px < img.Bounds().Max.X && py < img.Bounds().Max.Y {
				img.Set(px, py, bgColor)
			}
		}
	}

	// Draw text with solid pattern for better visibility
	for i := range text {
		charX := x + i*charWidth

		if charX >= img.Bounds().Max.X {
			break
		}

		// Draw each character as a solid rectangle
		for dy := 1; dy < charHeight-1; dy++ {
			for dx := 1; dx < charWidth-1; dx++ {
				px := charX + dx
				py := y + dy

				if px >= 0 && py >= 0 && px < img.Bounds().Max.X && py < img.Bounds().Max.Y {
					// Draw solid pixels for better readability
					img.Set(px, py, textColor)
				}
			}
		}
	}
}

// addClearOverlay adds timestamp and camera name with gg library for crisp text
func addClearOverlay(img *image.RGBA, cameraName string, detectionCount int) {
	bounds := img.Bounds()
	width := bounds.Max.X
	height := bounds.Max.Y

	// Debug logging
	fmt.Printf("Adding overlay - Image size: %dx%d, Camera name: '%s'\n", width, height, cameraName)

	// Get current time
	now := time.Now()
	timestamp := now.Format("2006-01-02 15:04:05")
	weekday := getChineseWeekday(now.Weekday())
	fullTimestamp := fmt.Sprintf("%s %s", timestamp, weekday)

	// Create gg context from the RGBA image
	ctx := gg.NewContextForRGBA(img)

	// Set reasonable font size for better visibility
	fontSize := 32.0  // Reduced from 96 to 32
	// Try to load a system font, fallback to basic drawing if it fails
	if err := ctx.LoadFontFace("/Windows/Fonts/arial.ttf", fontSize); err != nil {
		// If system font fails, try alternative methods
		if err := ctx.LoadFontFace("C:/Windows/Fonts/arial.ttf", fontSize); err != nil {
			// If still fails, we'll draw large text manually
			log.Printf("Warning: Could not load system font, using manual drawing: %v", err)
			// Draw timestamp and camera name with manual large text
			addLargeTextOverlay(img, fullTimestamp, cameraName, detectionCount)
			return
		}
	}

	// Left top corner - Date and time (white text with black outline)
	drawTextWithOutline(ctx, fullTimestamp, 25, 65, color.RGBA{255, 255, 255, 255}, color.RGBA{0, 0, 0, 255})

	// Right bottom corner - Camera name (yellow text with black outline)
	// Make sure we have a camera name to display
	if cameraName == "" {
		cameraName = "Camera-01" // Default name if empty
	}

	// Force display a camera name for debugging
	cameraName = "测试摄像头" // Use Chinese text for testing

	textWidth, textHeight := ctx.MeasureString(cameraName)
	fmt.Printf("Camera name '%s' - Text size: %.1fx%.1f\n", cameraName, textWidth, textHeight)

	cameraX := float64(width) - textWidth - 30
	cameraY := float64(height) - 30 // Simpler positioning
	fmt.Printf("Drawing camera name at position: (%.1f, %.1f)\n", cameraX, cameraY)

	drawTextWithOutline(ctx, cameraName, cameraX, cameraY, color.RGBA{255, 255, 0, 255}, color.RGBA{0, 0, 0, 255})
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
func addLargeTextOverlay(img *image.RGBA, timestamp string, cameraName string, detectionCount int) {
	bounds := img.Bounds()
	width := bounds.Max.X
	height := bounds.Max.Y

	// Use reasonable character sizes for manual drawing
	charWidth := 15  // Reasonable size characters
	charHeight := 25 // Reasonable height characters
	padding := 6     // Smaller padding

	// Left top corner - timestamp (white text)
	timestampX := 30
	timestampY := 60
	drawLargeText(img, timestampX, timestampY, timestamp, color.RGBA{255, 255, 255, 255}, charWidth, charHeight, padding)

	// Right bottom corner - camera name (yellow text)
	if cameraName == "" {
		cameraName = "测试摄像头"
	} else {
		cameraName = "测试摄像头" // Force for testing
	}

	cameraTextWidth := len(cameraName) * charWidth
	cameraX := width - cameraTextWidth - 40
	cameraY := height - charHeight - 40
	drawLargeText(img, cameraX, cameraY, cameraName, color.RGBA{255, 255, 0, 255}, charWidth, charHeight, padding)

	fmt.Printf("Drew large manual text - timestamp at (%d,%d), camera at (%d,%d)\n", timestampX, timestampY, cameraX, cameraY)
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

// containsChinese checks if string contains Chinese characters
func containsChinese(s string) bool {
	for _, r := range s {
		if r >= 0x4E00 && r <= 0x9FFF {
			return true
		}
	}
	return false
}
