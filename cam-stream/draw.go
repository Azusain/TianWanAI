package main

import (
	"fmt"
	"image"
	"image/color"
	"time"

	"github.com/fogleman/gg"
)

// ImageRenderer image renderer
type ImageRenderer struct {
	debugMode bool
}

// NewImageRenderer creates image renderer
func NewImageRenderer(debugMode bool) *ImageRenderer {
	return &ImageRenderer{
		debugMode: debugMode,
	}
}

// RenderDetections renders detection results on image
func (r *ImageRenderer) RenderDetections(img image.Image, detections []InferenceDetectionResult, cameraName string, timestamp time.Time) (image.Image, error) {
	bounds := img.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()

	// create drawing context
	dc := gg.NewContext(width, height)
	dc.DrawImage(img, 0, 0)

	// set font size (adjusted based on image size)
	fontSize := float64(width) / 50.0
	if fontSize < 12 {
		fontSize = 12
	} else if fontSize > 24 {
		fontSize = 24
	}

	// draw timestamp (top-left)
	r.drawTimestamp(dc, timestamp, fontSize)

	// draw camera name (bottom-right)
	r.drawCameraName(dc, cameraName, width, height, fontSize)

	// draw detection boxes
	r.drawDetectionBoxes(dc, detections, width, height, fontSize)

	return dc.Image(), nil
}

// drawTimestamp draws timestamp
func (r *ImageRenderer) drawTimestamp(dc *gg.Context, timestamp time.Time, fontSize float64) {
	timeStr := timestamp.Format("2006-01-02 15:04:05")
	
	// set font
	dc.LoadFontFace("arial", fontSize) // system font, will use default if not found
	
	// measure text size
	textWidth, textHeight := dc.MeasureString(timeStr)
	
	// set background
	padding := 8.0
	bgX := 10.0
	bgY := 10.0
	bgWidth := textWidth + padding*2
	bgHeight := textHeight + padding*2
	
	// draw semi-transparent background
	dc.SetColor(color.RGBA{0, 0, 0, 180})
	dc.DrawRoundedRectangle(bgX, bgY, bgWidth, bgHeight, 5)
	dc.Fill()
	
	// draw text
	dc.SetColor(color.RGBA{255, 255, 255, 255})
	dc.DrawString(timeStr, bgX+padding, bgY+textHeight+padding-5)
}

// drawCameraName draws camera name
func (r *ImageRenderer) drawCameraName(dc *gg.Context, cameraName string, width, height int, fontSize float64) {
	// set font
	dc.LoadFontFace("arial", fontSize)
	
	// measure text size
	textWidth, textHeight := dc.MeasureString(cameraName)
	
	// calculate position (bottom-right)
	padding := 8.0
	bgX := float64(width) - textWidth - padding*2 - 10
	bgY := float64(height) - textHeight - padding*2 - 10
	bgWidth := textWidth + padding*2
	bgHeight := textHeight + padding*2
	
	// draw semi-transparent background
	dc.SetColor(color.RGBA{0, 0, 0, 180})
	dc.DrawRoundedRectangle(bgX, bgY, bgWidth, bgHeight, 5)
	dc.Fill()
	
	// draw text
	dc.SetColor(color.RGBA{255, 255, 255, 255})
	dc.DrawString(cameraName, bgX+padding, bgY+textHeight+padding-5)
}

// drawDetectionBoxes draws detection boxes
func (r *ImageRenderer) drawDetectionBoxes(dc *gg.Context, detections []InferenceDetectionResult, width, height int, fontSize float64) {
	colors := []color.RGBA{
		{0, 255, 0, 255},   // green
		{255, 0, 0, 255},   // red
		{0, 0, 255, 255},   // blue
		{255, 255, 0, 255}, // yellow
		{255, 0, 255, 255}, // magenta
		{0, 255, 255, 255}, // cyan
	}

	for idx, detection := range detections {
		// select color
		color := colors[idx%len(colors)]
		
		// calculate pixel coordinates
		loc := detection.Location
		left := int(loc.Left * float64(width))
		top := int(loc.Top * float64(height))
		boxWidth := int(loc.Width * float64(width))
		boxHeight := int(loc.Height * float64(height))
		
		// ensure coordinates are within image bounds
		left = clamp(left, 0, width-1)
		top = clamp(top, 0, height-1)
		right := clamp(left+boxWidth, left+1, width)
		bottom := clamp(top+boxHeight, top+1, height)
		
		// draw detection box
		dc.SetColor(color)
		dc.SetLineWidth(3)
		dc.DrawRectangle(float64(left), float64(top), float64(right-left), float64(bottom-top))
		dc.Stroke()
		
		// draw confidence and class info in Debug mode
		if r.debugMode {
			labelText := fmt.Sprintf("Score: %.3f", detection.Score)
			if detection.Class != "" {
				labelText = fmt.Sprintf("%s %.3f", detection.Class, detection.Score)
			}
			
			r.drawDetectionLabel(dc, labelText, float64(left), float64(top), fontSize*0.8, color)
		}
	}
}

// drawDetectionLabel draws detection label
func (r *ImageRenderer) drawDetectionLabel(dc *gg.Context, text string, x, y, fontSize float64, bgColor color.RGBA) {
	// set font
	dc.LoadFontFace("arial", fontSize)
	
	// measure text size
	textWidth, textHeight := dc.MeasureString(text)
	
	// calculate label position
	padding := 4.0
	labelX := x
	labelY := y - textHeight - padding*2
	
	// if label exceeds image top, place inside box
	if labelY < 0 {
		labelY = y + padding
	}
	
	// draw label background
	dc.SetColor(bgColor)
	dc.DrawRoundedRectangle(labelX, labelY, textWidth+padding*2, textHeight+padding*2, 3)
	dc.Fill()
	
	// draw text
	dc.SetColor(color.RGBA{255, 255, 255, 255})
	dc.DrawString(text, labelX+padding, labelY+textHeight+padding-2)
}

// clamp restricts value within specified range
func clamp(value, min, max int) int {
	if value < min {
		return min
	}
	if value > max {
		return max
	}
	return value
}
