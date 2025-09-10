package main

import (
	"fmt"
	"image"
	"gocv.io/x/gocv"
)

func main() {
	fmt.Printf("GoCV version: %s\n", gocv.Version())
	fmt.Printf("OpenCV lib version: %s\n", gocv.OpenCVVersion())
	
	// Create a simple Mat
	mat := gocv.NewMat()
	defer mat.Close()
	
	// Check if Mat is empty
	fmt.Printf("Empty mat: %v\n", mat.Empty())
	
	// Try creating an image
	img := gocv.NewMatWithSize(300, 400, gocv.MatTypeCV8UC3)
	defer img.Close()
	
	// Fill it with a color
	blue := gocv.NewScalar(255, 0, 0, 0)
	gocv.Rectangle(&img, image.Rect(50, 50, 200, 200), blue, 2)
	
	fmt.Println("OpenCV Mat operations completed successfully")
}
