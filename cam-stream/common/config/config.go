package config

import "time"

const (
	OutputDir                   = "output"
	DebugDir                    = "debug"
	DefaultWebPort         uint = 8080
	TemplatesDir                = "templates"
	DataFile                    = "_data/cameras.json"
	DataDir                     = "_data"
	RetryTimeSecond        uint = 3
	DefaultGetFrameTimeout uint = 3
)

var (
	GlobalFrameRate     int
	GlobalFrameInterval time.Duration
	GlobalDebugMode     bool
)

// Readonly so we dont need to protect it with lock.

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
func GetClassIndexFromModelType(modelType string) int {
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
