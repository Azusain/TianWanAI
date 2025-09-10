### TianWan

Computer vision inference service supporting multiple detection models.

### Supported Models

- **GESTURE** = "gesture" - Gesture detection
- **PONDING** = "ponding" - Ponding/water accumulation detection  
- **SMOKE** = "smoke" - Smoke detection
- **TSHIRT** = "tshirt" - T-shirt detection
- **MOUSE** = "mouse" - Mouse detection
- **FALL** = "fall" - Fall detection

### Docker Deployment

```bash
# Build with git operations and dependencies
docker build -f Dockerfile_Actions -t tianwan:latest .

# GPU deployment
docker run -d -p 8901:8080 --gpus '"device=1"' --cpus=16 \
  tianwan:latest

# CPU testing
docker run -d --rm -p 8901:8080 tianwan:latest
```

### Tested Interface

- gesture
- mouse
- ponding
- smoke
- tshirt
- fall

### Environment Variables

- `PERSON_CONFIDENCE_THRESHOLD`: Person classification confidence threshold for fall detection (default: 0.5)
  - Controls the minimum confidence required to classify a detection as a person
  - Higher values reduce false positives but may miss some valid detections
  - Range: 0.0 - 1.0

### Notes

```bash
  # Dockerfile_Cuda_11_1 is not applicable to this repo, 
  # but it can be used for other projects.
  
  # Fall detection now includes person verification to reduce false positives
  # Set PERSON_CONFIDENCE_THRESHOLD environment variable to adjust sensitivity
  # Example: export PERSON_CONFIDENCE_THRESHOLD=0.7
```
