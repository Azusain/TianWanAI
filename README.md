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

#### Recommended: Volume Mount with Base Image

Use the pre-built `azusaing/ultralytics` base image and mount your source code:

```bash
# GPU deployment
docker run -d -p 8901:8080 --gpus '"device=1"' --cpus=16 \
  -v $(pwd):/root \
  azusaing/ultralytics:latest

# CPU testing  
docker run -d --rm -p 8901:8080 \
  -v $(pwd):/root \
  azusaing/ultralytics:latest
```

#### Legacy: Build Full Image

```bash
# Build with git operations and dependencies
docker build -f Dockerfile_Actions -t tianwan:latest .

# Run
docker run -d -p 8901:8080 --gpus '"device=1"' --cpus=16 \
  tianwan:latest
```

### Tested Interface

- gesture
- mouse
- ponding
- smoke
- tshirt
- fall

### Notes

```bash
  # Dockerfile_Cuda_11_1 is not applicable to this repo, 
  # but it can be used for other projects.
```