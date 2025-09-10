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

#### Option 1: Use Base Image + Volume Mount (Recommended)

Use the pre-built `azusaing/ultralytics` base image with your application code mounted:

```bash
# GPU deployment
docker run -d -p 8902:8080 --gpus '"device=1"' --cpus=16 \
  -v $(pwd):/root \
  -e NPROC=6 -e MODEL="gesture" \
  azusaing/ultralytics:latest bash run.bash

# CPU testing  
docker run -d --rm -p 8902:8080 \
  -v $(pwd):/root \
  -e NPROC=6 -e MODEL="gesture" \
  azusaing/ultralytics:latest bash run.bash
```

#### Option 2: Build Application Image

```bash
# Build application image
docker build -f Dockerfile_App -t tianwan-app:latest .

# Run application
docker run -d -p 8902:8080 --gpus '"device=1"' --cpus=16 \
  -e NPROC=6 -e MODEL="gesture" tianwan-app:latest
```

#### Option 3: Build Full Image (Legacy)

```bash
# Build with git operations and dependencies
docker build -f Dockerfile_Actions -t tianwan-full:latest .

# Run
docker run -d -p 8902:8080 --gpus '"device=1"' --cpus=16 \
  -e NPROC=6 -e MODEL="gesture" tianwan-full:latest
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