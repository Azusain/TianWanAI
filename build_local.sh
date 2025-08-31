#!/bin/bash
# Build TianWan Fall Detection Docker image
IMAGE_NAME="tianwan-fall-detection:cuda12.6.3"

# Check Docker
if ! docker info > /dev/null 2>&1; then
    echo "Docker not running"
    exit 1
fi

# Build
echo "Building ${IMAGE_NAME}..."
docker build -f Dockerfile_CUDA_12_6_3 -t "${IMAGE_NAME}" .

if [ $? -eq 0 ]; then
    echo "Build complete: ${IMAGE_NAME}"
    echo "Run: docker run -d -p 8080:8080 --gpus all -e MODEL=fall -e NPROC=1 ${IMAGE_NAME}"
else
    echo "Build failed"
    exit 1
fi
