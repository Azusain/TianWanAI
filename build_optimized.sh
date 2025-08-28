#!/bin/bash

# Build optimized tianwan Docker image
# This script builds a significantly smaller image compared to the original

IMAGE_NAME="azusaing/tianwan"
TAG_OPTIMIZED="1.0.0-optimized"
DOCKERFILE="Dockerfile_CUDA_12_6_3_optimized"

echo "🚀 Building optimized tianwan Docker image..."
echo "📦 Image: ${IMAGE_NAME}:${TAG_OPTIMIZED}"
echo "📄 Dockerfile: ${DOCKERFILE}"
echo ""

# Build the optimized image
echo "⏳ Starting Docker build process..."
docker build -f ${DOCKERFILE} -t ${IMAGE_NAME}:${TAG_OPTIMIZED} .

# Check if build was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Build completed successfully!"
    echo ""
    echo "📊 Image size comparison:"
    docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}" | grep "tianwan"
    echo ""
    echo "🏃 To run the optimized container:"
    echo "docker run --gpus all -p 5000:5000 ${IMAGE_NAME}:${TAG_OPTIMIZED}"
    echo ""
    echo "🔍 To inspect the container:"
    echo "docker run --gpus all -it ${IMAGE_NAME}:${TAG_OPTIMIZED} bash"
else
    echo ""
    echo "❌ Build failed! Please check the error messages above."
    exit 1
fi
