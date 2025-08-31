#!/bin/bash

# Build tianwan Docker image (matches GitHub workflow behavior)
# Uses the same Dockerfile and tag as the GitHub Actions workflow

IMAGE_NAME="tianwan"
TAG="latest"
DOCKERFILE="Dockerfile_Actions"

echo "🚀 Building tianwan Docker image (GitHub workflow compatible)..."
echo "📦 Image: ${IMAGE_NAME}:${TAG}"
echo "📄 Dockerfile: ${DOCKERFILE}"
echo ""

# Build the image (same as GitHub workflow)
echo "⏳ Starting Docker build process..."
docker build -f ${DOCKERFILE} -t ${IMAGE_NAME}:${TAG} .

# Check if build was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Build completed successfully!"
    echo ""
    echo "📊 Image size comparison:"
    docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}" | grep "tianwan"
    echo ""
    echo "🏃 To run the container:"
    echo "docker run --gpus all -p 5000:5000 ${IMAGE_NAME}:${TAG}"
    echo ""
    echo "🔍 To inspect the container:"
    echo "docker run --gpus all -it ${IMAGE_NAME}:${TAG} bash"
else
    echo ""
    echo "❌ Build failed! Please check the error messages above."
    exit 1
fi
