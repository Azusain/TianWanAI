# Build tianwan Docker image (matches GitHub workflow behavior)
# Uses the same Dockerfile and tag as the GitHub Actions workflow

$IMAGE_NAME = "tianwan"
$TAG = "latest" 
$DOCKERFILE = "Dockerfile_Actions"

Write-Host "🚀 Building tianwan Docker image (GitHub workflow compatible)..." -ForegroundColor Green
Write-Host "📦 Image: $IMAGE_NAME`:$TAG" -ForegroundColor Cyan
Write-Host "📄 Dockerfile: $DOCKERFILE" -ForegroundColor Cyan
Write-Host ""

# Build the image (same as GitHub workflow)
Write-Host "⏳ Starting Docker build process..." -ForegroundColor Yellow
$buildResult = docker build -f $DOCKERFILE -t "$IMAGE_NAME`:$TAG" .

# Check if build was successful
if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ Build completed successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "📊 Image size comparison:" -ForegroundColor Blue
    docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}" | Select-String "tianwan"
    Write-Host ""
    Write-Host "🏃 To run the container:" -ForegroundColor Yellow
    Write-Host "docker run --gpus all -p 5000:5000 $IMAGE_NAME`:$TAG" -ForegroundColor White
    Write-Host ""
    Write-Host "🔍 To inspect the container:" -ForegroundColor Yellow
    Write-Host "docker run --gpus all -it $IMAGE_NAME`:$TAG bash" -ForegroundColor White
} else {
    Write-Host ""
    Write-Host "❌ Build failed! Please check the error messages above." -ForegroundColor Red
    exit 1
}
