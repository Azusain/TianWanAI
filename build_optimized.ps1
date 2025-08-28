# Build optimized tianwan Docker image
# This script builds a significantly smaller image compared to the original

$IMAGE_NAME = "azusaing/tianwan"
$TAG_OPTIMIZED = "1.0.0-optimized" 
$DOCKERFILE = "Dockerfile_CUDA_12_6_3_optimized"

Write-Host "🚀 Building optimized tianwan Docker image..." -ForegroundColor Green
Write-Host "📦 Image: $IMAGE_NAME`:$TAG_OPTIMIZED" -ForegroundColor Cyan
Write-Host "📄 Dockerfile: $DOCKERFILE" -ForegroundColor Cyan
Write-Host ""

# Build the optimized image
Write-Host "⏳ Starting Docker build process..." -ForegroundColor Yellow
$buildResult = docker build -f $DOCKERFILE -t "$IMAGE_NAME`:$TAG_OPTIMIZED" .

# Check if build was successful
if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ Build completed successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "📊 Image size comparison:" -ForegroundColor Blue
    docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}" | Select-String "tianwan"
    Write-Host ""
    Write-Host "🏃 To run the optimized container:" -ForegroundColor Yellow
    Write-Host "docker run --gpus all -p 5000:5000 $IMAGE_NAME`:$TAG_OPTIMIZED" -ForegroundColor White
    Write-Host ""
    Write-Host "🔍 To inspect the container:" -ForegroundColor Yellow
    Write-Host "docker run --gpus all -it $IMAGE_NAME`:$TAG_OPTIMIZED bash" -ForegroundColor White
} else {
    Write-Host ""
    Write-Host "❌ Build failed! Please check the error messages above." -ForegroundColor Red
    exit 1
}
