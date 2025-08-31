# Build TianWan Fall Detection Docker image
$IMAGE_NAME = "tianwan-fall-detection:cuda12.6.3"

# Check Docker
docker info >$null 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker not running"
    exit 1
}

# Build
Write-Host "Building $IMAGE_NAME..."
docker build -f Dockerfile_CUDA_12_6_3 -t $IMAGE_NAME .

if ($LASTEXITCODE -eq 0) {
    Write-Host "Build complete: $IMAGE_NAME"
    Write-Host "Run: docker run -d -p 8080:8080 --gpus all -e MODEL=fall -e NPROC=1 $IMAGE_NAME"
} else {
    Write-Host "Build failed"
    exit 1
}
