# Docker Build Configurations

This repository contains two Dockerfile versions: one for regular builds and one optimized for GitHub Actions runners to prevent disk space issues.

## Problem

The original error occurred during Docker layer export phase:
```
System.IO.IOException: No space left on device : '/home/runner/actions-runner/cached/_diag/Worker_20250831-124623-utc.log'
```

This happens because Docker creates many intermediate layers during builds, consuming significant disk space on GitHub Actions runners.

## Solutions

### 1. `Dockerfile_CUDA_12_6_3` - Regular Build
- **Purpose**: Standard development and production builds
- **Features**:
  - Multi-stage build with separate downloader and runtime stages
  - Full feature set for development workflow
  - Git LFS and submodule support

### 2. `Dockerfile_Actions` - GitHub Actions Optimized
- **Purpose**: Optimized for GitHub Actions runners with limited disk space
- **Key optimizations**:
  - Combines multiple RUN commands into single layers
  - Uses `--no-cache-dir` for pip installations
  - Immediate cleanup of package caches
  - Removes unnecessary files after installation

## Optimization Techniques Used

### Layer Reduction
- Combine multiple `RUN` commands using `&&` and `\` 
- Group related operations in single layers
- Avoid creating unnecessary intermediate layers

### Cache Management
- Use `--no-cache-dir` for pip installations
- Clean package manager caches immediately after installation
- Remove build artifacts in the same layer they're created

### Space Cleanup
```dockerfile
# Example of aggressive cleanup
RUN apt-get install -y packages && \
    # ... operations ... && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    find . -name "__pycache__" -type d -exec rm -rf {} + && \
    pip cache purge
```

### File Management
- Copy only essential files first to leverage Docker cache
- Remove source files after compilation when possible
- Clean up temporary files and caches

## GitHub Actions Integration

The included workflow `.github/workflows/build-optimized.yml` demonstrates:

### Disk Space Management
```yaml
- name: Free Disk Space (Ubuntu)
  uses: jlumbroso/free-disk-space@main
  with:
    tool-cache: true
    android: true
    dotnet: true
    haskell: true
    large-packages: true
    docker-images: true
    swap-storage: true
```

### Build Optimization
- Uses Docker Buildx for better space efficiency
- Implements GitHub Actions cache (`cache-from: type=gha`)
- Exports images to tar files to avoid layer storage
- Monitors disk space before and after builds

## Usage Recommendations

### For Local Development
Use `Dockerfile_CUDA_12_6_3` - the standard build with full features.

### For GitHub Actions CI/CD
Use `Dockerfile_Actions` - optimized for space-constrained runner environments.

### Build Commands
```bash
# Regular build
docker build -f Dockerfile_CUDA_12_6_3 -t tianwan:latest .

# Actions optimized build
docker build -f Dockerfile_Actions -t tianwan:actions .
```

## Monitoring

Each build includes disk space monitoring:
```bash
df -h  # Check available disk space
docker system df  # Check Docker disk usage
```

## Additional Tips

1. **Use .dockerignore**: Exclude unnecessary files from build context
2. **Regular cleanup**: Use `docker system prune -af` in workflows
3. **Cache strategy**: Leverage GitHub Actions cache for Docker layers
4. **Multi-stage considerations**: While powerful, multi-stage builds create more layers - use single-stage for Actions when possible

## Troubleshooting

If you still encounter space issues:

1. Check the workflow uses the disk space cleanup action
2. Verify you're using the optimized Dockerfiles
3. Consider using `outputs: type=oci` instead of `type=docker`
4. Monitor actual space usage with `df -h` steps in your workflow
