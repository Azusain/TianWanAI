#!/bin/bash

echo "Building stream detector..."

# Check if we're in the correct directory
if [ ! -f "main.cpp" ]; then
    echo "Error: main.cpp not found. Please run this script from the stream_detector directory."
    exit 1
fi

# Create build directory
mkdir -p build
cd build

# Configure with cmake
if ! cmake ..; then
    echo "Error: CMake configuration failed."
    echo "Please make sure FFmpeg development libraries are installed:"
    echo "sudo apt install libavformat-dev libavcodec-dev libavdevice-dev libavfilter-dev libavutil-dev libswscale-dev libswresample-dev"
    exit 1
fi

# Build
if ! make; then
    echo "Error: Build failed."
    exit 1
fi

# Check if the executable was created
if [ -f "stream_detector" ]; then
    echo "Build successful! Executable: $(pwd)/stream_detector"
    
    # Make it executable
    chmod +x stream_detector
    
    # Test with a dummy URL (this will fail but shows the executable works)
    echo "Testing executable..."
    ./stream_detector "test" 2>/dev/null
    if [ $? -eq 1 ]; then
        echo "Executable is working (exit code 1 is expected for invalid URL)"
    fi
else
    echo "Error: Executable not found after build."
    exit 1
fi

echo "Done!"
