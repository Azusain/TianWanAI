cmake_minimum_required(VERSION 3.10)
project(stream_detector)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)

# Find FFmpeg libraries (only required ones)
pkg_check_modules(LIBAV REQUIRED IMPORTED_TARGET
    libavformat
    libavcodec
    libavutil
)

# Create executable
add_executable(stream_detector main.cpp)

# Link FFmpeg libraries
target_link_libraries(stream_detector PkgConfig::LIBAV)

# Set output directory
set_target_properties(stream_detector PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# For Windows, you might need to set additional properties
if(WIN32)
    # Add any Windows-specific configurations here if needed
endif()

# For Linux/WSL
if(UNIX)
    # Additional libraries that might be needed on Linux
    target_link_libraries(stream_detector pthread dl m)
endif()
