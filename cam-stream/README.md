# Multi-Camera Stream Platform

A simple multi-camera streaming platform with web interface and API.

## Quick Start

```powershell
# Run the application
.\run.ps1
# or
go run .
```

## Access Points

- **Web Interface**: http://localhost:3001
- **Camera Management**: http://localhost:3001/cameras  
- **API Debug**: http://localhost:3001/api/debug

## Key Features

- ✅ CORS-enabled REST API
- 🎥 Camera management interface  
- 🌐 Web-based control panel
- 📡 RTSP streaming support (via MediaMTX)

## Project Structure

```
cam-stream/
├── src/
│   ├── handlers/          # API handlers
│   └── server/            # Web server
├── mediamtx/              # RTSP server
├── main.go               # Application entry point
├── camera_management.html # Web interface
├── config.json           # Configuration
├── go.mod/go.sum         # Go modules
└── run.ps1               # Run script
```

## Environment Variables

- `FRAME_RATE`: Frame rate limit for video processing (default: 25)
  - Controls the maximum frames per second for camera streams
  - Valid range: 1-120 FPS
  - Higher values provide smoother video but consume more resources
  - Example: `export FRAME_RATE=30`

## Development

The project uses Go modules with clean package structure. All API endpoints include proper CORS headers for frontend integration.
