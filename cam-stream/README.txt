Cam-Stream with MediaMTX
========================

Setup (First time):
1. git lfs pull                   (Download MediaMTX binary)
2. Run: start-rtsp.bat            (Start server + push video)
3. Run: cam-stream.exe            (Start main application)
4. Visit: http://localhost:3000   (Web interface)

Files:
- start-rtsp.bat         - Start RTSP server + push video (combined)
- cam-stream.exe         - Main application (Git LFS)
- config.json            - Configuration (RTSP: rtsp://localhost:8554/test)
- mediamtx/              - MediaMTX server binaries (Git LFS)

To use custom video:
Run start-rtsp.bat and enter your video file path when prompted.
Example: C:\Videos\test.mp4

Note: All .exe files are stored with Git LFS to keep repository size small.
