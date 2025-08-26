@echo off
echo ====================================
echo      RTSP Server + Video Stream
echo ====================================
echo.

REM 检查 MediaMTX 是否存在
if not exist "mediamtx\mediamtx.exe" (
    echo ERROR: MediaMTX not found!
    echo Please run: git lfs pull
    echo Or manually download MediaMTX to mediamtx\ folder
    pause
    exit /b 1
)

REM 获取视频文件路径
set /p video_path=Enter video file path (or press Enter for test pattern): 

echo.
echo Starting MediaMTX server...
start /min "MediaMTX Server" cmd /c "cd mediamtx && mediamtx.exe"

echo Waiting for server to start...
timeout /t 3 /nobreak >nul

echo.
if "%video_path%"=="" goto :test_pattern
if not exist "%video_path%" (
    echo File not found: %video_path%
    echo Using test pattern instead...
    goto :test_pattern
)

echo Pushing video: %video_path%
echo RTSP URL: rtsp://localhost:8554/test
echo.
echo Press Ctrl+C to stop streaming
ffmpeg -re -stream_loop -1 -i "%video_path%" -c:v libx264 -preset ultrafast -f rtsp rtsp://localhost:8554/test
goto :end

:test_pattern
echo Using test pattern
echo RTSP URL: rtsp://localhost:8554/test
echo.
echo Press Ctrl+C to stop streaming
ffmpeg -re -f lavfi -i testsrc2=size=640x480:rate=25 -c:v libx264 -preset ultrafast -f rtsp rtsp://localhost:8554/test

:end
echo.
echo Stopping MediaMTX server...
taskkill /f /im mediamtx.exe >nul 2>&1
echo Stream stopped
pause
