@echo off
echo ====================================
echo      RTSP Server + Video Stream
echo ====================================
echo.

REM 
if not exist "mediamtx.exe" (
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
REM 优化的编码参数 - 高质量模式
ffmpeg -re -stream_loop -1 -i "%video_path%" ^
    -c:v libx264 ^
    -preset medium ^
    -crf 18 ^
    -maxrate 4M ^
    -bufsize 8M ^
    -g 50 ^
    -sc_threshold 0 ^
    -f rtsp rtsp://localhost:8554/test
goto :end

:test_pattern
echo Using test pattern
echo RTSP URL: rtsp://localhost:8554/test
echo.
echo Press Ctrl+C to stop streaming
REM 测试模式 - 高质量参数
ffmpeg -re -f lavfi -i testsrc2=size=1280x720:rate=25 ^
    -c:v libx264 ^
    -preset medium ^
    -crf 18 ^
    -maxrate 4M ^
    -bufsize 8M ^
    -g 50 ^
    -sc_threshold 0 ^
    -f rtsp rtsp://localhost:8554/test

:end
echo.
echo Stopping MediaMTX server...
taskkill /f /im mediamtx.exe >nul 2>&1
echo Stream stopped
pause