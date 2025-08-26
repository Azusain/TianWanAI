@echo off
chcp 65001 >nul
echo ====================================
echo        Building Cam-Stream
echo ====================================
echo.

REM Check if Go is available
go version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Go not found
    echo Please install Go: https://golang.org/dl/
    pause
    exit /b 1
)

echo Tidying dependencies...
go mod tidy
if errorlevel 1 (
    echo Failed to tidy dependencies
    pause
    exit /b 1
)

echo Building cam-stream...
echo.

REM Use go build . to build the entire package
go build -ldflags="-s -w" -o cam-stream.exe .
if errorlevel 1 (
    echo Build failed!
    echo.
    echo Common issues:
    echo 1. Make sure all .go files are in the same package
    echo 2. Verify all imports are available
    pause
    exit /b 1
)

echo Build successful!
echo Created: cam-stream.exe
echo.
echo How to run:
echo 1. Start RTSP test server: simple-rtsp-test.bat
echo 2. Run application: cam-stream.exe or start.bat
echo 3. Access web interface: http://localhost:3000
echo.
pause
