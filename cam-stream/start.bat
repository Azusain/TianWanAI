@echo off
echo ====================================
echo    Cam-Stream AI Camera System
echo ====================================
echo.

REM Check if executable exists
if not exist "cam-stream.exe" (
    echo Building cam-stream.exe...
    go build -o cam-stream.exe .
    if errorlevel 1 (
        echo Failed to build cam-stream.exe
        pause
        exit /b 1
    )
    echo Build completed successfully.
    echo.
)

REM Create output directory
if not exist "output" mkdir output

REM Display configuration info
if exist "config.json" (
    echo Using existing config.json
) else (
    echo Creating default config.json...
)

echo.
echo Starting Camera Stream Application...
echo Press Ctrl+C to stop the application
echo.

REM Start application
cam-stream.exe

echo.
echo Application stopped.
pause
