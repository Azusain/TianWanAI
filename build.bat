@echo off
REM build script for unified dataset tool
REM this script builds the executable using pyinstaller

echo building unified dataset tool...
echo.

REM ensure we're in the project root directory
cd /d "%~dp0"

REM activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo warning: virtual environment not found, using system python
)

REM check if pyinstaller is available
python -m pyinstaller --version >nul 2>&1
if errorlevel 1 (
    echo installing pyinstaller...
    python -m pip install pyinstaller
    if errorlevel 1 (
        echo failed to install pyinstaller
        pause
        exit /b 1
    )
)

REM clean previous build
if exist "build" (
    echo cleaning previous build...
    rmdir /s /q build
)
if exist "dist" (
    echo cleaning previous dist...
    rmdir /s /q dist
)

REM run pyinstaller
echo running pyinstaller...
python -m pyinstaller unified_dataset_tool.spec

if errorlevel 1 (
    echo build failed
    pause
    exit /b 1
) else (
    echo.
    echo build completed successfully!
    echo executable is in dist\unified-dataset-tool.exe
    echo.
)

pause
