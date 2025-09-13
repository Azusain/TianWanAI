# Unified Dataset Tool - Packaging Guide

This document describes how to package the Unified Dataset Tool into a standalone executable using PyInstaller.

## Requirements

- Python 3.9 or later
- Virtual environment (recommended)
- PyInstaller 6.0 or later
- All dependencies from requirements.txt

## Project Structure

```
tianwan/
├── tools/
│   └── unified_dataset_tool/
│       ├── main_window.py          # main GUI application
│       ├── dataset_manager.py      # dataset management logic
│       └── video_processor.py      # video processing logic
├── unified_dataset_tool.spec       # pyinstaller configuration
├── build.bat                       # automated build script
├── docs/                           # documentation
└── dist/                           # generated executables (after build)
```

## Build Process

### Method 1: Using the Automated Build Script (Recommended)

1. Open Command Prompt or PowerShell in the project root directory
2. Run the build script:
   ```batch
   build.bat
   ```

This script will:
- Activate the virtual environment if present
- Install PyInstaller if needed
- Clean previous builds
- Run PyInstaller with the spec file
- Generate the executable in the `dist` folder

### Method 2: Manual Build

1. **Activate Virtual Environment**
   ```batch
   # On Windows
   venv\Scripts\activate.bat
   
   # On PowerShell
   venv\Scripts\Activate.ps1
   ```

2. **Install PyInstaller**
   ```batch
   python -m pip install pyinstaller
   ```

3. **Clean Previous Builds** (optional)
   ```batch
   rmdir /s /q build
   rmdir /s /q dist
   ```

4. **Run PyInstaller**
   ```batch
   pyinstaller unified_dataset_tool.spec
   ```

## PyInstaller Configuration

The `unified_dataset_tool.spec` file contains the build configuration:

### Key Settings:
- **Entry Point**: `tools/unified_dataset_tool/main_window.py`
- **Console Mode**: `console=True` (shows console for debugging)
- **Single File**: All dependencies bundled into one executable
- **Hidden Imports**: Explicitly includes PyQt6 modules
- **UPX Compression**: Enabled to reduce file size
- **Icon**: Automatically uses `app.ico` if present

### Dependencies Included:
- PyQt6 (GUI framework)
- OpenCV (computer vision)
- Pillow (image processing)
- NumPy (numerical computing)
- Matplotlib (visualization)

## Output

After successful build, you will find:
- **Executable**: `dist/unified-dataset-tool.exe` (~89MB)
- **Build Files**: `build/` directory (can be deleted)

## Distribution

The generated executable is completely standalone and includes:
- Python runtime
- All Python packages
- Qt6 libraries
- Visual C++ runtime libraries

### System Requirements for Distribution:
- Windows 10 or later (64-bit)
- No additional installations required

## Troubleshooting

### Common Issues:

1. **Missing Module Errors**
   - Add missing modules to `hiddenimports` in the spec file
   - Example: `'module_name'` in hiddenimports list

2. **PyQt6 Import Errors**
   - Ensure all PyQt6 submodules are listed in hiddenimports
   - Check that PyQt6 is properly installed

3. **Large Executable Size**
   - Enable UPX compression: `upx=True`
   - Exclude unnecessary modules in `excludes`

4. **DLL Loading Issues**
   - Check that all required DLLs are included
   - Verify system PATH doesn't interfere

### Debug Mode:
To debug issues, modify the spec file:
```python
exe = EXE(
    # ... other parameters
    debug=True,    # Enable debug mode
    console=True,  # Keep console visible
)
```

## Performance Optimization

### Reducing Build Time:
1. Use `--noconfirm` flag to skip confirmations
2. Keep build cache by not deleting `build/` folder
3. Use incremental builds when possible

### Reducing Executable Size:
1. Enable UPX compression: `upx=True`
2. Exclude unused modules in `excludes` list
3. Use `--strip` flag for release builds

## Version Information

- **PyInstaller Version**: 6.16.0
- **Python Version**: 3.13.5
- **Target Platform**: Windows 11
- **Architecture**: 64-bit

## Development Notes

### Testing the Executable:
1. Test on clean Windows system without Python
2. Verify all features work correctly
3. Check file dialogs and paths work properly
4. Test with different datasets and videos

### Updating Dependencies:
1. Update requirements.txt
2. Reinstall in virtual environment
3. Test application functionality
4. Rebuild with PyInstaller
5. Test new executable

## Security Considerations

- Executable is not signed (may trigger security warnings)
- Consider code signing for production distribution
- Antivirus software may flag packed executables
- Test on target systems before deployment

## Maintenance

### Regular Updates:
1. Keep PyInstaller updated
2. Update Python dependencies
3. Test builds on new Windows versions
4. Monitor for new PyQt6 requirements

### Build Validation:
- Run automated tests after each build
- Verify UI scaling on different displays
- Test with various dataset formats
- Validate video processing functionality
