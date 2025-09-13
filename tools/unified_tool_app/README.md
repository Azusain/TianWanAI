# unified tool app

a comprehensive desktop application that integrates dataset management, format conversion, and video processing tools into a single tauri-based interface.

## features

### dataset management
- analyze yolo format datasets
- split datasets into train/validation sets
- visualize dataset samples with annotations
- detect and report dataset issues

### format conversion
- convert pascal voc xml annotations to yolo format
- batch conversion support
- class mapping through external files

### video processing
- extract frames from videos with customizable intervals
- batch processing of multiple videos
- support for multiple video formats (mp4, avi, mov, etc.)
- configurable output formats (jpg, png) and quality settings

## requirements

- rust 1.70+ 
- tauri dependencies
- windows 10+ (current configuration)

## building and running

### development mode
```powershell
# navigate to the project directory
cd "C:\Users\azusaing\Desktop\Code\tianwan\tools\unified_tool_app"

# run in development mode
cargo tauri dev
```

### building for release
```powershell
# build the application
cargo tauri build
```

### installing dependencies
if you encounter dependency issues, make sure you have:
- visual studio build tools
- windows sdk
- tauri prerequisites

## project structure

```
unified_tool_app/
├── src/                    # rust backend
│   ├── main.rs            # application entry point
│   ├── commands.rs        # tauri command handlers
│   ├── dataset.rs         # dataset management module
│   ├── format_converter.rs # voc to yolo conversion
│   ├── video_processor.rs # video frame extraction
│   └── utils.rs           # utility functions
├── dist/                  # frontend assets
│   ├── index.html         # main html page
│   ├── style.css          # application styles
│   └── script.js          # frontend logic
├── Cargo.toml             # rust dependencies
├── tauri.conf.json        # tauri configuration
└── build.rs               # build script
```

## usage

1. **dataset management**: 
   - select your dataset directory
   - analyze dataset statistics and quality
   - split into training and validation sets
   - visualize samples with bounding boxes

2. **format conversion**:
   - provide voc xml directory and classes.txt file
   - convert to yolo format with normalized coordinates
   - batch process multiple directories

3. **video processing**:
   - select video files and output directories
   - configure frame extraction parameters
   - batch process multiple videos

## development notes

this application integrates functionality from the original python tools:
- `dataset_manager.py` → dataset management module
- `pac2y8.py` → format conversion module  
- `video_processor.py` → video processing module

the tauri framework provides:
- secure ipc between frontend and backend
- native file system access
- cross-platform compatibility
- modern web-based ui with rust performance

## future enhancements

- add more annotation formats support
- implement opencv integration for better video processing
- add dataset augmentation features
- support for more image formats
- progress indicators for long-running operations
