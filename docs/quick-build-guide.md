# Quick Build Guide

## building the unified dataset tool executable

### prerequisites
- windows 10/11 (64-bit)
- python 3.9+ installed
- git (optional, for cloning)

### step-by-step build process

1. **open command prompt or powershell**
   ```bash
   # navigate to project directory
   cd path\to\tianwan
   ```

2. **set up virtual environment** (recommended)
   ```bash
   # create virtual environment
   python -m venv venv
   
   # activate it
   venv\Scripts\activate
   ```

3. **install dependencies**
   ```bash
   # install required packages
   pip install -r requirements.txt
   pip install pyinstaller
   ```

4. **run the automated build**
   ```bash
   # method 1: use build script (easiest)
   build.bat
   
   # method 2: run pyinstaller directly
   pyinstaller unified_dataset_tool.spec
   ```

5. **find your executable**
   ```
   dist\unified-dataset-tool.exe
   ```

### what you get
- **size**: ~89 MB standalone executable
- **dependencies**: all included (no separate installation needed)
- **compatibility**: runs on any windows 10+ system

### quick test
```bash
# run the executable
.\dist\unified-dataset-tool.exe
```

the gui should open with 4 main tabs:
- data processing (analysis, split, visualization)
- format conversion (voc to yolo)
- video processing (frame extraction)
- advanced tools (dataset reduction)

### troubleshooting
- if build fails, ensure all dependencies are installed
- check that you're in the correct directory
- virtual environment must be activated
- for errors, see full documentation: `docs/packaging-guide.md`

### distribution
the generated exe file is completely portable - just copy and run on any compatible windows system.

---
*build time: ~1-2 minutes on modern hardware*
