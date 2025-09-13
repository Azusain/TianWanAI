# unified dataset tool

a comprehensive PyQt6-based dataset management tool for computer vision projects, specifically designed for YOLO format datasets.

## features

- **🎥 video processing**: extract frames from videos (single or batch mode) with configurable intervals
- **📊 dataset analysis**: comprehensive statistics, class distribution, and issue detection
- **🔀 dataset splitting**: split datasets into train/val sets with custom ratios and seeds
- **👁️ sample viewer**: generate visualization samples to check annotation quality
- **📂 smart detection**: automatically detects common dataset directory structures
- **🎯 yolo support**: full support for yolo format annotations and class files

## installation

1. install python 3.8 or higher
2. clone or download this repository
3. install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## usage

### running the application

```bash
python main.py
```

### dataset analysis tab

1. select your dataset directory (should contain images and labels)
2. optionally specify custom images/labels directories
3. click "analyze dataset" to get comprehensive statistics
4. view results including:
   - total image count, labeled/unlabeled counts
   - class distribution and names
   - image size distribution
   - detected issues (missing labels, empty files, invalid annotations)

### dataset split tab

1. select dataset directory and output directory
2. configure split ratio (default 80% train, 20% val)
3. set random seed for reproducible splits
4. click "split dataset" to create train/val directories
5. output includes:
   - `train/images/` and `train/labels/`
   - `val/images/` and `val/labels/`
   - `data.yaml` configuration file

### video processing tab (🎥 抽帧工具)

**single video mode:**
1. click "browse..." next to "video file" and select a video
2. set output directory for extracted frames
3. configure extraction settings:
   - "extract every": interval between frames (default: every 30 frames)
   - "max frames per video": limit total frames (0 = unlimited)
4. click "get video info" to check video details
5. click "extract frames" to start extraction

**batch mode:**
1. click "browse..." next to "or video directory" and select folder with videos
2. set output directory (each video will get its own subfolder)
3. configure same settings as single mode
4. click "extract frames" to process all videos

### sample viewer tab (👁️ 可视化查看)

**purpose**: generate sample images with annotation overlays to check labeling quality

1. select dataset directory
2. choose number of samples to visualize (1-50)
3. optionally specify output directory
4. click "generate visualizations" to create sample images

**what it does**: randomly selects labeled images and draws actual bounding boxes with class labels on them. Uses different colors for different classes and includes class names.

## supported file formats

### images
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)
- WebP (.webp)

### annotations
- YOLO format (.txt files)
- format: `class_id center_x center_y width height` (normalized coordinates)

### class names
the tool automatically searches for class name files in the dataset directory:
- `classes.txt` - one class name per line
- `data.yaml` - YOLO format with `names` field
- `dataset.yaml` - similar to data.yaml

## directory structure examples

### standard yolo structure
```
dataset/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── labels/
│   ├── image1.txt
│   ├── image2.txt
│   └── ...
├── classes.txt
└── data.yaml
```

### custom structure
```
my_dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── classes.txt
```

## ⚙️ auto-detection behavior

**IMPORTANT**: The tool has smart auto-detection features to reduce configuration complexity:

### Dataset Analysis & Splitting
- If you only specify the main dataset directory, the tool will automatically search for:
  - **Images**: `images/`, `train/images/`, `val/images/`, then the root directory
  - **Labels**: `labels/`, `train/labels/`, `val/labels/`, then the root directory
- **Override**: Use the optional "images directory" and "labels directory" fields to specify custom locations

### Class Names
Automatically searches for class definitions in this order:
1. `classes.txt` (one class per line)
2. `data.yaml` (YOLO format with `names` field)
3. `dataset.yaml` (similar format)

### Video Processing
- **Single mode**: Select one video file
- **Batch mode**: Select a folder - processes all supported video files
- Each video gets its own output subfolder automatically

## requirements

- Python 3.8+
- PyQt6
- Pillow (PIL)
- PyYAML
- NumPy
- Matplotlib
- OpenCV (optional, for advanced visualization features)

## features in detail

### analysis features
- validates annotation format and coordinates
- detects missing label files
- identifies empty annotation files
- reports invalid coordinate values
- calculates class distribution statistics
- analyzes image size distribution

### splitting features
- maintains corresponding image-label pairs
- supports custom train/validation ratios
- reproducible splits with random seeds
- creates YOLO-compatible directory structure
- generates data.yaml configuration

### visualization features
- random sampling from labeled images
- real bounding box overlay with OpenCV integration
- class name display with colored backgrounds
- configurable sample count
- automatic color assignment per class

## troubleshooting

1. **import errors**: make sure all dependencies are installed
2. **path issues**: use absolute paths or ensure working directory is correct
3. **permission errors**: ensure write access to output directories
4. **large datasets**: processing may take time for datasets with thousands of images

## contributing

this tool was created to streamline dataset management workflows. contributions and improvements are welcome!
