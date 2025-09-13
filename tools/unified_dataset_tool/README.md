# unified dataset tool

a comprehensive PyQt6-based dataset management tool for computer vision projects, specifically designed for YOLO format datasets.

## features

- **dataset analysis**: comprehensive statistics including class distribution, image sizes, annotation counts, and issue detection
- **dataset splitting**: split datasets into train/validation sets with configurable ratios and random seeds
- **visualization**: generate sample visualizations with bounding box overlays
- **format support**: YOLO format annotations with automatic class name loading from various config files

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

### visualization tab

1. select dataset directory
2. choose number of samples to visualize (1-50)
3. optionally specify output directory
4. click "generate visualizations" to create sample images with annotations

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

the tool will auto-detect common directory structures or you can specify custom paths.

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
- bounding box overlay (placeholder implementation)
- class name display
- configurable sample count

## troubleshooting

1. **import errors**: make sure all dependencies are installed
2. **path issues**: use absolute paths or ensure working directory is correct
3. **permission errors**: ensure write access to output directories
4. **large datasets**: processing may take time for datasets with thousands of images

## contributing

this tool was created to streamline dataset management workflows. contributions and improvements are welcome!
