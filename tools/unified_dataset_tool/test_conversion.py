#!/usr/bin/env python3
"""
command line test for VOC to YOLO conversion
"""
import os
import xml.etree.ElementTree as ET
import shutil
from pathlib import Path

def voc_to_yolo_test(xml_dir, classes_file=None):
    """test VOC to YOLO conversion process"""
    xml_path = Path(xml_dir)
    
    # determine if this is a full VOC dataset or just annotations directory
    is_full_voc = False
    voc_root = xml_path
    images_dir = xml_path  # default to same directory
    
    # check if this is standard VOC structure
    if xml_path.name == "Annotations":
        voc_root = xml_path.parent
        if (voc_root / "JPEGImages").exists():
            images_dir = voc_root / "JPEGImages"
            is_full_voc = True
        elif (voc_root / "images").exists():
            images_dir = voc_root / "images"
            is_full_voc = True
    else:
        # check if images directory exists in same parent
        possible_img_dirs = ["JPEGImages", "images", "Images"]
        for img_dir_name in possible_img_dirs:
            candidate = xml_path.parent / img_dir_name
            if candidate.exists():
                images_dir = candidate
                is_full_voc = True
                break
    
    print(f"processing XML directory: {xml_path}")
    print(f"processing images directory: {images_dir}")
    print(f"is full VOC dataset: {is_full_voc}")
    
    # load or auto-detect classes
    classes_dict = {}
    auto_detected_classes = set()
    
    if classes_file and os.path.exists(classes_file):
        with open(classes_file, 'r') as f:
            for idx, line in enumerate(f.readlines()):
                class_name = line.strip()
                if class_name:
                    classes_dict[class_name] = idx
        print(f"loaded {len(classes_dict)} classes from file: {classes_file}")
    else:
        print("auto-detecting classes from XML files...")
        # auto-detect classes from XML files - sample first 100 files for speed
        xml_files = list(Path(xml_dir).glob('*.xml'))
        sample_files = xml_files[:min(100, len(xml_files))]
        print(f"sampling {len(sample_files)} files for class detection")
        
        for xml_file in sample_files:
            try:
                tree = ET.parse(xml_file)
                for obj in tree.findall('object'):
                    name_elem = obj.find('name')
                    if name_elem is not None and name_elem.text:
                        class_name = name_elem.text.strip()
                        auto_detected_classes.add(class_name)
            except Exception as e:
                print(f"warning: failed to parse {xml_file}: {e}")
                continue
        
        # create classes dictionary
        for idx, class_name in enumerate(sorted(auto_detected_classes)):
            classes_dict[class_name] = idx
        
        print(f"auto-detected {len(classes_dict)} classes: {', '.join(sorted(auto_detected_classes))}")
    
    if not classes_dict:
        raise Exception("no classes found. either provide a classes file or ensure XML files contain class names")
    
    # get all XML files
    xml_files = list(Path(xml_dir).glob('*.xml'))
    print(f"found {len(xml_files)} XML files for processing")
    
    # process first 10 files for testing
    test_files = xml_files[:10]
    print(f"testing conversion of first {len(test_files)} files:")
    
    converted_count = 0
    error_files = []
    
    for i, xml_file in enumerate(test_files):
        try:
            # parse XML
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # get image dimensions
            size = root.find('size')
            if size is None:
                error_files.append(f"{xml_file.name} (no size element)")
                continue
            
            width_elem = size.find('width')
            height_elem = size.find('height')
            
            if width_elem is None or height_elem is None:
                error_files.append(f"{xml_file.name} (missing width/height)")
                continue
            
            try:
                width = int(width_elem.text)
                height = int(height_elem.text)
            except (ValueError, TypeError):
                error_files.append(f"{xml_file.name} (invalid width/height values)")
                continue
            
            # get image filename from XML or use XML basename
            filename_elem = root.find('filename')
            if filename_elem is not None and filename_elem.text:
                image_filename = filename_elem.text.strip()
            else:
                xml_basename = xml_file.stem
                image_filename = xml_basename + ".jpg"
            
            # find actual image file
            image_path = None
            base_name = os.path.splitext(image_filename)[0]
            
            # try to find image with various extensions
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']:
                candidate = images_dir / (base_name + ext)
                if candidate.exists():
                    image_path = candidate
                    break
            
            if not image_path:
                error_files.append(f"{xml_file.name} (no corresponding image found)")
                continue
            
            # process annotations
            lines = []
            objects = root.findall('object')
            
            print(f"  {xml_file.name}: {width}x{height}, {len(objects)} objects")
            
            for obj in objects:
                name_elem = obj.find('name')
                if name_elem is None or not name_elem.text:
                    continue
                
                class_name = name_elem.text.strip()
                if class_name not in classes_dict:
                    print(f"    warning: unknown class '{class_name}', skipping object")
                    continue
                
                label = classes_dict[class_name]
                bbox = obj.find('bndbox')
                
                if bbox is None:
                    continue
                
                try:
                    xmin = float(bbox.find('xmin').text)
                    ymin = float(bbox.find('ymin').text)
                    xmax = float(bbox.find('xmax').text)
                    ymax = float(bbox.find('ymax').text)
                except (AttributeError, ValueError, TypeError):
                    print(f"    warning: invalid bbox, skipping object")
                    continue
                
                # convert to YOLO format (normalized center coordinates and dimensions)
                cx = (xmax + xmin) * 0.5 / width
                cy = (ymax + ymin) * 0.5 / height
                w = (xmax - xmin) / width
                h = (ymax - ymin) / height
                
                # clamp values to [0, 1]
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                w = max(0.0, min(1.0, w))
                h = max(0.0, min(1.0, h))
                
                # skip invalid boxes
                if w <= 0 or h <= 0:
                    print(f"    warning: invalid box dimensions, skipping object")
                    continue
                
                line = f"{label} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                lines.append(line)
                print(f"    -> {class_name}: {line}")
            
            converted_count += 1
            
        except Exception as e:
            error_files.append(f"{xml_file.name} (error: {str(e)})")
            print(f"  error processing {xml_file.name}: {e}")
    
    print(f"\n=== Test Results ===")
    print(f"tested {len(test_files)} files")
    print(f"successfully processed: {converted_count}")
    print(f"errors: {len(error_files)}")
    
    if error_files:
        print(f"error details:")
        for error in error_files:
            print(f"  - {error}")
    
    print(f"\nif this looks good, full conversion should process all {len(xml_files)} files")

if __name__ == "__main__":
    xml_directory = r"C:\Users\azusaing\Desktop\VOC2028\Annotations"
    voc_to_yolo_test(xml_directory)
