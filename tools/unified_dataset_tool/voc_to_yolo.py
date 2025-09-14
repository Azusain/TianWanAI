#!/usr/bin/env python3
"""
standalone VOC to YOLO converter script with progress tracking
"""
import os
import xml.etree.ElementTree as ET
import shutil
from pathlib import Path
import argparse
import time

def voc_to_yolo_conversion(xml_dir, output_dir=None, classes_file=None):
    """Convert VOC format XML files to YOLO format TXT files with detailed progress tracking"""
    start_time = time.time()
    
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
    
    print(f"=== VOC to YOLO Conversion ===")
    print(f"XML Directory: {xml_path}")
    print(f"Images Directory: {images_dir}")
    print(f"Is full VOC dataset: {is_full_voc}")
    
    # load or auto-detect classes
    classes_dict = {}
    auto_detected_classes = set()
    
    if classes_file and os.path.exists(classes_file):
        with open(classes_file, 'r') as f:
            for idx, line in enumerate(f.readlines()):
                class_name = line.strip()
                if class_name:
                    classes_dict[class_name] = idx
        print(f"Loaded {len(classes_dict)} classes from file: {classes_file}")
    else:
        print("Auto-detecting classes from XML files...")
        # auto-detect classes from XML files (scan all files)
        xml_files = list(Path(xml_dir).glob('*.xml'))
        print(f"Found {len(xml_files)} XML files for class detection")
        
        for i, xml_file in enumerate(xml_files):
            if i % 1000 == 0 and i > 0:
                print(f"Scanning classes: {i}/{len(xml_files)} files...")
                
            try:
                tree = ET.parse(xml_file)
                for obj in tree.findall('object'):
                    name_elem = obj.find('name')
                    if name_elem is not None and name_elem.text:
                        class_name = name_elem.text.strip()
                        auto_detected_classes.add(class_name)
            except Exception as e:
                print(f"Warning: Failed to parse {xml_file}: {e}")
                continue
        
        # create classes dictionary
        for idx, class_name in enumerate(sorted(auto_detected_classes)):
            classes_dict[class_name] = idx
            
        print(f"Auto-detected {len(classes_dict)} classes: {', '.join(sorted(auto_detected_classes))}")
    
    if not classes_dict:
        raise Exception("No classes found. Either provide a classes file or ensure XML files contain class names")
    
    # determine output directory
    if output_dir:
        output_root = Path(output_dir)
        labels_root = output_root / "labels"
        images_root = output_root / "images"
    else:
        # create output directory structure - directly output to images/labels
        if is_full_voc:
            output_root = voc_root
            images_root = voc_root / "images"
            labels_root = voc_root / "labels"
        else:
            output_root = xml_path.parent
            images_root = output_root / "images"
            labels_root = output_root / "labels"
    
    # create directories
    images_root.mkdir(parents=True, exist_ok=True)
    labels_root.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_root}")
    print(f"Images output: {images_root}")
    print(f"Labels output: {labels_root}")
    
    # get all XML files
    xml_files = list(Path(xml_dir).glob('*.xml'))
    total_files = len(xml_files)
    print(f"Processing {total_files} XML files...")
    
    converted_count = 0
    copied_images = 0
    skipped_files = []
    error_files = []
    
    processing_start_time = time.time()
    last_report_time = processing_start_time
    last_report_count = 0
    
    for i, xml_file in enumerate(xml_files):
        current_time = time.time()
        elapsed = current_time - processing_start_time
        
        # Progress reporting every 5 seconds or 1000 files
        if (i % 1000 == 0 and i > 0) or (current_time - last_report_time >= 5 and i > last_report_count):
            # Calculate processing speed and ETA
            files_processed = i
            if elapsed > 0:
                files_per_second = files_processed / elapsed
                remaining_files = total_files - files_processed
                eta_seconds = remaining_files / files_per_second if files_per_second > 0 else 0
                
                # Format ETA as MM:SS or HH:MM:SS
                if eta_seconds < 3600:
                    eta_str = f"{int(eta_seconds // 60):02d}:{int(eta_seconds % 60):02d}"
                else:
                    eta_str = f"{int(eta_seconds // 3600):02d}:{int((eta_seconds % 3600) // 60):02d}:{int(eta_seconds % 60):02d}"
                
                progress_pct = (files_processed / total_files) * 100
                print(f"Progress: {files_processed}/{total_files} files ({progress_pct:.1f}%), "
                      f"Speed: {files_per_second:.1f} files/sec, ETA: {eta_str}")
            
            last_report_time = current_time
            last_report_count = i
            
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
            
            # find actual image file - prioritize XML basename over XML filename field
            xml_basename = xml_file.stem
            image_path = None
            
            # Method 1: Try XML basename with various extensions (most reliable)
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']:
                candidate = images_dir / (xml_basename + ext)
                if candidate.exists():
                    image_path = candidate
                    break
            
            # Method 2: If not found, try filename from XML content
            if not image_path:
                filename_elem = root.find('filename')
                if filename_elem is not None and filename_elem.text:
                    image_filename = filename_elem.text.strip()
                    
                    # try exact filename from XML
                    exact_candidate = images_dir / image_filename
                    if exact_candidate.exists():
                        image_path = exact_candidate
                    else:
                        # try filename without extension + various extensions
                        base_name = os.path.splitext(image_filename)[0]
                        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']:
                            candidate = images_dir / (base_name + ext)
                            if candidate.exists():
                                image_path = candidate
                                break
            
            if not image_path:
                skipped_files.append(f"{xml_file.name} (no corresponding image found for base: {xml_basename})")
                continue
            
            # process annotations
            lines = []
            objects = root.findall('object')
            
            for obj in objects:
                name_elem = obj.find('name')
                if name_elem is None or not name_elem.text:
                    continue
                    
                class_name = name_elem.text.strip()
                if class_name not in classes_dict:
                    print(f"Warning: Unknown class '{class_name}' in {xml_file.name}, skipping object")
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
                    print(f"Warning: Invalid bbox in {xml_file.name}, skipping object")
                    continue
                
                # convert to YOLO format (normalized center coordinates and dimensions)
                cx = (xmax + xmin) * 0.5 / width
                cy = (ymax + ymin) * 0.5 / height
                w = (xmax - xmin) / width
                h = (ymax - ymin) / height
                
                # clamp values to [0, 1] to handle any rounding errors
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                w = max(0.0, min(1.0, w))
                h = max(0.0, min(1.0, h))
                
                # skip invalid boxes (width or height is 0)
                if w <= 0 or h <= 0:
                    print(f"Warning: Invalid box dimensions in {xml_file.name}, skipping object")
                    continue
                
                line = f"{label} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
                lines.append(line)
            
            # save YOLO format label file
            label_filename = image_path.stem + ".txt"
            label_path = labels_root / label_filename
            
            with open(label_path, "w", encoding='utf-8') as f:
                f.writelines(lines)
            
            # copy image to output directory
            dest_image_path = images_root / image_path.name
            if not dest_image_path.exists():
                shutil.copy2(image_path, dest_image_path)
                copied_images += 1
            
            converted_count += 1
            
        except Exception as e:
            error_files.append(f"{xml_file.name} (error: {str(e)})")
            print(f"Error processing {xml_file.name}: {e}")
    
    # create classes.txt file in output directory
    classes_file_path = output_root / "classes.txt"
    with open(classes_file_path, 'w', encoding='utf-8') as f:
        for class_name in sorted(classes_dict.keys(), key=lambda x: classes_dict[x]):
            f.write(f"{class_name}\n")
    
    # create data.yaml file
    data_yaml_path = output_root / "data.yaml"
    with open(data_yaml_path, 'w', encoding='utf-8') as f:
        f.write(f"path: {output_root.absolute()}\n")
        f.write("train: images\n")
        f.write("val: images\n")
        f.write(f"nc: {len(classes_dict)}\n")
        f.write("names:\n")
        for idx, class_name in sorted([(v, k) for k, v in classes_dict.items()]):
            f.write(f"  {idx}: {class_name}\n")
    
    # calculate total time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    
    # create summary
    summary_lines = [
        f"VOC to YOLO conversion completed in {time_str}:",
        f"- Processed {len(xml_files)} XML files",
        f"- Successfully converted {converted_count} files",
        f"- Copied {copied_images} unique images",
        f"- Found {len(classes_dict)} classes: {', '.join(sorted(classes_dict.keys()))}",
        f"- Output structure: {images_root} and {labels_root}",
        f"- Created files: {classes_file_path}, {data_yaml_path}"
    ]
    
    if skipped_files:
        summary_lines.append(f"- Skipped {len(skipped_files)} files (missing images)")
        if len(skipped_files) <= 10:
            for skipped in skipped_files:
                summary_lines.append(f"  • {skipped}")
        else:
            for skipped in skipped_files[:5]:
                summary_lines.append(f"  • {skipped}")
            summary_lines.append(f"  • ... and {len(skipped_files) - 5} more")
    
    if error_files:
        summary_lines.append(f"- {len(error_files)} files had errors")
        if len(error_files) <= 10:
            for error in error_files:
                summary_lines.append(f"  • {error}")
        else:
            for error in error_files[:5]:
                summary_lines.append(f"  • {error}")
            summary_lines.append(f"  • ... and {len(error_files) - 5} more")
    
    # final validation
    total_issues = len(skipped_files) + len(error_files)
    if total_issues == 0:
        summary_lines.append("✓ All files processed successfully!")
    elif converted_count > 0:
        success_rate = (converted_count / len(xml_files)) * 100
        summary_lines.append(f"Success rate: {success_rate:.1f}%")
    
    summary = "\n".join(summary_lines)
    print(f"\n{summary}")
    
    return {
        'summary': summary,
        'converted_count': converted_count,
        'copied_images': copied_images,
        'error_count': len(error_files),
        'skipped_count': len(skipped_files),
        'classes': sorted(classes_dict.keys()),
        'output_dir': str(output_root)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PASCAL VOC format to YOLO format")
    parser.add_argument("xml_dir", help="Directory containing XML annotation files")
    parser.add_argument("-o", "--output", help="Output directory (default: create images/labels next to XML dir)")
    parser.add_argument("-c", "--classes", help="Path to classes file (default: auto-detect from XMLs)")
    
    args = parser.parse_args()
    
    result = voc_to_yolo_conversion(
        args.xml_dir,
        args.output,
        args.classes
    )
