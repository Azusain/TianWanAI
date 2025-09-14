#!/usr/bin/env python3
"""
standard PASCAL VOC to YOLO converter that strictly follows VOC specifications
only works with properly formatted VOC datasets
"""
import os
import xml.etree.ElementTree as ET
import shutil
from pathlib import Path
import argparse
import time

def standard_voc_to_yolo(xml_dir, output_dir=None, classes_file=None):
    """Convert standard PASCAL VOC format to YOLO format"""
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
    
    print(f"=== Standard VOC to YOLO Converter ===")
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
        # create output directory structure
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
    
    for i, xml_file in enumerate(xml_files):
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - processing_start_time
            files_per_second = (i + 1) / elapsed if elapsed > 0 else 0
            remaining_files = total_files - (i + 1)
            eta_seconds = remaining_files / files_per_second if files_per_second > 0 else 0
            
            # Format ETA
            if eta_seconds < 60:
                eta_str = f"{int(eta_seconds)}s"
            elif eta_seconds < 3600:
                eta_str = f"{int(eta_seconds // 60):02d}:{int(eta_seconds % 60):02d}"
            else:
                eta_str = f"{int(eta_seconds // 3600):02d}:{int((eta_seconds % 3600) // 60):02d}:{int(eta_seconds % 60):02d}"
            
            progress_pct = ((i + 1) / total_files) * 100
            print(f"Progress: {i + 1}/{total_files} files ({progress_pct:.1f}%), "
                  f"Speed: {files_per_second:.1f} files/sec, ETA: {eta_str}")
            
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
            
            # STANDARD VOC APPROACH: use filename from XML content
            filename_elem = root.find('filename')
            if filename_elem is None or not filename_elem.text:
                error_files.append(f"{xml_file.name} (missing filename element)")
                continue
            
            image_filename = filename_elem.text.strip()
            image_path = images_dir / image_filename
            
            # verify image file exists exactly as specified in XML
            if not image_path.exists():
                skipped_files.append(f"{xml_file.name} (image file not found: {image_filename})")
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
                
                # clamp values to [0, 1]
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                w = max(0.0, min(1.0, w))
                h = max(0.0, min(1.0, h))
                
                # skip invalid boxes
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
    
    # create classes.txt file
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
        f"Standard VOC to YOLO conversion completed in {time_str}:",
        f"- Processed {len(xml_files)} XML files",
        f"- Successfully converted {converted_count} files",
        f"- Copied {copied_images} unique images",
        f"- Found {len(classes_dict)} classes: {', '.join(sorted(classes_dict.keys()))}",
        f"- Output structure: {images_root} and {labels_root}",
        f"- Created files: {classes_file_path}, {data_yaml_path}"
    ]
    
    if skipped_files:
        summary_lines.append(f"- Skipped {len(skipped_files)} files (missing images)")
        if len(skipped_files) <= 5:
            for skipped in skipped_files:
                summary_lines.append(f"  • {skipped}")
        else:
            for skipped in skipped_files[:5]:
                summary_lines.append(f"  • {skipped}")
            summary_lines.append(f"  • ... and {len(skipped_files) - 5} more")
    
    if error_files:
        summary_lines.append(f"- {len(error_files)} files had errors")
        if len(error_files) <= 5:
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
    parser = argparse.ArgumentParser(description="Convert standard PASCAL VOC format to YOLO format")
    parser.add_argument("xml_dir", help="Directory containing XML annotation files")
    parser.add_argument("-o", "--output", help="Output directory (default: create images/labels next to XML dir)")
    parser.add_argument("-c", "--classes", help="Path to classes file (default: auto-detect from XMLs)")
    
    args = parser.parse_args()
    
    result = standard_voc_to_yolo(
        args.xml_dir,
        args.output,
        args.classes
    )
