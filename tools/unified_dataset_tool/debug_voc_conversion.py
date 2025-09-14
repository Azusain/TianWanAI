#!/usr/bin/env python3
"""
debug script to analyze VOC to YOLO conversion issues
"""
import os
import xml.etree.ElementTree as ET
from pathlib import Path

def analyze_voc_dataset(voc_path):
    """Analyze VOC dataset for conversion issues"""
    voc_root = Path(voc_path)
    annotations_dir = voc_root / "Annotations"
    images_dir = voc_root / "JPEGImages"
    
    print("=== VOC Dataset Analysis ===")
    print(f"dataset path: {voc_root}")
    print(f"annotations directory: {annotations_dir}")
    print(f"images directory: {images_dir}")
    
    if not annotations_dir.exists():
        print("error: annotations directory not found!")
        return
    
    if not images_dir.exists():
        print("error: images directory not found!")
        return
    
    # get all XML files
    xml_files = list(annotations_dir.glob('*.xml'))
    print(f"found {len(xml_files)} XML files")
    
    # get all image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        image_files.extend(images_dir.glob(f'*{ext}'))
    print(f"found {len(image_files)} image files")
    
    # analyze XML files
    valid_files = 0
    empty_files = 0
    error_files = 0
    total_objects = 0
    classes_found = set()
    error_details = []
    
    for i, xml_file in enumerate(xml_files):
        if i % 1000 == 0 and i > 0:
            print(f"processed {i}/{len(xml_files)} files...")
            
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # check for valid structure
            if root.tag != 'annotation':
                error_files += 1
                error_details.append(f"{xml_file.name}: invalid root tag '{root.tag}'")
                continue
            
            # get objects
            objects = root.findall('object')
            if len(objects) == 0:
                empty_files += 1
            else:
                valid_files += 1
                total_objects += len(objects)
                
                # collect classes
                for obj in objects:
                    name_elem = obj.find('name')
                    if name_elem is not None and name_elem.text:
                        classes_found.add(name_elem.text.strip())
            
        except Exception as e:
            error_files += 1
            error_details.append(f"{xml_file.name}: {str(e)}")
    
    print("\n=== Analysis Results ===")
    print(f"total XML files: {len(xml_files)}")
    print(f"valid files (with objects): {valid_files}")
    print(f"empty files (no objects): {empty_files}")
    print(f"error files: {error_files}")
    print(f"total objects: {total_objects}")
    print(f"unique classes: {len(classes_found)}")
    print(f"classes found: {', '.join(sorted(classes_found))}")
    
    if error_details:
        print(f"\n=== Error Details (first 10) ===")
        for error in error_details[:10]:
            print(f"- {error}")
    
    # check image-xml matching
    xml_basenames = {f.stem for f in xml_files}
    img_basenames = {f.stem for f in image_files}
    
    matched = len(xml_basenames & img_basenames)
    missing_xml = len(img_basenames - xml_basenames)
    missing_img = len(xml_basenames - img_basenames)
    
    print(f"\n=== File Matching ===")
    print(f"matched pairs: {matched}")
    print(f"images missing XML: {missing_xml}")
    print(f"XML missing images: {missing_img}")
    
    # estimate conversion results
    convertible_files = valid_files - len([e for e in error_details if 'missing images' not in e])
    print(f"\n=== Conversion Estimate ===")
    print(f"files that should convert successfully: {convertible_files}")
    print(f"files that will be skipped: {len(xml_files) - convertible_files}")
    
    return {
        'total_xml': len(xml_files),
        'valid_files': valid_files,
        'empty_files': empty_files,
        'error_files': error_files,
        'convertible_files': convertible_files,
        'classes': sorted(classes_found)
    }

if __name__ == "__main__":
    voc_dataset_path = r"C:\Users\azusaing\Desktop\VOC2028"
    result = analyze_voc_dataset(voc_dataset_path)
