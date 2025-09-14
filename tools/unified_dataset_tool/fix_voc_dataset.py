#!/usr/bin/env python3
"""
script to fix non-standard PASCAL VOC datasets to comply with standards
fixes the <filename> field in XML files to match the actual image filenames
"""
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
import argparse

def fix_voc_dataset(annotations_dir, images_dir=None, backup=True):
    """Fix VOC dataset to make it compliant with PASCAL VOC standards"""
    annotations_path = Path(annotations_dir)
    
    if not annotations_path.exists():
        raise Exception(f"Annotations directory not found: {annotations_dir}")
    
    # determine images directory
    if images_dir:
        images_path = Path(images_dir)
    else:
        # try to find images directory automatically
        parent_dir = annotations_path.parent
        possible_image_dirs = ["JPEGImages", "images", "Images"]
        
        images_path = None
        for dir_name in possible_image_dirs:
            candidate = parent_dir / dir_name
            if candidate.exists():
                images_path = candidate
                break
        
        if not images_path:
            raise Exception(f"Could not find images directory. Please specify with --images parameter.")
    
    if not images_path.exists():
        raise Exception(f"Images directory not found: {images_path}")
    
    print(f"=== VOC Dataset Fixer ===")
    print(f"Annotations directory: {annotations_path}")
    print(f"Images directory: {images_path}")
    
    # create backup if requested
    if backup:
        backup_dir = annotations_path.parent / "Annotations_backup"
        if backup_dir.exists():
            print(f"Warning: Backup directory already exists: {backup_dir}")
            response = input("Continue without backup? (y/n): ").strip().lower()
            if response != 'y':
                return
        else:
            print(f"Creating backup: {backup_dir}")
            shutil.copytree(annotations_path, backup_dir)
    
    # get all image files for reference
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
    image_files = set()
    for ext in image_extensions:
        for img_file in images_path.glob(f'*{ext}'):
            image_files.add(img_file.stem)  # filename without extension
        for img_file in images_path.glob(f'*{ext.upper()}'):
            image_files.add(img_file.stem)
    
    print(f"Found {len(image_files)} image files in images directory")
    
    # process XML files
    xml_files = list(annotations_path.glob('*.xml'))
    print(f"Found {len(xml_files)} XML files to process")
    
    fixed_count = 0
    error_count = 0
    already_correct = 0
    no_corresponding_image = 0
    
    for i, xml_file in enumerate(xml_files):
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(xml_files)} files...")
        
        try:
            # parse XML
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            xml_basename = xml_file.stem
            
            # check if corresponding image exists
            if xml_basename not in image_files:
                no_corresponding_image += 1
                print(f"Warning: No corresponding image found for {xml_file.name}")
                continue
            
            # find the actual image file with extension
            actual_image_file = None
            for ext in image_extensions:
                candidate = images_path / (xml_basename + ext)
                if candidate.exists():
                    actual_image_file = candidate.name
                    break
            
            if not actual_image_file:
                candidate = images_path / (xml_basename + ext.upper())
                if candidate.exists():
                    actual_image_file = candidate.name
                    break
            
            if not actual_image_file:
                no_corresponding_image += 1
                continue
            
            # check current filename field
            filename_elem = root.find('filename')
            if filename_elem is not None:
                current_filename = filename_elem.text.strip() if filename_elem.text else ""
                
                # check if it's already correct
                if current_filename == actual_image_file:
                    already_correct += 1
                    continue
                
                # fix the filename field
                filename_elem.text = actual_image_file
                fixed_count += 1
                
            else:
                # create filename element if it doesn't exist
                filename_elem = ET.SubElement(root, 'filename')
                filename_elem.text = actual_image_file
                fixed_count += 1
            
            # write back to file
            tree.write(xml_file, encoding='utf-8', xml_declaration=True)
            
        except Exception as e:
            error_count += 1
            print(f"Error processing {xml_file.name}: {e}")
    
    print(f"\n=== Fix Results ===")
    print(f"Total XML files: {len(xml_files)}")
    print(f"Already correct: {already_correct}")
    print(f"Fixed: {fixed_count}")
    print(f"No corresponding image: {no_corresponding_image}")
    print(f"Errors: {error_count}")
    
    if fixed_count > 0:
        print(f"\n✓ Successfully fixed {fixed_count} XML files!")
        print("Dataset is now PASCAL VOC standard compliant.")
    else:
        print("\nNo files needed fixing.")
    
    return {
        'total': len(xml_files),
        'fixed': fixed_count,
        'already_correct': already_correct,
        'no_image': no_corresponding_image,
        'errors': error_count
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix non-standard VOC dataset to be PASCAL VOC compliant")
    parser.add_argument("annotations_dir", help="Directory containing XML annotation files")
    parser.add_argument("--images", help="Directory containing image files (auto-detect if not specified)")
    parser.add_argument("--no-backup", action="store_true", help="Skip creating backup")
    
    args = parser.parse_args()
    
    result = fix_voc_dataset(
        args.annotations_dir,
        args.images,
        backup=not args.no_backup
    )
