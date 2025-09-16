#!/usr/bin/env python3
"""
YOLO dataset to flat images+labels converter

converts standard YOLO format datasets (with train/val/test structure) 
to flat images + labels format for easier manual adjustment
"""
import os
import shutil
import time
from pathlib import Path
import argparse
from typing import List, Dict, Tuple, Optional

def find_yolo_structure(dataset_path: Path) -> Dict[str, Dict[str, Path]]:
    """detect YOLO dataset structure with enhanced logic for standard formats"""
    structure = {}
    
    print(f"detecting YOLO structure in: {dataset_path}")
    
    # check for standard YOLO structure (train/val/test with images/labels)
    splits_found = []
    for split in ['train', 'val', 'test']:
        split_dir = dataset_path / split
        if split_dir.exists():
            images_dir = split_dir / 'images'
            labels_dir = split_dir / 'labels'
            
            # check if images directory exists with actual image files
            if images_dir.exists():
                image_files = get_image_files(images_dir)
                if image_files:
                    # labels directory is optional (may contain unlabeled images)
                    if not labels_dir.exists():
                        labels_dir.mkdir(parents=True, exist_ok=True)
                        print(f"created missing labels directory: {labels_dir}")
                    
                    structure[split] = {'images': images_dir, 'labels': labels_dir}
                    splits_found.append(split)
                    print(f"found {split} split: {len(image_files)} images in {images_dir}")
    
    # if we found standard YOLO structure, return it
    if structure:
        print(f"detected standard YOLO structure with splits: {', '.join(splits_found)}")
        return structure
    
    # fallback: check for direct images/labels structure
    print("checking for flat images/labels structure...")
    images_dir = dataset_path / 'images'
    labels_dir = dataset_path / 'labels'
    
    if images_dir.exists():
        image_files = get_image_files(images_dir)
        if image_files:
            # labels directory is optional
            if not labels_dir.exists():
                labels_dir.mkdir(parents=True, exist_ok=True)
                print(f"created missing labels directory: {labels_dir}")
            
            structure['main'] = {'images': images_dir, 'labels': labels_dir}
            print(f"detected flat structure: {len(image_files)} images in {images_dir}")
            return structure
    
    # final fallback: check if dataset_path itself contains images directly
    print("checking if dataset path contains images directly...")
    image_files = get_image_files(dataset_path)
    if image_files:
        # create images and labels subdirectories
        images_dir = dataset_path / 'images'
        labels_dir = dataset_path / 'labels'
        
        if not images_dir.exists():
            print(f"images found directly in dataset path, but no images/ subdirectory")
            print(f"this appears to be an unorganized dataset structure")
            print(f"please organize images into an images/ subdirectory first")
    
    return structure

def get_image_files(directory: Path) -> List[Path]:
    """find all image files in directory"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    images = []
    
    if not directory.exists():
        return images
    
    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            images.append(file_path)
    
    return sorted(images)

def yolo_to_flat_conversion(input_dir: str, output_dir: str, 
                          merge_splits: bool = True, 
                          preserve_split_info: bool = True) -> Dict:
    """convert YOLO dataset to flat images + labels format
    
    Args:
        input_dir: input YOLO dataset directory
        output_dir: output directory for flat format
        merge_splits: whether to merge all splits into one folder
        preserve_split_info: whether to add split info to filenames
    """
    start_time = time.time()
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    print(f"=== YOLO to Flat Format Converter ===")
    print(f"input directory: {input_path}")
    print(f"output directory: {output_path}")
    
    # detect dataset structure
    structure = find_yolo_structure(input_path)
    
    if not structure:
        raise ValueError(f"no valid YOLO dataset structure found in {input_path}")
    
    print(f"detected splits: {list(structure.keys())}")
    
    # create output directories
    if merge_splits:
        output_images = output_path / "images"
        output_labels = output_path / "labels"
        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)
    else:
        # keep split structure
        for split_name in structure.keys():
            (output_path / split_name / "images").mkdir(parents=True, exist_ok=True)
            (output_path / split_name / "labels").mkdir(parents=True, exist_ok=True)
    
    # process each split
    total_images = 0
    total_labels = 0
    split_stats = {}
    
    for split_name, split_dirs in structure.items():
        print(f"\nprocessing {split_name} split...")
        
        images_dir = split_dirs['images']
        labels_dir = split_dirs['labels']
        
        # get all images
        image_files = get_image_files(images_dir)
        
        if not image_files:
            print(f"warning: no images found in {images_dir}")
            continue
        
        copied_images = 0
        copied_labels = 0
        
        for image_file in image_files:
            # determine output filename
            if preserve_split_info and merge_splits:
                base_name = image_file.stem
                ext = image_file.suffix
                new_name = f"{base_name}_{split_name}{ext}"
            else:
                new_name = image_file.name
            
            # copy image
            if merge_splits:
                dest_image = output_images / new_name
            else:
                dest_image = output_path / split_name / "images" / new_name
            
            # handle filename conflicts
            counter = 1
            original_dest = dest_image
            while dest_image.exists():
                stem = original_dest.stem
                ext = original_dest.suffix
                if preserve_split_info and merge_splits:
                    dest_image = original_dest.parent / f"{stem}_{counter}{ext}"
                else:
                    dest_image = original_dest.parent / f"{stem}_{counter}{ext}"
                counter += 1
            
            shutil.copy2(image_file, dest_image)
            copied_images += 1
            
            # copy corresponding label if exists
            label_file = labels_dir / f"{image_file.stem}.txt"
            if label_file.exists():
                label_name = f"{dest_image.stem}.txt"
                if merge_splits:
                    dest_label = output_labels / label_name
                else:
                    dest_label = output_path / split_name / "labels" / label_name
                
                shutil.copy2(label_file, dest_label)
                copied_labels += 1
        
        split_stats[split_name] = {
            'images': copied_images,
            'labels': copied_labels
        }
        
        total_images += copied_images
        total_labels += copied_labels
        
        print(f"  {split_name}: {copied_images} images, {copied_labels} labels")
    
    # copy class files
    class_files_copied = []
    for class_file in ['classes.txt', 'data.yaml', 'dataset.yaml']:
        src_file = input_path / class_file
        if src_file.exists():
            dest_file = output_path / class_file
            shutil.copy2(src_file, dest_file)
            class_files_copied.append(class_file)
    
    # create conversion info file
    conversion_info = {
        'source': str(input_path),
        'conversion_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'merge_splits': merge_splits,
        'preserve_split_info': preserve_split_info,
        'splits_processed': list(structure.keys()),
        'total_images': total_images,
        'total_labels': total_labels,
        'split_stats': split_stats,
        'class_files_copied': class_files_copied
    }
    
    import json
    with open(output_path / "conversion_info.json", 'w', encoding='utf-8') as f:
        json.dump(conversion_info, f, indent=2, ensure_ascii=False)
    
    # calculate total time
    total_time = time.time() - start_time
    minutes, seconds = divmod(total_time, 60)
    time_str = f"{int(minutes):02d}:{int(seconds):02d}"
    
    # create summary
    summary_lines = [
        f"YOLO to flat format conversion completed in {time_str}:",
        f"- processed {len(structure)} splits: {', '.join(structure.keys())}",
        f"- total images: {total_images}",
        f"- total labels: {total_labels}",
        f"- merge splits: {'Yes' if merge_splits else 'No'}",
        f"- preserve split info: {'Yes' if preserve_split_info else 'No'}",
        f"- output structure: {output_path}"
    ]
    
    if class_files_copied:
        summary_lines.append(f"- copied class files: {', '.join(class_files_copied)}")
    
    for split_name, stats in split_stats.items():
        summary_lines.append(f"  {split_name}: {stats['images']} images, {stats['labels']} labels")
    
    summary = "\n".join(summary_lines)
    print(f"\n{summary}")
    
    return {
        'summary': summary,
        'total_images': total_images,
        'total_labels': total_labels,
        'split_stats': split_stats,
        'output_dir': str(output_path),
        'conversion_info': conversion_info
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="convert YOLO dataset to flat images+labels format")
    parser.add_argument("input_dir", help="input YOLO dataset directory")
    parser.add_argument("output_dir", help="output directory for flat format")
    parser.add_argument("--no-merge", action="store_true", 
                      help="keep split structure (don't merge into single folder)")
    parser.add_argument("--no-split-info", action="store_true",
                      help="don't add split info to filenames")
    
    args = parser.parse_args()
    
    result = yolo_to_flat_conversion(
        args.input_dir,
        args.output_dir,
        merge_splits=not args.no_merge,
        preserve_split_info=not args.no_split_info
    )
