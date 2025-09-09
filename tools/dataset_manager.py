#!/usr/bin/env python3
"""
Unified Dataset Management Tool

Combines dataset management functionalities:
- Dataset splitting (train/validation with various strategies)
- Dataset statistics and analysis
- Dataset visualization with bounding box overlay
- Missing label detection
- Support for YOLO format datasets
- Comprehensive dataset health checks

Replaces: dataset_splitter.py, split_mouse_dataset_v6.py, split_train_test.py,
         dataset_statistics.py, visualize_dataset.py, find_missing_labels.py
"""

import os
import shutil
import random
import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from loguru import logger

class DatasetManager:
    def __init__(self, dataset_path: str):
        """
        Initialize dataset manager
        
        Args:
            dataset_path: path to dataset root directory
        """
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"dataset path {dataset_path} does not exist")
        
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        self.stats_cache = {}
        
    def _find_images(self, directory: Path) -> List[Path]:
        """Find all image files in directory"""
        images = []
        for ext in self.image_extensions:
            images.extend(directory.glob(f"*{ext}"))
            images.extend(directory.glob(f"*{ext.upper()}"))
        return images
    
    def _read_yolo_label(self, label_path: Path) -> List[Dict]:
        """
        Read YOLO format label file
        
        Returns:
            list of annotations, each containing: class_id, center_x, center_y, width, height
        """
        annotations = []
        if not label_path.exists() or label_path.stat().st_size == 0:
            return annotations
        
        try:
            with open(label_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) != 5:
                        logger.warning(f"invalid annotation in {label_path}:{line_num}: {line}")
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        center_x = float(parts[1])
                        center_y = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # validate ranges
                        if not (0 <= center_x <= 1 and 0 <= center_y <= 1 and 
                               0 < width <= 1 and 0 < height <= 1):
                            logger.warning(f"invalid coordinates in {label_path}:{line_num}: {line}")
                            continue
                        
                        annotations.append({
                            'class_id': class_id,
                            'center_x': center_x,
                            'center_y': center_y,
                            'width': width,
                            'height': height
                        })
                    except ValueError as e:
                        logger.warning(f"parse error in {label_path}:{line_num}: {e}")
                        continue
        except Exception as e:
            logger.error(f"error reading {label_path}: {e}")
        
        return annotations
    
    def _load_class_names(self, classes_file: Optional[Path] = None) -> Dict[int, str]:
        """Load class names from classes.txt or data.yaml"""
        class_names = {}
        
        # try specified classes file
        if classes_file and classes_file.exists():
            try:
                with open(classes_file, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        line = line.strip()
                        if line:
                            class_names[i] = line
                logger.info(f"loaded {len(class_names)} classes from {classes_file}")
                return class_names
            except Exception as e:
                logger.warning(f"failed to load classes from {classes_file}: {e}")
        
        # try common locations
        for filename in ['classes.txt', 'data.yaml', 'dataset.yaml']:
            file_path = self.dataset_path / filename
            if not file_path.exists():
                continue
            
            try:
                if filename.endswith('.yaml') or filename.endswith('.yml'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)
                    
                    if 'names' in data:
                        if isinstance(data['names'], dict):
                            class_names = {int(k): str(v) for k, v in data['names'].items()}
                        elif isinstance(data['names'], list):
                            class_names = {i: name for i, name in enumerate(data['names'])}
                        logger.info(f"loaded {len(class_names)} classes from {file_path}")
                        return class_names
                else:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for i, line in enumerate(f):
                            line = line.strip()
                            if line:
                                class_names[i] = line
                    logger.info(f"loaded {len(class_names)} classes from {file_path}")
                    return class_names
            except Exception as e:
                logger.warning(f"failed to load classes from {file_path}: {e}")
        
        logger.warning("no class names found, using generic names")
        return {}
    
    def analyze_dataset(self, images_dir: Optional[str] = None, 
                       labels_dir: Optional[str] = None) -> Dict:
        """
        Comprehensive dataset analysis
        
        Args:
            images_dir: images directory (auto-detect if None)
            labels_dir: labels directory (auto-detect if None)
            
        Returns:
            detailed analysis results
        """
        logger.info("starting dataset analysis...")
        
        # auto-detect directory structure
        if images_dir is None:
            for possible_dir in ['images', 'train/images', 'val/images', '.']:
                candidate = self.dataset_path / possible_dir
                if candidate.exists() and self._find_images(candidate):
                    images_dir = str(candidate)
                    break
        
        if labels_dir is None:
            for possible_dir in ['labels', 'train/labels', 'val/labels', '.']:
                candidate = self.dataset_path / possible_dir
                if candidate.exists():
                    labels_dir = str(candidate)
                    break
        
        images_path = Path(images_dir) if images_dir else self.dataset_path
        labels_path = Path(labels_dir) if labels_dir else self.dataset_path
        
        logger.info(f"images directory: {images_path}")
        logger.info(f"labels directory: {labels_path}")
        
        # find all images
        image_files = self._find_images(images_path)
        logger.info(f"found {len(image_files)} image files")
        
        if not image_files:
            logger.error("no image files found")
            return {}
        
        # analyze images and labels
        analysis = {
            'dataset_path': str(self.dataset_path),
            'images_dir': str(images_path),
            'labels_dir': str(labels_path),
            'total_images': len(image_files),
            'labeled_images': 0,
            'unlabeled_images': 0,
            'empty_labels': 0,
            'total_annotations': 0,
            'class_distribution': defaultdict(int),
            'image_sizes': defaultdict(int),
            'annotation_stats': {
                'min_bbox_area': float('inf'),
                'max_bbox_area': 0,
                'avg_bbox_area': 0,
                'min_objects_per_image': float('inf'),
                'max_objects_per_image': 0,
                'avg_objects_per_image': 0
            },
            'issues': {
                'missing_labels': [],
                'empty_label_files': [],
                'corrupted_labels': [],
                'invalid_annotations': []
            }
        }
        
        bbox_areas = []
        objects_per_image = []
        
        for img_file in image_files:
            # check corresponding label file
            label_file = labels_path / f"{img_file.stem}.txt"
            
            if not label_file.exists():
                analysis['unlabeled_images'] += 1
                analysis['issues']['missing_labels'].append(str(img_file))
                continue
            
            # read annotations
            annotations = self._read_yolo_label(label_file)
            
            if not annotations:
                analysis['empty_labels'] += 1
                analysis['issues']['empty_label_files'].append(str(label_file))
            else:
                analysis['labeled_images'] += 1
                analysis['total_annotations'] += len(annotations)
                objects_per_image.append(len(annotations))
                
                # analyze annotations
                for ann in annotations:
                    analysis['class_distribution'][ann['class_id']] += 1
                    
                    # calculate bbox area (normalized)
                    bbox_area = ann['width'] * ann['height']
                    bbox_areas.append(bbox_area)
            
            # get image size
            try:
                img = cv2.imread(str(img_file))
                if img is not None:
                    h, w = img.shape[:2]
                    size_key = f"{w}x{h}"
                    analysis['image_sizes'][size_key] += 1
                else:
                    analysis['issues']['corrupted_labels'].append(str(img_file))
            except Exception as e:
                logger.warning(f"failed to read image {img_file}: {e}")
                analysis['issues']['corrupted_labels'].append(str(img_file))
        
        # calculate statistics
        if bbox_areas:
            analysis['annotation_stats']['min_bbox_area'] = min(bbox_areas)
            analysis['annotation_stats']['max_bbox_area'] = max(bbox_areas)
            analysis['annotation_stats']['avg_bbox_area'] = sum(bbox_areas) / len(bbox_areas)
        
        if objects_per_image:
            analysis['annotation_stats']['min_objects_per_image'] = min(objects_per_image)
            analysis['annotation_stats']['max_objects_per_image'] = max(objects_per_image)
            analysis['annotation_stats']['avg_objects_per_image'] = sum(objects_per_image) / len(objects_per_image)
        
        # load class names
        class_names = self._load_class_names()
        analysis['class_names'] = class_names
        
        self.stats_cache = analysis
        
        logger.success("dataset analysis complete!")
        return analysis
    
    def print_analysis(self, analysis: Optional[Dict] = None):
        """Print formatted analysis results"""
        if analysis is None:
            analysis = self.stats_cache
        
        if not analysis:
            logger.error("no analysis data available")
            return
        
        print("\n" + "="*60)
        print("DATASET ANALYSIS REPORT")
        print("="*60)
        
        print(f"Dataset path: {analysis['dataset_path']}")
        print(f"Images directory: {analysis['images_dir']}")
        print(f"Labels directory: {analysis['labels_dir']}")
        
        print(f"\nImage Statistics:")
        print(f"  Total images: {analysis['total_images']}")
        print(f"  Labeled images: {analysis['labeled_images']}")
        print(f"  Unlabeled images: {analysis['unlabeled_images']}")
        print(f"  Empty label files: {analysis['empty_labels']}")
        
        if analysis['total_annotations'] > 0:
            print(f"\nAnnotation Statistics:")
            print(f"  Total annotations: {analysis['total_annotations']}")
            print(f"  Avg objects per image: {analysis['annotation_stats']['avg_objects_per_image']:.1f}")
            print(f"  Min/Max objects per image: {analysis['annotation_stats']['min_objects_per_image']}/{analysis['annotation_stats']['max_objects_per_image']}")
            print(f"  Avg bbox area: {analysis['annotation_stats']['avg_bbox_area']:.4f}")
            print(f"  Min/Max bbox area: {analysis['annotation_stats']['min_bbox_area']:.4f}/{analysis['annotation_stats']['max_bbox_area']:.4f}")
        
        if analysis['class_distribution']:
            print(f"\nClass Distribution:")
            class_names = analysis.get('class_names', {})
            for class_id, count in sorted(analysis['class_distribution'].items()):
                class_name = class_names.get(class_id, f"class_{class_id}")
                percentage = count / analysis['total_annotations'] * 100
                print(f"  {class_id}: {class_name} - {count} ({percentage:.1f}%)")
        
        if analysis['image_sizes']:
            print(f"\nImage Sizes:")
            for size, count in sorted(analysis['image_sizes'].items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {size}: {count} images")
        
        # issues
        issues = analysis['issues']
        total_issues = sum(len(issue_list) for issue_list in issues.values())
        
        if total_issues > 0:
            print(f"\nIssues Found ({total_issues} total):")
            for issue_type, issue_list in issues.items():
                if issue_list:
                    print(f"  {issue_type}: {len(issue_list)} files")
                    if len(issue_list) <= 5:
                        for item in issue_list:
                            print(f"    - {Path(item).name}")
                    else:
                        print(f"    - {Path(issue_list[0]).name} ... and {len(issue_list)-1} more")
        else:
            print("\n✅ No issues found!")
        
        print("\n" + "="*60)
    
    def split_dataset(self, output_path: str, train_ratio: float = 0.8,
                     split_mode: str = "simple", seed: int = 42,
                     images_dir: Optional[str] = None, labels_dir: Optional[str] = None) -> bool:
        """
        Split dataset into train/validation sets
        
        Args:
            output_path: output directory for split dataset
            train_ratio: ratio for training set
            split_mode: 'simple' or 'enhanced' (with unlabeled handling)
            seed: random seed for reproducibility
            images_dir: source images directory
            labels_dir: source labels directory
            
        Returns:
            success status
        """
        random.seed(seed)
        output_path = Path(output_path)
        
        logger.info(f"splitting dataset with {train_ratio:.1%} train ratio")
        logger.info(f"split mode: {split_mode}")
        logger.info(f"output path: {output_path}")
        
        # auto-detect source directories
        if images_dir is None:
            for possible_dir in ['images', '.']:
                candidate = self.dataset_path / possible_dir
                if candidate.exists() and self._find_images(candidate):
                    images_dir = str(candidate)
                    break
        
        if labels_dir is None:
            for possible_dir in ['labels', '.']:
                candidate = self.dataset_path / possible_dir
                if candidate.exists():
                    labels_dir = str(candidate)
                    break
        
        images_path = Path(images_dir) if images_dir else self.dataset_path
        labels_path = Path(labels_dir) if labels_dir else self.dataset_path
        
        logger.info(f"source images: {images_path}")
        logger.info(f"source labels: {labels_path}")
        
        # find all images
        image_files = self._find_images(images_path)
        if not image_files:
            logger.error("no image files found")
            return False
        
        # separate labeled and unlabeled images
        labeled_pairs = []
        unlabeled_images = []
        
        for img_file in image_files:
            label_file = labels_path / f"{img_file.stem}.txt"
            
            if label_file.exists() and label_file.stat().st_size > 0:
                # verify label file is valid
                annotations = self._read_yolo_label(label_file)
                if annotations:  # has valid annotations
                    labeled_pairs.append((img_file, label_file))
                else:
                    unlabeled_images.append(img_file)
            else:
                unlabeled_images.append(img_file)
        
        logger.info(f"labeled pairs: {len(labeled_pairs)}")
        logger.info(f"unlabeled images: {len(unlabeled_images)}")
        
        if len(labeled_pairs) == 0:
            logger.error("no labeled data found")
            return False
        
        # shuffle data
        random.shuffle(labeled_pairs)
        random.shuffle(unlabeled_images)
        
        # calculate split sizes
        labeled_train_size = int(len(labeled_pairs) * train_ratio)
        unlabeled_train_size = int(len(unlabeled_images) * train_ratio) if unlabeled_images else 0
        
        labeled_train = labeled_pairs[:labeled_train_size]
        labeled_val = labeled_pairs[labeled_train_size:]
        unlabeled_train = unlabeled_images[:unlabeled_train_size]
        unlabeled_val = unlabeled_images[unlabeled_train_size:]
        
        logger.info(f"split sizes:")
        logger.info(f"  labeled train: {len(labeled_train)}")
        logger.info(f"  labeled val: {len(labeled_val)}")
        logger.info(f"  unlabeled train: {len(unlabeled_train)}")
        logger.info(f"  unlabeled val: {len(unlabeled_val)}")
        
        # create output structure
        output_path.mkdir(parents=True, exist_ok=True)
        
        train_images_dir = output_path / "train" / "images"
        train_labels_dir = output_path / "train" / "labels"
        val_images_dir = output_path / "val" / "images"
        val_labels_dir = output_path / "val" / "labels"
        
        for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        if split_mode == "enhanced" and unlabeled_images:
            train_unlabeled_dir = output_path / "train" / "unlabeled"
            val_unlabeled_dir = output_path / "val" / "unlabeled"
            train_unlabeled_dir.mkdir(exist_ok=True)
            val_unlabeled_dir.mkdir(exist_ok=True)
        
        # copy files
        def copy_with_progress(file_pairs, desc):
            for i, (src_img, src_label) in enumerate(file_pairs):
                if i % 100 == 0 or i < 10:
                    logger.info(f"{desc}: {i+1}/{len(file_pairs)}")
                yield src_img, src_label
        
        # copy labeled training data
        logger.info("copying labeled training data...")
        for img_file, label_file in labeled_train:
            shutil.copy2(img_file, train_images_dir / img_file.name)
            shutil.copy2(label_file, train_labels_dir / label_file.name)
        
        # copy labeled validation data
        logger.info("copying labeled validation data...")
        for img_file, label_file in labeled_val:
            shutil.copy2(img_file, val_images_dir / img_file.name)
            shutil.copy2(label_file, val_labels_dir / label_file.name)
        
        if split_mode == "enhanced":
            # copy unlabeled training data
            if unlabeled_train:
                logger.info("copying unlabeled training data...")
                for img_file in unlabeled_train:
                    shutil.copy2(img_file, train_unlabeled_dir / img_file.name)
                    # create empty label file for consistency
                    empty_label = train_labels_dir / f"{img_file.stem}.txt"
                    empty_label.touch()
            
            # copy unlabeled validation data
            if unlabeled_val:
                logger.info("copying unlabeled validation data...")
                for img_file in unlabeled_val:
                    shutil.copy2(img_file, val_unlabeled_dir / img_file.name)
                    # create empty label file for consistency
                    empty_label = val_labels_dir / f"{img_file.stem}.txt"
                    empty_label.touch()
        
        # copy class names if available
        for classes_file in ['classes.txt', 'data.yaml', 'dataset.yaml']:
            src_file = self.dataset_path / classes_file
            if src_file.exists():
                shutil.copy2(src_file, output_path / classes_file)
                logger.info(f"copied {classes_file}")
        
        # create data.yaml
        self._create_data_yaml(output_path, len(labeled_train), len(labeled_val),
                              len(unlabeled_train), len(unlabeled_val), split_mode)
        
        logger.success(f"dataset split complete! output: {output_path}")
        return True
    
    def _create_data_yaml(self, output_path: Path, labeled_train: int, labeled_val: int,
                         unlabeled_train: int, unlabeled_val: int, split_mode: str):
        """Create data.yaml configuration file"""
        
        class_names = self._load_class_names()
        
        yaml_content = {
            'path': str(output_path.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'names': class_names if class_names else {0: 'object'}
        }
        
        if split_mode == "enhanced":
            yaml_content.update({
                'train_unlabeled': 'train/unlabeled',
                'val_unlabeled': 'val/unlabeled'
            })
        
        # add statistics
        yaml_content['statistics'] = {
            'labeled_train_samples': labeled_train,
            'labeled_val_samples': labeled_val,
            'unlabeled_train_samples': unlabeled_train,
            'unlabeled_val_samples': unlabeled_val,
            'total_samples': labeled_train + labeled_val + unlabeled_train + unlabeled_val
        }
        
        with open(output_path / 'data.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
        
        logger.info("created data.yaml configuration")
    
    def visualize_samples(self, output_dir: Optional[str] = None, 
                         sample_count: int = 5, images_dir: Optional[str] = None,
                         labels_dir: Optional[str] = None, save_images: bool = True) -> bool:
        """
        Visualize random dataset samples with annotations
        
        Args:
            output_dir: directory to save visualization images
            sample_count: number of samples to visualize
            images_dir: images directory
            labels_dir: labels directory
            save_images: whether to save visualization images
            
        Returns:
            success status
        """
        logger.info(f"visualizing {sample_count} random samples...")
        
        # auto-detect directories
        if images_dir is None:
            for possible_dir in ['images', 'train/images', '.']:
                candidate = self.dataset_path / possible_dir
                if candidate.exists() and self._find_images(candidate):
                    images_dir = str(candidate)
                    break
        
        if labels_dir is None:
            for possible_dir in ['labels', 'train/labels', '.']:
                candidate = self.dataset_path / possible_dir
                if candidate.exists():
                    labels_dir = str(candidate)
                    break
        
        images_path = Path(images_dir) if images_dir else self.dataset_path
        labels_path = Path(labels_dir) if labels_dir else self.dataset_path
        
        # find image files
        image_files = self._find_images(images_path)
        if not image_files:
            logger.error("no image files found")
            return False
        
        # filter to only labeled images
        labeled_images = []
        for img_file in image_files:
            label_file = labels_path / f"{img_file.stem}.txt"
            if label_file.exists():
                annotations = self._read_yolo_label(label_file)
                if annotations:
                    labeled_images.append(img_file)
        
        if not labeled_images:
            logger.error("no labeled images found")
            return False
        
        logger.info(f"found {len(labeled_images)} labeled images")
        
        # sample images
        sample_files = random.sample(labeled_images, min(sample_count, len(labeled_images)))
        class_names = self._load_class_names()
        
        # create output directory
        if save_images and output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta']
        
        for i, img_file in enumerate(sample_files, 1):
            logger.info(f"visualizing sample {i}: {img_file.name}")
            
            # load image
            image = cv2.imread(str(img_file))
            if image is None:
                logger.error(f"failed to load image: {img_file}")
                continue
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_height, img_width = image.shape[:2]
            
            # load annotations
            label_file = labels_path / f"{img_file.stem}.txt"
            annotations = self._read_yolo_label(label_file)
            
            # create visualization
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.imshow(image)
            ax.set_title(f'Sample {i}: {img_file.name}')
            
            # draw annotations
            for j, ann in enumerate(annotations):
                color = colors[ann['class_id'] % len(colors)]
                class_name = class_names.get(ann['class_id'], f"class_{ann['class_id']}")
                
                # convert normalized coordinates to pixels
                center_x = ann['center_x'] * img_width
                center_y = ann['center_y'] * img_height
                width = ann['width'] * img_width
                height = ann['height'] * img_height
                
                # calculate bbox corners
                x1 = center_x - width / 2
                y1 = center_y - height / 2
                x2 = center_x + width / 2
                y2 = center_y + height / 2
                
                # create rectangle
                rect = patches.Rectangle(
                    (x1, y1), width, height,
                    linewidth=2, edgecolor=color, facecolor='none'
                )
                ax.add_patch(rect)
                
                # add label
                ax.text(x1, y1 - 10, f'{class_name}',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
                       fontsize=10, color='white', weight='bold')
            
            # add info
            info_text = f'Size: {img_width}x{img_height}, Objects: {len(annotations)}'
            ax.text(10, img_height - 20, info_text,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7),
                   fontsize=10, color='white')
            
            ax.axis('off')
            plt.tight_layout()
            
            # save or show
            if save_images and output_dir:
                output_file = output_path / f'sample_{i:02d}_{img_file.stem}.png'
                plt.savefig(output_file, dpi=150, bbox_inches='tight')
                logger.info(f"saved: {output_file}")
            
            plt.show()
            
            # print details
            print(f"\nSample {i} - {img_file.name}:")
            print(f"  Image size: {img_width}x{img_height}")
            print(f"  Annotations: {len(annotations)}")
            for j, ann in enumerate(annotations):
                class_name = class_names.get(ann['class_id'], f"class_{ann['class_id']}")
                print(f"    Object {j+1}: {class_name}")
                print(f"      Center: ({ann['center_x']:.3f}, {ann['center_y']:.3f})")
                print(f"      Size: {ann['width']:.3f} x {ann['height']:.3f}")
        
        logger.success("visualization complete!")
        return True

def main():
    parser = argparse.ArgumentParser(description="unified dataset management tool")
    parser.add_argument("dataset_path", help="path to dataset directory")
    
    subparsers = parser.add_subparsers(dest="command", help="available commands")
    
    # analyze command
    analyze_parser = subparsers.add_parser("analyze", help="analyze dataset")
    analyze_parser.add_argument("--images-dir", help="images directory")
    analyze_parser.add_argument("--labels-dir", help="labels directory")
    analyze_parser.add_argument("--output", help="save analysis to JSON file")
    
    # split command
    split_parser = subparsers.add_parser("split", help="split dataset")
    split_parser.add_argument("output", help="output directory")
    split_parser.add_argument("--ratio", type=float, default=0.8, help="train ratio")
    split_parser.add_argument("--mode", choices=["simple", "enhanced"], default="simple",
                             help="split mode")
    split_parser.add_argument("--seed", type=int, default=42, help="random seed")
    split_parser.add_argument("--images-dir", help="images directory")
    split_parser.add_argument("--labels-dir", help="labels directory")
    
    # visualize command
    viz_parser = subparsers.add_parser("visualize", help="visualize samples")
    viz_parser.add_argument("--output", help="output directory for visualizations")
    viz_parser.add_argument("--samples", type=int, default=5, help="number of samples")
    viz_parser.add_argument("--images-dir", help="images directory")
    viz_parser.add_argument("--labels-dir", help="labels directory")
    viz_parser.add_argument("--no-save", action="store_true", help="don't save images")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        manager = DatasetManager(args.dataset_path)
        
        if args.command == "analyze":
            analysis = manager.analyze_dataset(args.images_dir, args.labels_dir)
            manager.print_analysis(analysis)
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(analysis, f, indent=2, default=str)
                logger.info(f"analysis saved to {args.output}")
        
        elif args.command == "split":
            success = manager.split_dataset(
                output_path=args.output,
                train_ratio=args.ratio,
                split_mode=args.mode,
                seed=args.seed,
                images_dir=args.images_dir,
                labels_dir=args.labels_dir
            )
            return 0 if success else 1
        
        elif args.command == "visualize":
            success = manager.visualize_samples(
                output_dir=args.output,
                sample_count=args.samples,
                images_dir=args.images_dir,
                labels_dir=args.labels_dir,
                save_images=not args.no_save
            )
            return 0 if success else 1
    
    except Exception as e:
        logger.error(f"command failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
