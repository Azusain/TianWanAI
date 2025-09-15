import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import yaml
import json
from PIL import Image
import logging

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class YoloAnnotation:
    class_id: int
    center_x: float
    center_y: float
    width: float
    height: float

@dataclass
class DatasetIssues:
    missing_labels: List[str]
    empty_label_files: List[str]
    invalid_annotations: List[str]

@dataclass
class DatasetAnalysis:
    total_images: int
    labeled_images: int
    unlabeled_images: int
    total_annotations: int
    class_distribution: Dict[int, int]
    class_names: Dict[int, str]
    image_sizes: Dict[str, int]
    issues: DatasetIssues

class DatasetManager:
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"dataset path does not exist: {dataset_path}")
        
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        
    def find_images(self, directory: Path) -> List[Path]:
        """find all image files in directory (non-recursive)"""
        images = []
        
        if not directory.exists():
            return images
            
        for file_path in directory.iterdir():
            if file_path.is_file():
                if file_path.suffix.lower() in self.image_extensions:
                    images.append(file_path)
        
        return sorted(images)
    
    def read_yolo_label(self, label_path: Path) -> List[YoloAnnotation]:
        """read yolo format annotation file"""
        if not label_path.exists():
            return []
            
        annotations = []
        
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
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
                        
                        # validate coordinates are in [0, 1] range
                        if not (0.0 <= center_x <= 1.0 and 0.0 <= center_y <= 1.0 and 
                               0.0 <= width <= 1.0 and 0.0 <= height <= 1.0):
                            logger.warning(f"invalid coordinates in {label_path}:{line_num}: {line}")
                            continue
                            
                        annotations.append(YoloAnnotation(
                            class_id=class_id,
                            center_x=center_x,
                            center_y=center_y,
                            width=width,
                            height=height
                        ))
                        
                    except ValueError:
                        logger.warning(f"parse error in {label_path}:{line_num}: {line}")
                        continue
                        
        except Exception as e:
            logger.error(f"failed to read label file {label_path}: {e}")
            
        return annotations
    
    def load_class_names(self) -> Dict[int, str]:
        """load class names from various config files"""
        class_names = {}
        
        # try common class name files
        for filename in ['classes.txt', 'data.yaml', 'dataset.yaml']:
            file_path = self.dataset_path / filename
            if not file_path.exists():
                continue
                
            try:
                if filename.endswith(('.yaml', '.yml')):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)
                        if 'names' in data:
                            names = data['names']
                            if isinstance(names, dict):
                                # names as dict: {0: 'class1', 1: 'class2'}
                                for key, value in names.items():
                                    class_names[int(key)] = str(value)
                            elif isinstance(names, list):
                                # names as list: ['class1', 'class2']
                                for i, name in enumerate(names):
                                    class_names[i] = str(name)
                            return class_names
                else:
                    # classes.txt format: one class per line
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for i, line in enumerate(f):
                            line = line.strip()
                            if line:
                                class_names[i] = line
                        return class_names
                        
            except Exception as e:
                logger.warning(f"failed to read class names from {file_path}: {e}")
                continue
                
        return class_names
    
    @staticmethod
    def merge_yolo_datasets(sources: list, output_dir: str) -> str:
        """merge multiple standard YOLO-format datasets into a new dataset.
        expected structure for each source: <root>/(train|val)/(images|labels)
        output structure: <root>/(train|val)/(images|labels) + data.yaml
        - merges train and val splits from all source datasets
        - on filename conflicts, renames files with suffix to avoid overwrites
        - creates data.yaml config file for the merged dataset
        - preserves class information from source datasets
        returns a human-readable summary
        
        Args:
            sources: list of source dataset directory paths
            output_dir: output directory path
        """
        if not sources or len(sources) < 2:
            raise ValueError("at least 2 source datasets are required for merging")
        from pathlib import Path
        import shutil
        import yaml
        import random
        
        def find_dataset_structure(root: Path) -> dict:
            """detect dataset structure and return available splits"""
            structure = {}
            # check for standard YOLO structure
            for split in ['train', 'val', 'test']:
                split_dir = root / split
                if split_dir.exists():
                    images_dir = split_dir / 'images'
                    labels_dir = split_dir / 'labels'
                    if images_dir.exists() and labels_dir.exists():
                        structure[split] = {'images': images_dir, 'labels': labels_dir}
            
            # fallback: check for direct images/labels structure
            if not structure:
                images_dir = root / 'images'
                labels_dir = root / 'labels'
                if images_dir.exists() and labels_dir.exists():
                    structure['train'] = {'images': images_dir, 'labels': labels_dir}
            
            return structure
        
        
        def next_available_name(dest_dir: Path, base_name: str, suffix_tag: str) -> str:
            """generate unique filename to avoid conflicts"""
            stem = Path(base_name).stem
            ext = Path(base_name).suffix
            candidate = f"{stem}{ext}"
            idx = 1
            while (dest_dir / candidate).exists():
                candidate = f"{stem}_{suffix_tag}{idx}{ext}"
                idx += 1
            return candidate
        
        def copy_image_label_pair(img_path: Path, src_labels_dir: Path, 
                                dest_images_dir: Path, dest_labels_dir: Path, 
                                tag: str) -> Tuple[bool, bool]:
            """copy image and its corresponding label file"""
            nonlocal total_images, total_labels
            
            # determine target filename
            target_name = img_path.name
            if (dest_images_dir / target_name).exists():
                target_name = next_available_name(dest_images_dir, img_path.name, tag)
            
            # copy image
            dest_img = dest_images_dir / target_name
            shutil.copy2(img_path, dest_img)
            total_images += 1
            
            # copy corresponding label if exists
            base_stem = img_path.stem
            src_label_path = src_labels_dir / f"{base_stem}.txt"
            copied_label = False
            
            if src_label_path.exists():
                target_stem = Path(target_name).stem
                dest_label = dest_labels_dir / f"{target_stem}.txt"
                shutil.copy2(src_label_path, dest_label)
                total_labels += 1
                copied_label = True
            
            return True, copied_label
        
        # initialize counters
        total_images = 0
        total_labels = 0
        
        # setup output path
        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        
        # convert sources to Path objects and detect structures
        source_paths = [Path(src) for src in sources]
        source_structures = []
        
        for i, src_path in enumerate(source_paths):
            if not src_path.exists():
                logger.warning(f"source dataset {i} does not exist: {src_path}")
                continue
                
            structure = find_dataset_structure(src_path)
            if not structure:
                logger.warning(f"no valid YOLO dataset structure found in source {i}: {src_path}")
                continue
                
            source_structures.append({
                'path': src_path,
                'structure': structure,
                'index': i
            })
            logger.info(f"source {i} ({src_path.name}): {list(structure.keys())}")
        
        if len(source_structures) < 2:
            raise Exception(f"need at least 2 valid datasets, found {len(source_structures)}")
        
        # get all available splits from all datasets
        all_splits = set()
        for src_info in source_structures:
            all_splits.update(src_info['structure'].keys())
        
        split_stats = {}
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        
        # process each split
        for split in all_splits:
            logger.info(f"processing {split} split...")
            
            # create output directories for this split
            out_split_images = out_root / split / 'images'
            out_split_labels = out_root / split / 'labels'
            out_split_images.mkdir(parents=True, exist_ok=True)
            out_split_labels.mkdir(parents=True, exist_ok=True)
            
            split_images = 0
            split_labels = 0
            
            # process each source dataset that has this split
            for src_info in source_structures:
                src_structure = src_info['structure']
                src_index = src_info['index']
                
                if split not in src_structure:
                    continue
                    
                src_images_dir = src_structure[split]['images']
                src_labels_dir = src_structure[split]['labels']
                tag = f"src{src_index}"
                
                logger.info(f"  processing source {src_index} for {split} split...")
                
                for img_file in sorted(src_images_dir.iterdir()):
                    if img_file.is_file() and img_file.suffix.lower() in image_exts:
                        copied_img, copied_lbl = copy_image_label_pair(
                            img_file, src_labels_dir, out_split_images, out_split_labels, tag
                        )
                        if copied_img:
                            split_images += 1
                        if copied_lbl:
                            split_labels += 1
            
            split_stats[split] = {'images': split_images, 'labels': split_labels}
            logger.info(f"{split} split: {split_images} images, {split_labels} labels")
        
        # create data.yaml configuration
        data_yaml = {
            'path': str(out_root.absolute()),
            'train': 'train/images' if 'train' in all_splits else None,
            'val': 'val/images' if 'val' in all_splits else None,
            'test': 'test/images' if 'test' in all_splits else None,
        }
        
        # remove None entries
        data_yaml = {k: v for k, v in data_yaml.items() if v is not None}
        
        # write data.yaml
        with open(out_root / 'data.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)
        
        # create summary
        summary_lines = [
            f"YOLO dataset merge completed:",
            f"- merged {len(source_structures)} datasets:"
        ]
        
        # add source dataset info
        for src_info in source_structures:
            summary_lines.append(f"  • source {src_info['index']}: {src_info['path']} ({', '.join(src_info['structure'].keys())})")
        
        summary_lines.extend([
            f"- output: {out_root}",
            f"- total images: {total_images}",
            f"- total labels: {total_labels}",
            f"- splits merged: {', '.join(all_splits)}"
        ])
        
        # add per-split statistics
        for split, stats in split_stats.items():
            summary_lines.append(f"  - {split}: {stats['images']} images, {stats['labels']} labels")
        
        summary = "\n".join(summary_lines)
        logger.info(summary)
        return summary
    
    def analyze_dataset(self, images_dir: Optional[str] = None, labels_dir: Optional[str] = None) -> DatasetAnalysis:
        """analyze dataset and return comprehensive statistics"""
        logger.info("starting dataset analysis...")
        
        # determine directories
        if images_dir:
            images_path = Path(images_dir)
        else:
            # auto-detect images directory
            images_path = self.dataset_path
            for possible_dir in ['images', 'train/images', 'val/images']:
                candidate = self.dataset_path / possible_dir
                if candidate.exists() and self.find_images(candidate):
                    images_path = candidate
                    break
        
        if labels_dir:
            labels_path = Path(labels_dir)
        else:
            # auto-detect labels directory
            labels_path = self.dataset_path
            for possible_dir in ['labels', 'train/labels', 'val/labels']:
                candidate = self.dataset_path / possible_dir
                if candidate.exists():
                    labels_path = candidate
                    break
        
        logger.info(f"analyzing images in: {images_path}")
        logger.info(f"analyzing labels in: {labels_path}")
        
        # find all images
        image_files = self.find_images(images_path)
        logger.info(f"found {len(image_files)} image files")
        
        if not image_files:
            raise ValueError("no image files found")
        
        # initialize analysis
        issues = DatasetIssues([], [], [])
        analysis = DatasetAnalysis(
            total_images=len(image_files),
            labeled_images=0,
            unlabeled_images=0,
            total_annotations=0,
            class_distribution={},
            class_names={},
            image_sizes={},
            issues=issues
        )
        
        # analyze each image
        for img_file in image_files:
            img_stem = img_file.stem
            label_file = labels_path / f"{img_stem}.txt"
            
            # check if label exists
            if not label_file.exists():
                analysis.unlabeled_images += 1
                issues.missing_labels.append(str(img_file))
                continue
            
            # read annotations
            try:
                annotations = self.read_yolo_label(label_file)
            except Exception as e:
                logger.warning(f"failed to read label file {label_file}: {e}")
                issues.invalid_annotations.append(str(label_file))
                continue
            
            if not annotations:
                analysis.unlabeled_images += 1
                issues.empty_label_files.append(str(label_file))
            else:
                analysis.labeled_images += 1
                analysis.total_annotations += len(annotations)
                
                # count class occurrences
                for ann in annotations:
                    analysis.class_distribution[ann.class_id] = (
                        analysis.class_distribution.get(ann.class_id, 0) + 1
                    )
            
            # get image dimensions
            try:
                with Image.open(img_file) as img:
                    width, height = img.size
                    size_key = f"{width}x{height}"
                    analysis.image_sizes[size_key] = analysis.image_sizes.get(size_key, 0) + 1
            except Exception:
                # fallback for corrupted images
                ext = img_file.suffix.lower()
                size_key = f"corrupted{ext}"
                analysis.image_sizes[size_key] = analysis.image_sizes.get(size_key, 0) + 1
        
        # load class names
        analysis.class_names = self.load_class_names()
        
        logger.info("dataset analysis complete!")
        logger.info(f"total images: {analysis.total_images}")
        logger.info(f"labeled images: {analysis.labeled_images}")
        logger.info(f"unlabeled images: {analysis.unlabeled_images}")
        logger.info(f"total annotations: {analysis.total_annotations}")
        
        return analysis
    
    def split_dataset(self, output_path: str, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15,
                     split_mode: str = "random", seed: int = 42,
                     images_dir: Optional[str] = None, labels_dir: Optional[str] = None) -> str:
        """split dataset into train/val/test sets
        
        Args:
            output_path: output directory path
            train_ratio: ratio for training set (default: 0.7)
            val_ratio: ratio for validation set (default: 0.15)
            test_ratio: ratio for test set (default: 0.15)
            split_mode: splitting mode ('random' or 'sequential')
            seed: random seed for reproducible splits
            images_dir: custom images directory
            labels_dir: custom labels directory
        """
        # validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 0.001:
            raise ValueError(f"ratios must sum to 1.0, got {total_ratio:.3f}")
        
        logger.info(f"starting dataset split with ratios - train: {train_ratio}, val: {val_ratio}, test: {test_ratio}")
        
        # determine source directories
        if images_dir:
            source_images = Path(images_dir)
        else:
            source_images = self.dataset_path / "images"
            if not source_images.exists():
                source_images = self.dataset_path
        
        if labels_dir:
            source_labels = Path(labels_dir)
        else:
            source_labels = self.dataset_path / "labels"
            if not source_labels.exists():
                source_labels = self.dataset_path
        
        # find all images
        image_files = self.find_images(source_images)
        if not image_files:
            raise ValueError("no image files found")
        
        # set random seed
        random.seed(seed)
        
        # shuffle images
        if split_mode == "random":
            random.shuffle(image_files)
        
        # calculate split points
        total_files = len(image_files)
        train_end = int(total_files * train_ratio)
        val_end = train_end + int(total_files * val_ratio)
        
        train_files = image_files[:train_end]
        val_files = image_files[train_end:val_end]
        test_files = image_files[val_end:]
        
        # create output directories
        output_dir = Path(output_path)
        train_img_dir = output_dir / "train" / "images"
        train_label_dir = output_dir / "train" / "labels"
        val_img_dir = output_dir / "val" / "images"
        val_label_dir = output_dir / "val" / "labels"
        test_img_dir = output_dir / "test" / "images"
        test_label_dir = output_dir / "test" / "labels"
        
        for dir_path in [train_img_dir, train_label_dir, val_img_dir, val_label_dir, test_img_dir, test_label_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # copy files
        def copy_files(file_list: List[Path], img_dest: Path, label_dest: Path):
            copied_images = 0
            copied_labels = 0
            
            for img_file in file_list:
                # copy image
                dest_img = img_dest / img_file.name
                shutil.copy2(img_file, dest_img)
                copied_images += 1
                
                # copy corresponding label if exists
                label_file = source_labels / f"{img_file.stem}.txt"
                if label_file.exists():
                    dest_label = label_dest / f"{img_file.stem}.txt"
                    shutil.copy2(label_file, dest_label)
                    copied_labels += 1
            
            return copied_images, copied_labels
        
        train_img_count, train_label_count = copy_files(train_files, train_img_dir, train_label_dir)
        val_img_count, val_label_count = copy_files(val_files, val_img_dir, val_label_dir)
        test_img_count, test_label_count = copy_files(test_files, test_img_dir, test_label_dir)
        
        # create data.yaml file
        data_yaml = {
            'path': str(output_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'names': self.load_class_names()
        }
        
        with open(output_dir / "data.yaml", 'w', encoding='utf-8') as f:
            yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)
        
        summary = (
            f"dataset split completed:\n"
            f"- output: {output_path}\n"
            f"- train ratio: {train_ratio:.1%} ({train_img_count} images, {train_label_count} labels)\n"
            f"- val ratio: {val_ratio:.1%} ({val_img_count} images, {val_label_count} labels)\n"
            f"- test ratio: {test_ratio:.1%} ({test_img_count} images, {test_label_count} labels)\n"
            f"- mode: {split_mode}\n"
            f"- seed: {seed}"
        )
        
        logger.info(summary)
        return summary
    
    def visualize_samples(self, output_dir: Optional[str] = None, sample_count: int = 5,
                         images_dir: Optional[str] = None, labels_dir: Optional[str] = None) -> List[str]:
        """generate visualization samples with bounding boxes"""
        logger.info(f"generating {sample_count} visualization samples")
        
        # determine source directories
        if images_dir:
            source_images = Path(images_dir)
        else:
            source_images = self.dataset_path / "images"
            if not source_images.exists():
                source_images = self.dataset_path
        
        if labels_dir:
            source_labels = Path(labels_dir)
        else:
            source_labels = self.dataset_path / "labels"
            if not source_labels.exists():
                source_labels = self.dataset_path
        
        # find labeled images
        image_files = self.find_images(source_images)
        labeled_images = []
        
        for img_file in image_files:
            label_file = source_labels / f"{img_file.stem}.txt"
            if label_file.exists():
                annotations = self.read_yolo_label(label_file)
                if annotations:
                    labeled_images.append((img_file, annotations))
        
        if not labeled_images:
            return []
        
        # select random samples
        random.shuffle(labeled_images)
        samples = labeled_images[:min(sample_count, len(labeled_images))]
        
        # determine output directory
        if output_dir:
            viz_dir = Path(output_dir)
        else:
            viz_dir = self.dataset_path / "visualizations"
        
        viz_dir.mkdir(exist_ok=True)
        
        # generate visualizations with bounding boxes
        output_files = []
        class_names = self.load_class_names()
        
        # colors for different classes (BGR format)
        colors = [
            (255, 0, 0),    # blue
            (0, 255, 0),    # green
            (0, 0, 255),    # red
            (255, 255, 0),  # cyan
            (255, 0, 255),  # magenta
            (0, 255, 255),  # yellow
            (128, 0, 255),  # purple
            (255, 128, 0),  # orange
            (0, 128, 255),  # light blue
            (128, 255, 0),  # lime
        ]
        
        try:
            import cv2
        except ImportError:
            logger.warning("opencv not available, creating copies without bounding boxes")
            # fallback to simple copy
            for i, (img_file, annotations) in enumerate(samples, 1):
                output_file = viz_dir / f"sample_{i:02d}.png"
                shutil.copy2(img_file, output_file)
                output_files.append(str(output_file))
            return output_files
        
        for i, (img_file, annotations) in enumerate(samples, 1):
            output_file = viz_dir / f"sample_{i:02d}.png"
            
            try:
                # load image
                image = cv2.imread(str(img_file))
                if image is None:
                    logger.warning(f"failed to load image: {img_file}")
                    continue
                
                height, width = image.shape[:2]
                
                # draw bounding boxes
                for ann in annotations:
                    # convert yolo coordinates to pixel coordinates
                    center_x_px = int(ann.center_x * width)
                    center_y_px = int(ann.center_y * height)
                    box_width_px = int(ann.width * width)
                    box_height_px = int(ann.height * height)
                    
                    # calculate corner coordinates
                    x1 = center_x_px - box_width_px // 2
                    y1 = center_y_px - box_height_px // 2
                    x2 = center_x_px + box_width_px // 2
                    y2 = center_y_px + box_height_px // 2
                    
                    # get color for this class
                    color = colors[ann.class_id % len(colors)]
                    
                    # draw rectangle
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    
                    # draw class name
                    class_name = class_names.get(ann.class_id, f"class_{ann.class_id}")
                    label_text = f"{class_name} ({ann.class_id})"
                    
                    # calculate text size for background
                    font_scale = 0.6
                    font_thickness = 1
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
                    )
                    
                    # draw background rectangle for text
                    text_y = max(y1 - 5, text_height + 5)
                    cv2.rectangle(
                        image, 
                        (x1, text_y - text_height - baseline), 
                        (x1 + text_width, text_y + baseline),
                        color, -1
                    )
                    
                    # draw text
                    cv2.putText(
                        image, label_text, (x1, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness
                    )
                
                # save image
                cv2.imwrite(str(output_file), image)
                output_files.append(str(output_file))
                
                # log annotation info
                logger.info(f"sample {i}: {img_file.name} with {len(annotations)} annotations")
                for ann in annotations:
                    class_name = class_names.get(ann.class_id, f"class_{ann.class_id}")
                    logger.info(f"  - {class_name}: ({ann.center_x:.3f}, {ann.center_y:.3f}) {ann.width:.3f}x{ann.height:.3f}")
                    
            except Exception as e:
                logger.error(f"failed to process {img_file}: {e}")
                continue
        
        return output_files
