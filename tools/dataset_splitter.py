#!/usr/bin/env python3
"""
dataset splitter - split dataset into train and validation sets
"""

import os
import shutil
import random
import argparse
from pathlib import Path
from loguru import logger


class DatasetSplitter:
    def __init__(self, dataset_path: str, train_ratio: float = 0.8):
        """
        initialize dataset splitter
        
        args:
            dataset_path: path to dataset directory
            train_ratio: ratio for training set (default: 0.8 for 4:1 split)
        """
        self.dataset_path = Path(dataset_path)
        self.train_ratio = train_ratio
        self.val_ratio = 1.0 - train_ratio
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"dataset path {dataset_path} does not exist")
    
    def split_class_folder(self, class_name: str, output_base_path: str):
        """
        split single class folder into train and validation sets
        
        args:
            class_name: name of the class folder (e.g., 'tshirt', 'other')
            output_base_path: base path for output directories
        """
        class_folder = self.dataset_path / class_name
        if not class_folder.exists():
            logger.warning(f"class folder {class_folder} does not exist, skipping")
            return
        
        # get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = [f for f in class_folder.iterdir() 
                      if f.is_file() and f.suffix.lower() in image_extensions]
        
        if not image_files:
            logger.warning(f"no image files found in {class_folder}")
            return
        
        # shuffle files randomly
        random.shuffle(image_files)
        
        # calculate split point
        total_files = len(image_files)
        train_count = int(total_files * self.train_ratio)
        
        train_files = image_files[:train_count]
        val_files = image_files[train_count:]
        
        logger.info(f"splitting {class_name}: {len(train_files)} train, {len(val_files)} val")
        
        # create output directories
        output_base = Path(output_base_path)
        train_dir = output_base / class_name
        val_dir = output_base / f"{class_name}_val"
        
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        
        # copy train files
        for file_path in train_files:
            dst_path = train_dir / file_path.name
            shutil.copy2(file_path, dst_path)
        
        # copy validation files
        for file_path in val_files:
            dst_path = val_dir / file_path.name
            shutil.copy2(file_path, dst_path)
        
        logger.success(f"completed {class_name} split: {len(train_files)} -> {train_dir}, {len(val_files)} -> {val_dir}")
    
    def split_dataset(self, output_path: str, classes: list = None):
        """
        split entire dataset
        
        args:
            output_path: path for output dataset
            classes: list of class names to split (if None, split all subdirectories)
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if classes is None:
            # auto-detect classes from subdirectories
            classes = [d.name for d in self.dataset_path.iterdir() if d.is_dir()]
        
        logger.info(f"splitting dataset from {self.dataset_path} to {output_path}")
        logger.info(f"classes to split: {classes}")
        logger.info(f"split ratio - train: {self.train_ratio:.1%}, val: {self.val_ratio:.1%}")
        
        for class_name in classes:
            try:
                self.split_class_folder(class_name, output_path)
            except Exception as e:
                logger.error(f"failed to split class {class_name}: {e}")
        
        logger.success("dataset splitting completed!")


def main():
    parser = argparse.ArgumentParser(description="split dataset into train and validation sets")
    parser.add_argument("input", help="input dataset path")
    parser.add_argument("-o", "--output", required=True, help="output path for split dataset")
    parser.add_argument("-r", "--ratio", type=float, default=0.8, 
                       help="train ratio (default: 0.8 for 4:1 split)")
    parser.add_argument("--classes", nargs="+", 
                       help="specific classes to split (default: all subdirectories)")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    
    args = parser.parse_args()
    
    # set random seed
    random.seed(args.seed)
    
    try:
        splitter = DatasetSplitter(args.input, args.ratio)
        splitter.split_dataset(args.output, args.classes)
    except Exception as e:
        logger.error(f"dataset splitting failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
