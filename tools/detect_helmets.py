#!/usr/bin/env python3

import os
import sys
from pathlib import Path
from ultralytics import YOLO
import cv2

def main():
    if len(sys.argv) != 2:
        print("usage: python detect_helmets.py <dataset_path>")
        sys.exit(1)
    
    dataset_path = Path(sys.argv[1])
    if not dataset_path.exists():
        print(f"dataset path not found: {dataset_path}")
        sys.exit(1)
    
    # load model from same directory
    script_dir = Path(__file__).parent
    model_path = script_dir / "helmet_v1.pt"
    
    if not model_path.exists():
        print(f"model file not found: {model_path}")
        sys.exit(1)
    
    print(f"loading model: {model_path}")
    model = YOLO(str(model_path))
    
    # check if images folder exists (YOLO dataset structure)
    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"
    
    if images_dir.exists():
        # YOLO dataset structure
        image_search_path = images_dir
        use_labels_dir = True
        if not labels_dir.exists():
            labels_dir.mkdir()
    else:
        # flat structure
        image_search_path = dataset_path
        use_labels_dir = False
    
    # get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(image_search_path.glob(f"*{ext}"))
        image_files.extend(image_search_path.glob(f"*{ext.upper()}"))
    
    print(f"processing {len(image_files)} images...")
    
    updated_count = 0
    for i, image_path in enumerate(image_files):
        # run detection
        results = model(str(image_path), conf=0.5, verbose=False)
        
        # get label file path
        if use_labels_dir:
            label_path = labels_dir / (image_path.stem + '.txt')
        else:
            label_path = image_path.with_suffix('.txt')
        
        # read existing labels (non-helmet)
        existing_labels = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                existing_labels = [line.strip() for line in f.readlines() 
                                 if line.strip() and not line.startswith('0 ')]
        
        # process detections
        new_labels = existing_labels.copy()
        helmet_count = 0
        
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                img = cv2.imread(str(image_path))
                if img is not None:
                    img_height, img_width = img.shape[:2]
                    
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # convert to yolo format
                        x_center = (x1 + x2) / 2.0 / img_width
                        y_center = (y1 + y2) / 2.0 / img_height
                        width = (x2 - x1) / img_width
                        height = (y2 - y1) / img_height
                        
                        label_line = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                        new_labels.append(label_line)
                        helmet_count += 1
        
        # write labels
        with open(label_path, 'w') as f:
            for label in new_labels:
                f.write(f"{label}\n")
        
        if helmet_count > 0:
            updated_count += 1
            print(f"[{i+1}/{len(image_files)}] {image_path.name}: {helmet_count} helmets")
        elif (i + 1) % 100 == 0:
            print(f"[{i+1}/{len(image_files)}] processed...")
    
    print(f"completed: {updated_count} images updated with helmet detections")

if __name__ == '__main__':
    main()
