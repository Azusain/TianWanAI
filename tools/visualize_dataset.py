import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import argparse
from pathlib import Path


def read_yolo_label(label_path, img_width, img_height):
    """
    Read YOLO format label file and convert to pixel coordinates
    YOLO format: class_id center_x center_y width height (all normalized 0-1)
    """
    boxes = []
    if not os.path.exists(label_path):
        return boxes
        
    with open(label_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) != 5:
                continue
                
            class_id = int(parts[0])
            center_x = float(parts[1])
            center_y = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            # Convert normalized coordinates to pixel coordinates
            pixel_center_x = center_x * img_width
            pixel_center_y = center_y * img_height
            pixel_width = width * img_width
            pixel_height = height * img_height
            
            # Convert center format to corner format (x1, y1, x2, y2)
            x1 = pixel_center_x - pixel_width / 2
            y1 = pixel_center_y - pixel_height / 2
            x2 = pixel_center_x + pixel_width / 2
            y2 = pixel_center_y + pixel_height / 2
            
            boxes.append({
                'class_id': class_id,
                'x1': x1,
                'y1': y1, 
                'x2': x2,
                'y2': y2,
                'center_x': pixel_center_x,
                'center_y': pixel_center_y,
                'width': pixel_width,
                'height': pixel_height
            })
    
    return boxes


def load_class_names(classes_file):
    """Load class names from classes.txt file"""
    class_names = []
    if os.path.exists(classes_file):
        with open(classes_file, 'r') as f:
            class_names = [line.strip() for line in f.readlines() if line.strip()]
    
    if not class_names:
        class_names = ['class_0']  # Default class name
        
    return class_names


def visualize_sample(dataset_path, sample_count=5, split='train', save_output=False):
    """
    Visualize random samples from the dataset
    
    Args:
        dataset_path: Path to dataset root (contains images/, labels/, classes.txt)
        sample_count: Number of samples to visualize
        split: 'train' or 'val'
        save_output: Whether to save visualization images
    """
    images_path = os.path.join(dataset_path, 'images', split)
    labels_path = os.path.join(dataset_path, 'labels', split) 
    classes_file = os.path.join(dataset_path, 'classes.txt')
    
    if not os.path.exists(images_path):
        print(f"Images path not found: {images_path}")
        return
        
    if not os.path.exists(labels_path):
        print(f"Labels path not found: {labels_path}")
        return
    
    # Load class names
    class_names = load_class_names(classes_file)
    print(f"Classes: {class_names}")
    
    # Get all image files
    image_files = [f for f in os.listdir(images_path) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not image_files:
        print(f"No image files found in {images_path}")
        return
        
    print(f"Found {len(image_files)} images in {split} split")
    
    # Randomly sample images
    sample_files = random.sample(image_files, min(sample_count, len(image_files)))
    
    # Create output directory if saving
    output_dir = None
    if save_output:
        output_dir = os.path.join(dataset_path, 'visualization')
        os.makedirs(output_dir, exist_ok=True)
    
    for i, img_file in enumerate(sample_files):
        # Load image
        img_path = os.path.join(images_path, img_file)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to load image: {img_path}")
            continue
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_height, img_width = image.shape[:2]
        
        # Load corresponding label
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(labels_path, label_file)
        boxes = read_yolo_label(label_path, img_width, img_height)
        
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        ax.set_title(f'Sample {i+1}: {img_file}')
        
        # Draw bounding boxes
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta']
        
        for j, box in enumerate(boxes):
            color = colors[box['class_id'] % len(colors)]
            class_name = class_names[box['class_id']] if box['class_id'] < len(class_names) else f'class_{box["class_id"]}'
            
            # Create rectangle patch
            rect = patches.Rectangle(
                (box['x1'], box['y1']), 
                box['x2'] - box['x1'], 
                box['y2'] - box['y1'],
                linewidth=2, 
                edgecolor=color, 
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label text
            ax.text(
                box['x1'], box['y1'] - 10, 
                f'{class_name}',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
                fontsize=10,
                color='white',
                weight='bold'
            )
        
        # Add image info
        info_text = f'Image: {img_width}x{img_height}, Objects: {len(boxes)}'
        ax.text(
            10, img_height - 20,
            info_text,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7),
            fontsize=10,
            color='white'
        )
        
        ax.axis('off')
        plt.tight_layout()
        
        # Save or show
        if save_output and output_dir:
            output_file = os.path.join(output_dir, f'sample_{i+1}_{os.path.splitext(img_file)[0]}.png')
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"Saved: {output_file}")
        
        plt.show()
        
        # Print detection details
        print(f"\nSample {i+1} - {img_file}:")
        print(f"  Image size: {img_width}x{img_height}")
        print(f"  Detections: {len(boxes)}")
        
        for j, box in enumerate(boxes):
            class_name = class_names[box['class_id']] if box['class_id'] < len(class_names) else f'class_{box["class_id"]}'
            print(f"    Box {j+1}: {class_name}")
            print(f"      Center: ({box['center_x']:.1f}, {box['center_y']:.1f})")
            print(f"      Size: {box['width']:.1f}x{box['height']:.1f}")
            print(f"      Corners: ({box['x1']:.1f}, {box['y1']:.1f}) -> ({box['x2']:.1f}, {box['y2']:.1f})")
        print("-" * 50)


def analyze_dataset(dataset_path):
    """Analyze dataset statistics"""
    print(f"\nAnalyzing dataset: {dataset_path}")
    print("=" * 60)
    
    classes_file = os.path.join(dataset_path, 'classes.txt')
    class_names = load_class_names(classes_file)
    
    for split in ['train', 'val']:
        images_path = os.path.join(dataset_path, 'images', split)
        labels_path = os.path.join(dataset_path, 'labels', split)
        
        if not os.path.exists(images_path):
            continue
            
        image_files = [f for f in os.listdir(images_path) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        label_files = [f for f in os.listdir(labels_path) 
                       if f.endswith('.txt')] if os.path.exists(labels_path) else []
        
        # Count objects per class
        class_counts = {i: 0 for i in range(len(class_names))}
        total_boxes = 0
        
        for label_file in label_files:
            label_path = os.path.join(labels_path, label_file)
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        class_id = int(line.split()[0])
                        class_counts[class_id] = class_counts.get(class_id, 0) + 1
                        total_boxes += 1
        
        print(f"\n{split.upper()} Split:")
        print(f"  Images: {len(image_files)}")
        print(f"  Labels: {len(label_files)}")
        print(f"  Total objects: {total_boxes}")
        print(f"  Class distribution:")
        for class_id, count in class_counts.items():
            if count > 0:
                class_name = class_names[class_id] if class_id < len(class_names) else f'class_{class_id}'
                print(f"    {class_name}: {count} objects")


def main():
    parser = argparse.ArgumentParser(description='Visualize YOLO format dataset')
    parser.add_argument('dataset_path', help='Path to dataset root directory')
    parser.add_argument('--samples', '-s', type=int, default=3, help='Number of samples to visualize')
    parser.add_argument('--split', choices=['train', 'val'], default='train', help='Dataset split to use')
    parser.add_argument('--save', action='store_true', help='Save visualization images')
    parser.add_argument('--analyze-only', action='store_true', help='Only analyze dataset, no visualization')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset_path):
        print(f"Dataset path not found: {args.dataset_path}")
        return
    
    # Always analyze first
    analyze_dataset(args.dataset_path)
    
    # Then visualize if not analyze-only
    if not args.analyze_only:
        print(f"\nVisualizing {args.samples} samples from {args.split} split...")
        visualize_sample(args.dataset_path, args.samples, args.split, args.save)


if __name__ == '__main__':
    main()
