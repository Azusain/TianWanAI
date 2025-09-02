#!/usr/bin/env python3
"""
Dataset splitting script for mouse3 dataset
Organizes dataset into Ultralytics YOLO format with train/val splits
"""

import os
import shutil
import random
from pathlib import Path

def split_mouse_dataset():
    # Paths
    source_path = Path("C:/Users/azusaing/Desktop/tianwan_dataset/mouse3_raw_frames")
    target_path = Path("C:/Users/azusaing/Desktop/Code/tianwan/dataset/mouse_dataset_v3")
    
    source_images = source_path / "images"
    source_labels = source_path / "labels"
    
    print("=== Mouse Dataset v3 Splitting ===")
    print(f"源数据集: {source_path}")
    print(f"目标路径: {target_path}")
    
    # Check source exists
    if not source_path.exists():
        print(f"❌ 源路径不存在: {source_path}")
        return False
        
    if not source_images.exists() or not source_labels.exists():
        print("❌ 源数据集缺少 images 或 labels 文件夹")
        return False
    
    # Get all image files
    image_files = list(source_images.glob("*.jpg"))
    print(f"📸 找到 {len(image_files)} 张图片")
    
    # Filter to only images that have corresponding labels
    valid_pairs = []
    for img_file in image_files:
        label_file = source_labels / f"{img_file.stem}.txt"
        if label_file.exists():
            valid_pairs.append((img_file, label_file))
    
    print(f"✅ 有效的图片-标注对: {len(valid_pairs)}")
    
    if len(valid_pairs) == 0:
        print("❌ 没有找到有效的图片-标注对")
        return False
    
    # Shuffle for random split
    random.seed(42)  # For reproducible results
    random.shuffle(valid_pairs)
    
    # Calculate split indices
    total_samples = len(valid_pairs)
    train_size = int(total_samples * 0.8)
    
    train_pairs = valid_pairs[:train_size]
    val_pairs = valid_pairs[train_size:]
    
    print(f"📊 数据划分:")
    print(f"  训练集: {len(train_pairs)} 样本 ({len(train_pairs)/total_samples*100:.1f}%)")
    print(f"  验证集: {len(val_pairs)} 样本 ({len(val_pairs)/total_samples*100:.1f}%)")
    
    # Create target directory structure
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Create YOLO directory structure
    train_images_dir = target_path / "train" / "images"
    train_labels_dir = target_path / "train" / "labels" 
    val_images_dir = target_path / "val" / "images"
    val_labels_dir = target_path / "val" / "labels"
    
    for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 创建目录结构完成")
    
    # Copy training files
    print("📋 复制训练集文件...")
    for img_file, label_file in train_pairs:
        # Copy image
        shutil.copy2(img_file, train_images_dir / img_file.name)
        # Copy label
        shutil.copy2(label_file, train_labels_dir / label_file.name)
    
    # Copy validation files
    print("📋 复制验证集文件...")
    for img_file, label_file in val_pairs:
        # Copy image
        shutil.copy2(img_file, val_images_dir / img_file.name)
        # Copy label
        shutil.copy2(label_file, val_labels_dir / label_file.name)
    
    # Copy classes.txt if it exists
    classes_file = source_labels / "classes.txt"
    if classes_file.exists():
        shutil.copy2(classes_file, target_path / "classes.txt")
        print("📄 复制 classes.txt 文件")
    
    # Create data.yaml for Ultralytics
    data_yaml_content = f"""# Mouse Dataset v3 Configuration
path: {target_path.as_posix()}  # dataset root dir
train: train/images  # train images (relative to 'path')
val: val/images  # val images (relative to 'path')

# Classes
names:
  0: mouse  # assuming mouse detection task
"""
    
    with open(target_path / "data.yaml", "w", encoding="utf-8") as f:
        f.write(data_yaml_content)
    
    print("📄 创建 data.yaml 配置文件")
    
    # Final verification
    train_img_count = len(list(train_images_dir.glob("*.jpg")))
    train_lbl_count = len(list(train_labels_dir.glob("*.txt")))
    val_img_count = len(list(val_images_dir.glob("*.jpg")))
    val_lbl_count = len(list(val_labels_dir.glob("*.txt")))
    
    print(f"\n=== 最终结果验证 ===")
    print(f"✅ 训练集: {train_img_count} 图片, {train_lbl_count} 标注")
    print(f"✅ 验证集: {val_img_count} 图片, {val_lbl_count} 标注")
    print(f"✅ 总计: {train_img_count + val_img_count} 图片, {train_lbl_count + val_lbl_count} 标注")
    print(f"✅ 数据集保存到: {target_path}")
    
    return True

if __name__ == "__main__":
    success = split_mouse_dataset()
    if success:
        print("\n🎉 数据集划分和组织完成!")
    else:
        print("\n❌ 数据集处理失败")
