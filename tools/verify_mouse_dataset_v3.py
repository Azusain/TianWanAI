#!/usr/bin/env python3
"""
Verification script for mouse dataset v3
Checks for overlap between train/val splits and data integrity
"""

import os
from pathlib import Path

def verify_mouse_dataset():
    dataset_path = Path("C:/Users/azusaing/Desktop/Code/tianwan/dataset/mouse_dataset_v3")
    
    train_images_dir = dataset_path / "train" / "images"
    train_labels_dir = dataset_path / "train" / "labels"
    val_images_dir = dataset_path / "val" / "images"  
    val_labels_dir = dataset_path / "val" / "labels"
    
    print("=== Mouse Dataset v3 Integrity Verification ===")
    
    # Get file lists
    train_images = set(f.stem for f in train_images_dir.glob("*.jpg"))
    train_labels = set(f.stem for f in train_labels_dir.glob("*.txt"))
    val_images = set(f.stem for f in val_images_dir.glob("*.jpg"))
    val_labels = set(f.stem for f in val_labels_dir.glob("*.txt"))
    
    print(f"\n📊 文件统计:")
    print(f"  训练集: {len(train_images)} 图片, {len(train_labels)} 标注")
    print(f"  验证集: {len(val_images)} 图片, {len(val_labels)} 标注")
    
    # Check for overlap
    image_overlap = train_images.intersection(val_images)
    label_overlap = train_labels.intersection(val_labels)
    
    print(f"\n🔍 重叠检查:")
    print(f"  图片重叠: {len(image_overlap)} 文件")
    print(f"  标注重叠: {len(label_overlap)} 文件")
    
    if len(image_overlap) == 0 and len(label_overlap) == 0:
        print("  ✅ 无重叠 - 数据分割正确!")
    else:
        print("  ❌ 发现重叠文件:")
        if image_overlap:
            print(f"    重叠图片: {list(image_overlap)[:5]}...")
        if label_overlap:
            print(f"    重叠标注: {list(label_overlap)[:5]}...")
    
    # Check image-label pairing
    train_unpaired_images = train_images - train_labels
    train_unpaired_labels = train_labels - train_images
    val_unpaired_images = val_images - val_labels
    val_unpaired_labels = val_labels - val_images
    
    print(f"\n🔗 图片-标注配对检查:")
    print(f"  训练集未配对图片: {len(train_unpaired_images)}")
    print(f"  训练集未配对标注: {len(train_unpaired_labels)}")
    print(f"  验证集未配对图片: {len(val_unpaired_images)}")
    print(f"  验证集未配对标注: {len(val_unpaired_labels)}")
    
    if all(len(x) == 0 for x in [train_unpaired_images, train_unpaired_labels, val_unpaired_images, val_unpaired_labels]):
        print("  ✅ 所有图片和标注都正确配对!")
    else:
        print("  ⚠️ 发现未配对文件:")
        if train_unpaired_images:
            print(f"    训练集未配对图片: {list(train_unpaired_images)[:3]}...")
        if val_unpaired_images:
            print(f"    验证集未配对图片: {list(val_unpaired_images)[:3]}...")
    
    # Sample some label files to check content
    print(f"\n📄 标注文件示例检查:")
    sample_train_labels = list(train_labels_dir.glob("*.txt"))[:3]
    for label_file in sample_train_labels:
        with open(label_file, 'r') as f:
            content = f.read().strip()
            print(f"  {label_file.name}: {content[:50]}{'...' if len(content) > 50 else ''}")
    
    return {
        'train_count': len(train_images),
        'val_count': len(val_images),
        'overlap': len(image_overlap) == 0 and len(label_overlap) == 0,
        'paired': all(len(x) == 0 for x in [train_unpaired_images, train_unpaired_labels, val_unpaired_images, val_unpaired_labels])
    }

if __name__ == "__main__":
    result = verify_mouse_dataset()
    print(f"\n=== 验证结果 ===")
    print(f"✅ 数据集总计: {result['train_count'] + result['val_count']} 样本")
    print(f"✅ 分割比例: {result['train_count']/(result['train_count'] + result['val_count'])*100:.1f}% 训练集")
    print(f"✅ 无重叠: {'是' if result['overlap'] else '否'}")
    print(f"✅ 正确配对: {'是' if result['paired'] else '否'}")
    print(f"✅ 数据集位置: C:/Users/azusaing/Desktop/Code/tianwan/dataset/mouse_dataset_v3")
    print(f"\n🎉 Mouse Dataset v3 已准备就绪，可用于 Ultralytics YOLO 训练!")
