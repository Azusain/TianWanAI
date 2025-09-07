#!/usr/bin/env python3
"""
Enhanced dataset splitting script for mouse dataset
Organizes dataset into Ultralytics YOLO format with train/val splits
Includes both labeled and unlabeled images for comprehensive training
"""

import os
import shutil
import random
from pathlib import Path

def split_mouse_dataset_enhanced():
    # paths
    source_path = Path("C:\\Users\\azusaing\\Desktop\\Code\\tianwan\\dataset\\mouse_dataset_v5")
    target_path = Path("C:\\Users\\azusaing\\Desktop\\Code\\tianwan\\dataset\\mouse_dataset_v5_split_enhanced")
    
    source_images = source_path / "images"
    source_labels = source_path / "labels"
    
    print("=== Enhanced Mouse Dataset Splitting ===")
    print(f"source dataset: {source_path}")
    print(f"target path: {target_path}")
    
    # check source exists
    if not source_path.exists():
        print(f"❌ source path does not exist: {source_path}")
        return False
        
    if not source_images.exists():
        print(f"❌ source images folder missing: {source_images}")
        return False
    
    # get all image files
    image_files = list(source_images.glob("*.jpg"))
    print(f"📸 found {len(image_files)} images total")
    
    # separate labeled and unlabeled images
    labeled_pairs = []
    unlabeled_images = []
    
    for img_file in image_files:
        if source_labels.exists():
            label_file = source_labels / f"{img_file.stem}.txt"
            if label_file.exists() and label_file.stat().st_size > 0:  # check file exists and not empty
                labeled_pairs.append((img_file, label_file))
            else:
                unlabeled_images.append(img_file)
        else:
            unlabeled_images.append(img_file)
    
    print(f"✅ labeled image-annotation pairs: {len(labeled_pairs)}")
    print(f"📋 unlabeled images: {len(unlabeled_images)}")
    
    if len(labeled_pairs) == 0 and len(unlabeled_images) == 0:
        print("❌ no valid images found")
        return False
    
    # shuffle for random split
    random.seed(42)  # for reproducible results
    random.shuffle(labeled_pairs)
    random.shuffle(unlabeled_images)
    
    # calculate split indices for labeled data
    labeled_total = len(labeled_pairs)
    labeled_train_size = int(labeled_total * 0.8) if labeled_total > 0 else 0
    
    labeled_train_pairs = labeled_pairs[:labeled_train_size]
    labeled_val_pairs = labeled_pairs[labeled_train_size:]
    
    # calculate split indices for unlabeled data
    unlabeled_total = len(unlabeled_images)
    unlabeled_train_size = int(unlabeled_total * 0.8) if unlabeled_total > 0 else 0
    
    unlabeled_train_images = unlabeled_images[:unlabeled_train_size]
    unlabeled_val_images = unlabeled_images[unlabeled_train_size:]
    
    print(f"📊 data split:")
    print(f"  labeled training: {len(labeled_train_pairs)} samples")
    print(f"  labeled validation: {len(labeled_val_pairs)} samples")
    print(f"  unlabeled training: {len(unlabeled_train_images)} samples")  
    print(f"  unlabeled validation: {len(unlabeled_val_images)} samples")
    print(f"  total training: {len(labeled_train_pairs) + len(unlabeled_train_images)} samples")
    print(f"  total validation: {len(labeled_val_pairs) + len(unlabeled_val_images)} samples")
    
    # create target directory structure
    target_path.mkdir(parents=True, exist_ok=True)
    
    # create YOLO directory structure with unlabeled subdirs
    train_images_dir = target_path / "train" / "images"
    train_labels_dir = target_path / "train" / "labels" 
    train_unlabeled_dir = target_path / "train" / "unlabeled"
    
    val_images_dir = target_path / "val" / "images"
    val_labels_dir = target_path / "val" / "labels"
    val_unlabeled_dir = target_path / "val" / "unlabeled"
    
    for dir_path in [train_images_dir, train_labels_dir, train_unlabeled_dir, 
                     val_images_dir, val_labels_dir, val_unlabeled_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 directory structure created")
    
    # copy labeled training files
    if labeled_train_pairs:
        print("📋 copying labeled training files...")
        for img_file, label_file in labeled_train_pairs:
            # copy image
            shutil.copy2(img_file, train_images_dir / img_file.name)
            # copy label
            shutil.copy2(label_file, train_labels_dir / label_file.name)
    
    # copy labeled validation files
    if labeled_val_pairs:
        print("📋 copying labeled validation files...")
        for img_file, label_file in labeled_val_pairs:
            # copy image
            shutil.copy2(img_file, val_images_dir / img_file.name)
            # copy label
            shutil.copy2(label_file, val_labels_dir / label_file.name)
    
    # copy unlabeled training files
    if unlabeled_train_images:
        print("📋 copying unlabeled training files...")
        for img_file in unlabeled_train_images:
            shutil.copy2(img_file, train_unlabeled_dir / img_file.name)
            # create empty label file for consistency
            empty_label = train_labels_dir / f"{img_file.stem}.txt"
            empty_label.touch()
    
    # copy unlabeled validation files
    if unlabeled_val_images:
        print("📋 copying unlabeled validation files...")
        for img_file in unlabeled_val_images:
            shutil.copy2(img_file, val_unlabeled_dir / img_file.name)
            # create empty label file for consistency
            empty_label = val_labels_dir / f"{img_file.stem}.txt"
            empty_label.touch()
    
    # copy classes.txt if it exists
    if source_labels.exists():
        classes_file = source_labels / "classes.txt"
        if classes_file.exists():
            shutil.copy2(classes_file, target_path / "classes.txt")
            print("📄 copied classes.txt file")
    
    # create data.yaml for ultralytics
    data_yaml_content = f"""# enhanced mouse dataset configuration
path: {target_path.as_posix()}  # dataset root dir
train: train/images  # train images (relative to 'path')
val: val/images  # val images (relative to 'path')

# unlabeled data paths for self-supervised learning
train_unlabeled: train/unlabeled  # unlabeled training images
val_unlabeled: val/unlabeled  # unlabeled validation images

# classes
names:
  0: mouse  # assuming mouse detection task

# dataset statistics
labeled_train_samples: {len(labeled_train_pairs)}
labeled_val_samples: {len(labeled_val_pairs)}
unlabeled_train_samples: {len(unlabeled_train_images)}
unlabeled_val_samples: {len(unlabeled_val_images)}
total_samples: {len(image_files)}
"""
    
    with open(target_path / "data.yaml", "w", encoding="utf-8") as f:
        f.write(data_yaml_content)
    
    print("📄 created data.yaml configuration file")
    
    # create README with usage instructions
    readme_content = f"""# Enhanced Mouse Dataset v5

## dataset structure
```
{target_path.name}/
├── train/
│   ├── images/          # labeled training images
│   ├── labels/          # corresponding label files (including empty ones for unlabeled)
│   └── unlabeled/       # unlabeled training images
├── val/
│   ├── images/          # labeled validation images  
│   ├── labels/          # corresponding label files (including empty ones for unlabeled)
│   └── unlabeled/       # unlabeled validation images
├── data.yaml            # ultralytics configuration
├── classes.txt          # class definitions (if available)
└── README.md           # this file

## dataset statistics
- labeled training samples: {len(labeled_train_pairs)}
- labeled validation samples: {len(labeled_val_pairs)} 
- unlabeled training samples: {len(unlabeled_train_images)}
- unlabeled validation samples: {len(unlabeled_val_images)}
- total samples: {len(image_files)}

## usage notes
1. unlabeled images are stored separately in `unlabeled/` folders
2. empty label files are created for unlabeled images to maintain consistency
3. you can use unlabeled data for:
   - self-supervised learning
   - pseudo-labeling
   - data augmentation
   - background sampling

## training with unlabeled data
you can leverage unlabeled data in various ways:
- use them for background/negative sampling
- apply pseudo-labeling techniques
- implement self-supervised learning approaches
- use for data augmentation and hard negative mining
"""
    
    with open(target_path / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print("📄 created README.md with usage instructions")
    
    # final verification
    train_labeled_img_count = len(list(train_images_dir.glob("*.jpg")))
    train_labeled_lbl_count = len(list(train_labels_dir.glob("*.txt")))
    train_unlabeled_count = len(list(train_unlabeled_dir.glob("*.jpg")))
    
    val_labeled_img_count = len(list(val_images_dir.glob("*.jpg")))
    val_labeled_lbl_count = len(list(val_labels_dir.glob("*.txt")))
    val_unlabeled_count = len(list(val_unlabeled_dir.glob("*.jpg")))
    
    print(f"\n=== final verification ===")
    print(f"✅ training set:")
    print(f"   - labeled: {train_labeled_img_count} images, {train_labeled_lbl_count} labels")
    print(f"   - unlabeled: {train_unlabeled_count} images")
    print(f"✅ validation set:")
    print(f"   - labeled: {val_labeled_img_count} images, {val_labeled_lbl_count} labels") 
    print(f"   - unlabeled: {val_unlabeled_count} images")
    print(f"✅ total: {train_labeled_img_count + val_labeled_img_count + train_unlabeled_count + val_unlabeled_count} images")
    print(f"✅ dataset saved to: {target_path}")
    
    return True

if __name__ == "__main__":
    success = split_mouse_dataset_enhanced()
    if success:
        print("\n🎉 enhanced dataset splitting and organization completed!")
        print("\n💡 unlabeled images are now included and can be used for:")
        print("   - self-supervised learning")
        print("   - pseudo-labeling")  
        print("   - background/negative sampling")
        print("   - data augmentation")
    else:
        print("\n❌ dataset processing failed")
