#!/usr/bin/env python3
"""
Convert YOLOS fashionpedia model to standard YOLO .pt format
"""

import torch
import json
from pathlib import Path
import sys
import os

def convert_yolos_to_yolo_pt():
    """
    Convert YOLOS fashionpedia model to YOLO .pt format
    """
    print("=== YOLOS Fashionpedia Model Conversion ===")
    
    # Paths
    model_dir = Path("C:/Users/azusaing/Desktop/Code/tianwan/models/fashionpedia")
    output_path = Path("C:/Users/azusaing/Desktop/Code/tianwan/models/fashionpedia.pt")
    
    config_path = model_dir / "config.json"
    weights_path = model_dir / "pytorch_model.bin"
    
    print(f"源模型目录: {model_dir}")
    print(f"输出路径: {output_path}")
    
    # Check if files exist
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return False
    
    if not weights_path.exists():
        print(f"❌ 权重文件不存在: {weights_path}")
        return False
    
    try:
        # Load config
        print("📋 读取配置文件...")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Load weights
        print("⚖️ 读取权重文件...")
        weights = torch.load(weights_path, map_location='cpu')
        
        print(f"📊 模型信息:")
        print(f"  模型类型: {config.get('model_type', 'unknown')}")
        print(f"  架构: {config.get('architectures', ['unknown'])[0]}")
        print(f"  类别数: {len(config.get('id2label', {}))}")
        print(f"  图像尺寸: {config.get('image_size', 'unknown')}")
        print(f"  权重参数数量: {len(weights)}")
        
        # Extract class names
        id2label = config.get('id2label', {})
        class_names = [id2label[str(i)] for i in range(len(id2label))]
        
        print(f"📝 类别列表 (前10个):")
        for i, name in enumerate(class_names[:10]):
            print(f"  {i}: {name}")
        if len(class_names) > 10:
            print(f"  ... 还有 {len(class_names) - 10} 个类别")
        
        # Create YOLO-compatible model structure
        print("🔄 创建 YOLO 兼容的模型结构...")
        
        # This is a simplified conversion - the actual model architecture
        # would need to be properly converted from YOLOS to YOLO format
        yolo_model = {
            'model': weights,  # Original weights (would need architecture conversion)
            'epoch': 0,
            'date': None,
            'model_info': {
                'source_model': 'YOLOS-fashionpedia',
                'original_architecture': config.get('architectures', ['YolosForObjectDetection'])[0],
                'num_classes': len(class_names),
                'class_names': class_names,
                'image_size': config.get('image_size', [512, 864]),
                'model_type': config.get('model_type', 'yolos')
            },
            'optimizer': None,
            'training_results': None,
            'date': None
        }
        
        # Save as .pt file
        print("💾 保存为 .pt 格式...")
        torch.save(yolo_model, output_path)
        
        # Verify saved file
        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        print(f"✅ 模型已保存: {output_path}")
        print(f"📁 文件大小: {file_size:.1f} MB")
        
        # Create accompanying metadata file
        metadata_path = output_path.with_suffix('.json')
        metadata = {
            'model_name': 'fashionpedia-yolos',
            'source': 'valentinafeve/yolos-fashionpedia',
            'model_type': 'YOLOS (Vision Transformer + YOLO)',
            'num_classes': len(class_names),
            'classes': class_names,
            'image_size': config.get('image_size', [512, 864]),
            'description': 'Fine-tuned YOLOS model for fashion object detection',
            'note': 'This is a converted model from HuggingFace format. Architecture may need adaptation for standard YOLO usage.'
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"📄 元数据已保存: {metadata_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ 转换失败: {str(e)}")
        return False

def test_model_loading():
    """
    Test if the converted model can be loaded
    """
    print("\n=== 测试模型加载 ===")
    
    model_path = Path("C:/Users/azusaing/Desktop/Code/tianwan/models/fashionpedia.pt")
    
    if not model_path.exists():
        print("❌ 模型文件不存在")
        return False
    
    try:
        print("📂 加载转换后的模型...")
        model_data = torch.load(model_path, map_location='cpu')
        
        print("✅ 模型加载成功!")
        print(f"📊 模型信息:")
        
        if 'model_info' in model_data:
            info = model_data['model_info']
            print(f"  源模型: {info.get('source_model', 'unknown')}")
            print(f"  类别数: {info.get('num_classes', 'unknown')}")
            print(f"  图像尺寸: {info.get('image_size', 'unknown')}")
        
        if 'model' in model_data:
            weights = model_data['model']
            print(f"  权重参数: {len(weights)} 个参数组")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型加载失败: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 开始 YOLOS Fashionpedia 模型转换...")
    
    success = convert_yolos_to_yolo_pt()
    
    if success:
        print("\n✅ 转换完成!")
        # Test loading
        test_model_loading()
        
        print(f"\n📋 使用说明:")
        print(f"1. 转换后的模型: C:/Users/azusaing/Desktop/Code/tianwan/models/fashionpedia.pt")
        print(f"2. 模型元数据: C:/Users/azusaing/Desktop/Code/tianwan/models/fashionpedia.json")
        print(f"3. ⚠️ 注意: 这是一个 YOLOS 模型，架构与标准 YOLO 不同")
        print(f"4. 💡 建议: 使用 HuggingFace transformers 库来加载和使用此模型")
        
    else:
        print("\n❌ 转换失败!")
        sys.exit(1)
