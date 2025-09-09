#!/usr/bin/env python3
"""
图像分类测试工具
加载 tshirt_cls_v2 模型，对文件夹中的图片进行分类并可视化显示
"""

import os
import cv2
import numpy as np
import glob
from pathlib import Path
import argparse
import torch
from ultralytics import YOLO
from loguru import logger


class ImageClassifierTester:
    def __init__(self, model_path):
        """
        初始化分类器测试工具
        
        args:
            model_path: 分类模型路径
        """
        self.model_path = model_path
        
        logger.info(f"loading model from {model_path}")
        self.model = YOLO(model_path)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        logger.info(f"using device: {device}")
        
        # 模型类别
        self.class_names = {
            0: "other",
            1: "tshirt"
        }
        
        logger.info("model class mapping:")
        for idx, name in self.class_names.items():
            logger.info(f"  {idx}: {name}")
        
        # warmup
        dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        self.model.predict(source=dummy_img, verbose=False)
        logger.success("model loaded successfully!")
    
    def classify_image(self, image_path):
        """
        对单张图片进行分类
        
        args:
            image_path: 图片路径
            
        returns:
            tuple: (predicted_class, confidence, class_name)
        """
        try:
            # 预测
            results = self.model.predict(source=image_path, verbose=False)
            result = results[0]
            
            if result.probs is not None:
                # 获取预测结果
                probs = result.probs.data.cpu().numpy()
                predicted_class = int(result.probs.top1)
                confidence = float(result.probs.top1conf)
                class_name = self.class_names.get(predicted_class, "unknown")
                
                return predicted_class, confidence, class_name
            else:
                return None, 0.0, "no_prediction"
                
        except Exception as e:
            logger.error(f"error classifying {image_path}: {e}")
            return None, 0.0, "error"
    
    def draw_prediction(self, image, predicted_class, confidence, class_name):
        """
        在图片上绘制预测结果
        
        args:
            image: 输入图像
            predicted_class: 预测类别
            confidence: 置信度
            class_name: 类别名称
            
        returns:
            绘制后的图像
        """
        # 复制图像
        img_copy = image.copy()
        h, w = img_copy.shape[:2]
        
        # 设置文字参数
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = min(w, h) / 400  # 根据图片大小调整字体大小
        thickness = max(1, int(font_scale * 2))
        
        # 预测文本
        pred_text = f"Class: {class_name} ({predicted_class})"
        conf_text = f"Confidence: {confidence:.2%}"
        
        # 获取文本大小
        (pred_w, pred_h), _ = cv2.getTextSize(pred_text, font, font_scale, thickness)
        (conf_w, conf_h), _ = cv2.getTextSize(conf_text, font, font_scale, thickness)
        
        # 背景颜色（根据类别设置）
        if predicted_class == 1:  # tshirt
            bg_color = (0, 255, 0)  # 绿色
            text_color = (0, 0, 0)  # 黑色文字
        else:  # other
            bg_color = (0, 0, 255)  # 红色
            text_color = (255, 255, 255)  # 白色文字
        
        # 绘制背景矩形
        padding = 10
        rect_h = pred_h + conf_h + padding * 3
        rect_w = max(pred_w, conf_w) + padding * 2
        
        cv2.rectangle(img_copy, (padding, padding), 
                     (padding + rect_w, padding + rect_h), bg_color, -1)
        
        # 绘制文字
        cv2.putText(img_copy, pred_text, (padding * 2, padding * 2 + pred_h), 
                   font, font_scale, text_color, thickness)
        cv2.putText(img_copy, conf_text, (padding * 2, padding * 2 + pred_h + conf_h + padding), 
                   font, font_scale, text_color, thickness)
        
        return img_copy
    
    def test_folder(self, folder_path, image_size=(800, 600)):
        """
        测试文件夹中的所有图片
        
        args:
            folder_path: 图片文件夹路径
            image_size: 显示窗口大小 (width, height)
        """
        # 获取所有图片文件
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
            image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
        
        if not image_files:
            logger.warning(f"no image files found in {folder_path}")
            return
        
        image_files.sort()  # 按文件名排序
        total_images = len(image_files)
        logger.info(f"found {total_images} images in {folder_path}")
        
        # 统计信息
        stats = {"other": 0, "tshirt": 0, "error": 0}
        
        # 创建窗口
        window_name = "T-shirt Classifier Test"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, image_size[0], image_size[1])
        
        current_idx = 0
        while current_idx < total_images:
            image_path = image_files[current_idx]
            image_name = os.path.basename(image_path)
            
            logger.info(f"processing {current_idx + 1}/{total_images}: {image_name}")
            
            # 加载图片
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"cannot load image: {image_path}")
                current_idx += 1
                continue
            
            # 分类
            predicted_class, confidence, class_name = self.classify_image(image_path)
            
            if predicted_class is not None:
                stats[class_name] += 1
                logger.info(f"prediction: {class_name} ({predicted_class}), confidence: {confidence:.2%}")
            else:
                stats["error"] += 1
                logger.error(f"failed to classify {image_name}")
                class_name = "error"
                confidence = 0.0
                predicted_class = -1
            
            # 绘制预测结果
            display_image = self.draw_prediction(image, predicted_class, confidence, class_name)
            
            # 添加导航信息
            nav_text = f"Image {current_idx + 1}/{total_images} | Press: 'n'=next, 'p'=prev, 'q'=quit"
            cv2.putText(display_image, nav_text, (10, display_image.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # 显示图片
            cv2.imshow(window_name, display_image)
            
            # 等待按键
            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == ord('q'):  # 退出
                    cv2.destroyAllWindows()
                    self.print_statistics(stats, current_idx + 1)
                    return
                elif key == ord('n') or key == 32:  # 下一张 (n键或空格)
                    current_idx += 1
                    break
                elif key == ord('p'):  # 上一张 (p键)
                    current_idx = max(0, current_idx - 1)
                    break
                elif key == 27:  # ESC键退出
                    cv2.destroyAllWindows()
                    self.print_statistics(stats, current_idx + 1)
                    return
        
        # 显示完所有图片
        cv2.destroyAllWindows()
        self.print_statistics(stats, total_images)
        logger.success("finished testing all images!")
    
    def print_statistics(self, stats, processed_count):
        """打印统计信息"""
        logger.info("=== test statistics ===")
        logger.info(f"processed images: {processed_count}")
        logger.info(f"other: {stats['other']} images")
        logger.info(f"tshirt: {stats['tshirt']} images") 
        logger.info(f"errors: {stats['error']} images")
        
        if processed_count > stats['error']:
            total_valid = processed_count - stats['error']
            logger.info(f"tshirt rate: {stats['tshirt']/total_valid:.1%}")
            logger.info(f"other rate: {stats['other']/total_valid:.1%}")


def main():
    parser = argparse.ArgumentParser(description="test image classification model with visualization")
    parser.add_argument("folder", help="folder containing test images")
    parser.add_argument("--model", default="../models/tshirt_cls/weights/tshirt_cls_v2.pt", 
                       help="model path")
    parser.add_argument("--size", nargs=2, type=int, default=[800, 600], 
                       help="display window size (width height)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.folder):
        logger.error(f"folder not found: {args.folder}")
        return 1
    
    if not os.path.exists(args.model):
        logger.error(f"model not found: {args.model}")
        return 1
    
    try:
        tester = ImageClassifierTester(args.model)
        tester.test_folder(args.folder, tuple(args.size))
    except Exception as e:
        logger.error(f"testing failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
