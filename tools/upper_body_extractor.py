#!/usr/bin/env python3
"""
上半身裁剪工具 - 从视频中提取上半身图片

基于YOLO姿态检测，自动裁剪肩膀到髋部的上半身区域
支持单个视频处理和批量文件夹处理
"""

import os
import cv2
import numpy as np
import argparse
import glob
from pathlib import Path
from ultralytics import YOLO
import torch
from loguru import logger

class UpperBodyExtractor:
    def __init__(self, model_path, output_dir="upper_body_crops"):
        self.output_dir = output_dir
        Path(output_dir).mkdir(exist_ok=True)
        
        logger.info(f"loading model from {model_path}")
        self.model = YOLO(model_path)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        logger.info(f"using device: {device}")
        
        # warmup
        dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        self.model.predict(source=dummy_img, verbose=False)
        logger.success("model loaded successfully!")

    def extract_upper_body(self, img, keypoints, bbox):
        """extract upper body region based on keypoints"""
        H, W = img.shape[:2]
        
        # shoulder and hip points (COCO format)
        left_shoulder, right_shoulder = keypoints[5][:2], keypoints[6][:2]
        left_hip, right_hip = keypoints[11][:2], keypoints[12][:2]
        
        # calculate boundaries
        shoulder_y = min(left_shoulder[1], right_shoulder[1])
        hip_y = max(left_hip[1], right_hip[1])
        
        # add margin
        margin = int((hip_y - shoulder_y) * 0.1)
        y1 = max(0, int(shoulder_y - margin))
        y2 = min(H, int(hip_y + margin))
        x1 = max(0, int(bbox[0]))
        x2 = min(W, int(bbox[2]))
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        upper_body = img[y1:y2, x1:x2]
        
        # check minimum size
        if upper_body.shape[0] < 50 or upper_body.shape[1] < 50:
            return None
            
        return upper_body

    def process_video(self, video_path, interval=30, max_frames=100):
        """process single video file"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"cannot open video: {video_path}")
            return
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_name = Path(video_path).stem
        
        logger.info(f"processing {video_name}: {total_frames} frames")
        
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % interval == 0:
                results = self.model.predict(source=frame, imgsz=640, verbose=False)
                result = results[0]
                
                if result.keypoints is not None and result.boxes is not None:
                    keypoints = result.keypoints.data.cpu().numpy()
                    boxes = result.boxes.data.cpu().numpy()
                    
                    for person_idx, (kpts, box) in enumerate(zip(keypoints, boxes)):
                        if box[4] < 0.5:  # confidence threshold
                            continue
                            
                        upper_body = self.extract_upper_body(frame, kpts, box)
                        if upper_body is not None:
                            filename = f"upper_body_{video_name}_frame{frame_count:06d}_person{person_idx}.jpg"
                            save_path = os.path.join(self.output_dir, filename)
                            cv2.imwrite(save_path, upper_body)
                            saved_count += 1
                            
                            if saved_count >= max_frames:
                                logger.info(f"reached max frames limit ({max_frames})")
                                cap.release()
                                return saved_count
            
            frame_count += 1
            
        cap.release()
        logger.success(f"saved {saved_count} images from {video_name}")
        return saved_count

    def process_image(self, image_path):
        """process single image file"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                logger.warning(f"cannot load image: {image_path}")
                return 0
                
            image_name = Path(image_path).stem
            
            results = self.model.predict(source=img, imgsz=640, verbose=False)
            result = results[0]
            
            saved_count = 0
            if result.keypoints is not None and result.boxes is not None:
                keypoints = result.keypoints.data.cpu().numpy()
                boxes = result.boxes.data.cpu().numpy()
                
                for person_idx, (kpts, box) in enumerate(zip(keypoints, boxes)):
                    if box[4] < 0.5:  # confidence threshold
                        continue
                        
                    upper_body = self.extract_upper_body(img, kpts, box)
                    if upper_body is not None:
                        filename = f"upper_body_{image_name}_person{person_idx}.jpg"
                        save_path = os.path.join(self.output_dir, filename)
                        success = cv2.imwrite(save_path, upper_body)
                        if success:
                            saved_count += 1
                        else:
                            logger.warning(f"failed to save image: {save_path}")
            
            if saved_count == 0:
                logger.info(f"no persons detected in {image_name}, skipped")
            else:
                logger.success(f"saved {saved_count} upper body images from {image_name}")
            
            return saved_count
        except Exception as e:
            logger.error(f"error processing {image_path}: {str(e)}")
            return 0

    def process_folder(self, folder_path, interval=30, max_frames=100, process_images=False):
        """process all videos or images in folder"""
        if process_images:
            # process image files
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(glob.glob(os.path.join(folder_path, ext)))
                image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
            
            if not image_files:
                logger.warning(f"no image files found in {folder_path}")
                return
                
            logger.info(f"found {len(image_files)} images to process")
            
            total_saved = 0
            for i, image_path in enumerate(image_files, 1):
                if i % 100 == 0 or i <= 10:  # show progress every 100 images or first 10
                    logger.info(f"processing {i}/{len(image_files)}: {os.path.basename(image_path)}")
                saved = self.process_image(image_path)
                total_saved += saved
                
            logger.success(f"batch processing complete! total saved: {total_saved} upper body images")
        else:
            # process video files
            video_files = []
            for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
                video_files.extend(glob.glob(os.path.join(folder_path, ext)))
            
            if not video_files:
                logger.warning(f"no video files found in {folder_path}")
                return
                
            logger.info(f"found {len(video_files)} videos to process")
            
            total_saved = 0
            for i, video_path in enumerate(video_files, 1):
                logger.info(f"processing {i}/{len(video_files)}: {os.path.basename(video_path)}")
                saved = self.process_video(video_path, interval, max_frames)
                total_saved += saved
                
            logger.success(f"batch processing complete! total saved: {total_saved} images")

def main():
    parser = argparse.ArgumentParser(description='extract upper body regions from videos and images')
    parser.add_argument('input', help='video file, image file, or folder path')
    parser.add_argument('--model', default='../models/yolo11m-pose.pt', help='pose model path')
    parser.add_argument('--output', '-o', default='upper_body_crops', help='output directory')
    parser.add_argument('--interval', '-i', type=int, default=30, help='frame interval for videos (default: 30)')
    parser.add_argument('--max-frames', type=int, default=100, help='max images per video (default: 100)')
    parser.add_argument('--batch', '-b', action='store_true', help='process folder')
    parser.add_argument('--images', action='store_true', help='process image files instead of videos')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        logger.error(f"input not found: {args.input}")
        return
        
    if not os.path.exists(args.model):
        logger.error(f"model not found: {args.model}")
        return
    
    extractor = UpperBodyExtractor(args.model, args.output)
    
    if args.batch or os.path.isdir(args.input):
        extractor.process_folder(args.input, args.interval, args.max_frames, args.images)
    elif args.images:
        # process single image
        extractor.process_image(args.input)
    else:
        # process single video
        extractor.process_video(args.input, args.interval, args.max_frames)

if __name__ == "__main__":
    main()
