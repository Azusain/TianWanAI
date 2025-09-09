#!/usr/bin/env python3
"""
时间段上半身提取工具
从视频的指定时间段提取所有帧，并使用姿态检测提取上半身区域
基于TshirtDetectionService的逻辑
"""

import cv2
import os
import shutil
import argparse
import numpy as np
from pathlib import Path
from loguru import logger
import time
import torch
from ultralytics import YOLO


class UpperBodyTimeRangeExtractor:
    def __init__(self, pose_model_path):
        """
        初始化上半身时间段提取工具
        
        args:
            pose_model_path: 姿态检测模型路径
        """
        self.pose_model_path = pose_model_path
        
        # 加载姿态检测模型
        logger.info(f"loading pose model from {pose_model_path}")
        self.pose_model = YOLO(pose_model_path)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pose_model.to(device)
        logger.info(f"using device: {device}")
        
        # warmup
        dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        self.pose_model.predict(source=dummy_img, verbose=False)
        logger.success("pose model loaded successfully!")
    
    def extract_upper_body_from_frame(self, frame):
        """
        从单帧中提取所有人的上半身区域
        基于TshirtDetectionService.Predict的逻辑
        
        args:
            frame: 输入帧
            
        returns:
            list: 上半身区域列表，每个元素为 {"upper_body": image_array, "bbox": [x1,y1,x2,y2], "det_confidence": float}
        """
        try:
            # step 1: 使用姿态模型检测人体
            pose_results = self.pose_model.predict(
                source=frame,
                imgsz=640,
                verbose=False
            )
            
            result = pose_results[0]
            upper_bodies = []
            
            if result.keypoints is not None and result.boxes is not None:
                keypoints = result.keypoints.data  # [N, 17, 3] for COCO format
                boxes = result.boxes.data  # [N, 6] - xyxy + conf + cls
                H, W = frame.shape[:2]
                
                for i in range(len(keypoints)):
                    person_keypoints = keypoints[i].cpu().numpy()  # [17, 3]
                    person_box = boxes[i].cpu().numpy()  # [6]
                    
                    # 人体检测置信度过滤
                    if person_box[4] < 0.5:
                        continue
                    
                    # extract shoulder and hip keypoints (COCO format)
                    left_shoulder = person_keypoints[5]   # [x, y, conf]
                    right_shoulder = person_keypoints[6]  # [x, y, conf] 
                    left_hip = person_keypoints[11]       # [x, y, conf]
                    right_hip = person_keypoints[12]      # [x, y, conf]
                    
                    # find shoulder top and hip bottom
                    shoulder_y_min = min(left_shoulder[1], right_shoulder[1])
                    hip_y_max = max(left_hip[1], right_hip[1])
                    
                    # use person bbox for x boundaries  
                    x1, y1, x2, y2 = person_box[:4]
                    
                    # calculate upper body region with margin
                    margin_y = int((hip_y_max - shoulder_y_min) * 0.1)
                    upper_y = max(0, int(shoulder_y_min - margin_y))
                    lower_y = min(H, int(hip_y_max + margin_y))
                    
                    x1, y1, x2, y2 = int(x1), int(upper_y), int(x2), int(lower_y)
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(W, x2)
                    y2 = min(H, y2)
                    
                    # 检查区域有效性
                    if x2 > x1 and y2 > y1 and (x2 - x1) > 30 and (y2 - y1) > 30:
                        # crop upper body region
                        upper_body_region = frame[y1:y2, x1:x2]
                        
                        upper_bodies.append({
                            "upper_body": upper_body_region,
                            "bbox": [x1, y1, x2, y2],
                            "det_confidence": float(person_box[4])
                        })
                        
            return upper_bodies
                
        except Exception as e:
            logger.error(f"error extracting upper body from frame: {e}")
            return []
    
    def extract_time_range_upper_bodies(self, video_path, start_time_sec, end_time_sec, output_folder, prefix="upper_body"):
        """
        从视频指定时间段提取所有上半身区域
        
        args:
            video_path: 视频文件路径
            start_time_sec: 开始时间（秒）
            end_time_sec: 结束时间（秒）
            output_folder: 输出文件夹
            prefix: 文件名前缀
        
        returns:
            提取的上半身图片数量
        """
        
        # 创建输出目录
        output_path = Path(output_folder)
        if output_path.exists():
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"created output directory: {output_folder}")
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"cannot open video: {video_path}")
            return 0
        
        # 获取视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"video info:")
        logger.info(f"  total frames: {total_frames}")
        logger.info(f"  fps: {fps:.2f}")
        logger.info(f"  duration: {duration:.2f} seconds ({duration/60:.1f} minutes)")
        logger.info(f"  resolution: {width}x{height}")
        
        # 验证时间范围
        if start_time_sec >= duration:
            logger.error(f"start time ({start_time_sec}s) exceeds video duration ({duration:.2f}s)")
            cap.release()
            return 0
        
        if end_time_sec > duration:
            logger.warning(f"end time ({end_time_sec}s) exceeds video duration ({duration:.2f}s), will extract until end")
            end_time_sec = duration
        
        # 计算帧范围
        start_frame = int(start_time_sec * fps)
        end_frame = int(end_time_sec * fps)
        expected_frames = end_frame - start_frame
        
        logger.info(f"extraction range:")
        logger.info(f"  start time: {start_time_sec}s (frame {start_frame})")
        logger.info(f"  end time: {end_time_sec}s (frame {end_frame})")
        logger.info(f"  expected frames to process: {expected_frames}")
        
        # 跳转到开始帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # 统计信息
        stats = {
            "processed_frames": 0,
            "frames_with_persons": 0,
            "frames_without_persons": 0,
            "total_upper_bodies": 0,
            "total_persons_detected": 0
        }
        
        current_frame = start_frame
        extract_start_time = time.time()
        
        logger.info("starting upper body extraction...")
        
        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"failed to read frame {current_frame}, stopping extraction")
                break
            
            stats["processed_frames"] += 1
            frame_number = current_frame - start_frame + 1
            
            # 提取上半身
            upper_bodies = self.extract_upper_body_from_frame(frame)
            
            if upper_bodies:
                stats["frames_with_persons"] += 1
                stats["total_persons_detected"] += len(upper_bodies)
                
                # 保存每个检测到的上半身
                for person_idx, upper_body_data in enumerate(upper_bodies):
                    upper_body_img = upper_body_data["upper_body"]
                    det_confidence = upper_body_data["det_confidence"]
                    
                    # 文件名格式：prefix_frame编号_person编号_置信度.jpg
                    filename = f"{prefix}_f{frame_number:06d}_p{person_idx}_{det_confidence:.2f}.jpg"
                    output_file = output_path / filename
                    
                    success = cv2.imwrite(str(output_file), upper_body_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    if success:
                        stats["total_upper_bodies"] += 1
                        
                        # 每100个上半身或每秒报告一次进度
                        if stats["total_upper_bodies"] % 50 == 0 or stats["processed_frames"] % int(fps) == 0:
                            elapsed = time.time() - extract_start_time
                            progress = stats["processed_frames"] / expected_frames * 100
                            current_time = frame_number / fps + start_time_sec
                            logger.info(f"processed {stats['processed_frames']} frames ({progress:.1f}%) - "
                                      f"extracted {stats['total_upper_bodies']} upper bodies - "
                                      f"current time: {current_time:.1f}s")
                    else:
                        logger.warning(f"failed to save upper body from frame {current_frame}")
            else:
                stats["frames_without_persons"] += 1
                
                # 偶尔报告无人帧的情况
                if stats["processed_frames"] % (int(fps) * 5) == 0:  # 每5秒报告一次
                    current_time = frame_number / fps + start_time_sec
                    logger.info(f"processed {stats['processed_frames']} frames - no persons at time {current_time:.1f}s")
            
            current_frame += 1
        
        # 统计信息
        extraction_time = time.time() - extract_start_time
        cap.release()
        
        logger.success(f"upper body extraction completed!")
        logger.info(f"summary:")
        logger.info(f"  processed frames: {stats['processed_frames']}")
        logger.info(f"  expected frames: {expected_frames}")
        logger.info(f"  frames with persons: {stats['frames_with_persons']}")
        logger.info(f"  frames without persons: {stats['frames_without_persons']}")
        logger.info(f"  total persons detected: {stats['total_persons_detected']}")
        logger.info(f"  total upper bodies saved: {stats['total_upper_bodies']}")
        logger.info(f"  extraction time: {extraction_time:.2f}s")
        logger.info(f"  processing speed: {stats['processed_frames']/extraction_time:.1f} frames/sec")
        logger.info(f"  output directory: {output_folder}")
        
        # 检查输出文件
        saved_files = list(output_path.glob("*.jpg"))
        logger.info(f"verification: found {len(saved_files)} jpg files in output directory")
        
        if saved_files:
            # 计算文件大小统计
            total_size = sum(f.stat().st_size for f in saved_files)
            avg_size = total_size / len(saved_files)
            logger.info(f"file statistics:")
            logger.info(f"  total size: {total_size/1024/1024:.1f} MB")
            logger.info(f"  average file size: {avg_size/1024:.1f} KB per upper body")
            
            # 检测率统计
            if stats["processed_frames"] > 0:
                person_detection_rate = stats["frames_with_persons"] / stats["processed_frames"] * 100
                logger.info(f"detection statistics:")
                logger.info(f"  person detection rate: {person_detection_rate:.1f}%")
                if stats["frames_with_persons"] > 0:
                    avg_persons_per_frame = stats["total_persons_detected"] / stats["frames_with_persons"]
                    logger.info(f"  average persons per frame (when detected): {avg_persons_per_frame:.1f}")
        
        return stats["total_upper_bodies"]


def main():
    parser = argparse.ArgumentParser(description="extract upper body regions from specific time range of video")
    parser.add_argument("video", help="input video file path")
    parser.add_argument("--start", type=float, required=True, help="start time in seconds")
    parser.add_argument("--end", type=float, required=True, help="end time in seconds")
    parser.add_argument("--pose-model", default="../models/yolo11m-pose.pt", 
                       help="pose detection model path")
    parser.add_argument("--output", "-o", default="extracted_upper_bodies", help="output directory")
    parser.add_argument("--prefix", default="upper_body", help="output filename prefix")
    
    args = parser.parse_args()
    
    # 验证输入
    if not os.path.exists(args.video):
        logger.error(f"video file not found: {args.video}")
        return 1
    
    if not os.path.exists(args.pose_model):
        logger.error(f"pose model not found: {args.pose_model}")
        return 1
    
    if args.start >= args.end:
        logger.error(f"start time ({args.start}s) must be less than end time ({args.end}s)")
        return 1
    
    if args.start < 0:
        logger.error(f"start time cannot be negative: {args.start}s")
        return 1
    
    logger.info(f"time range upper body extraction")
    logger.info(f"video: {args.video}")
    logger.info(f"time range: {args.start}s - {args.end}s ({args.end - args.start}s duration)")
    logger.info(f"pose model: {args.pose_model}")
    logger.info(f"output: {args.output}")
    
    try:
        extractor = UpperBodyTimeRangeExtractor(args.pose_model)
        extracted_count = extractor.extract_time_range_upper_bodies(
            video_path=args.video,
            start_time_sec=args.start,
            end_time_sec=args.end,
            output_folder=args.output,
            prefix=args.prefix
        )
        
        if extracted_count > 0:
            logger.success(f"successfully extracted {extracted_count} upper body regions!")
            return 0
        else:
            logger.error("upper body extraction failed!")
            return 1
            
    except Exception as e:
        logger.error(f"extraction failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
