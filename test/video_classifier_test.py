#!/usr/bin/env python3
"""
视频实时分类工具
播放视频并在画面上实时显示T恤分类结果
包含人体检测、上半身提取和分类的完整流程
"""

import os
import cv2
import numpy as np
import argparse
import torch
from ultralytics import YOLO
from loguru import logger
import time


class VideoClassifierTester:
    def __init__(self, pose_model_path, cls_model_path):
        """
        初始化视频分类测试工具
        
        args:
            pose_model_path: 姿态检测模型路径
            cls_model_path: 分类模型路径
        """
        self.pose_model_path = pose_model_path
        self.cls_model_path = cls_model_path
        
        # 加载姿态检测模型
        logger.info(f"loading pose model from {pose_model_path}")
        self.pose_model = YOLO(pose_model_path)
        
        # 加载分类模型
        logger.info(f"loading classification model from {cls_model_path}")
        self.cls_model = YOLO(cls_model_path)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pose_model.to(device)
        self.cls_model.to(device)
        logger.info(f"using device: {device}")
        
        # 模型类别
        self.class_names = {
            0: "other",
            1: "tshirt"
        }
        
        logger.info("classification model class mapping:")
        for idx, name in self.class_names.items():
            logger.info(f"  {idx}: {name}")
        
        # warmup
        dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        self.pose_model.predict(source=dummy_img, verbose=False)
        self.cls_model.predict(source=dummy_img, verbose=False)
        logger.success("models loaded successfully!")
    
    def detect_and_classify_persons(self, frame):
        """
        检测人体、提取上半身并进行分类
        
        args:
            frame: 输入帧
            
        returns:
            list: 每个人的检测结果 [{"bbox": [x1,y1,x2,y2], "class": int, "confidence": float, "class_name": str}]
        """
        try:
            # step 1: 使用姿态模型检测人体
            pose_results = self.pose_model.predict(
                source=frame,
                imgsz=640,
                verbose=False
            )
            
            result = pose_results[0]
            persons_results = []
            
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
                    
                    if x2 > x1 and y2 > y1 and (x2 - x1) > 30 and (y2 - y1) > 30:
                        # crop upper body region
                        upper_body_region = frame[y1:y2, x1:x2]
                        
                        # classify tshirt on upper body region
                        cls_results = self.cls_model.predict(source=upper_body_region, verbose=False)
                        cls_result = cls_results[0]
                        
                        if cls_result.probs is not None:
                            predicted_class = int(cls_result.probs.top1)
                            confidence = float(cls_result.probs.top1conf)
                            class_name = self.class_names.get(predicted_class, "unknown")
                            
                            persons_results.append({
                                "bbox": [x1, y1, x2, y2],
                                "det_confidence": float(person_box[4]),
                                "class": predicted_class,
                                "confidence": confidence,
                                "class_name": class_name
                            })
                        
            return persons_results
                
        except Exception as e:
            logger.error(f"error detecting and classifying frame: {e}")
            return []
    
    def draw_detections(self, frame, persons_results, frame_info=""):
        """
        在帧上绘制所有人的检测结果
        
        args:
            frame: 输入帧
            persons_results: 检测结果列表
            frame_info: 帧信息
            
        returns:
            绘制后的帧
        """
        # 复制帧
        frame_copy = frame.copy()
        h, w = frame_copy.shape[:2]
        
        # 调整字体大小
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = min(w, h) / 1000  # 更小的字体
        thickness = max(1, int(font_scale * 4))
        
        # 绘制每个检测到的人
        for i, person in enumerate(persons_results):
            x1, y1, x2, y2 = person["bbox"]
            predicted_class = person["class"]
            confidence = person["confidence"]
            class_name = person["class_name"]
            det_confidence = person["det_confidence"]
            
            # 根据类别设置颜色
            if predicted_class == 1:  # tshirt
                bbox_color = (0, 255, 0)  # 绿色
                text_bg_color = (0, 255, 0)
                text_color = (0, 0, 0)  # 黑色文字
            else:  # other
                bbox_color = (0, 0, 255)  # 红色
                text_bg_color = (0, 0, 255)
                text_color = (255, 255, 255)  # 白色文字
            
            # 绘制边界框
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), bbox_color, 2)
            
            # 准备文本
            main_text = f"{class_name.upper()}"
            conf_text = f"{confidence:.1%}"
            det_text = f"Det: {det_confidence:.1%}"
            
            # 获取文本大小
            (main_w, main_h), _ = cv2.getTextSize(main_text, font, font_scale, thickness)
            (conf_w, conf_h), _ = cv2.getTextSize(conf_text, font, font_scale * 0.8, thickness)
            (det_w, det_h), _ = cv2.getTextSize(det_text, font, font_scale * 0.7, thickness)
            
            # 计算标签框大小
            label_w = max(main_w, conf_w, det_w) + 10
            label_h = main_h + conf_h + det_h + 15
            
            # 标签位置（在边界框上方）
            label_x = x1
            label_y = max(y1 - label_h - 5, 0)
            
            # 绘制标签背景
            cv2.rectangle(frame_copy, (label_x, label_y), 
                         (label_x + label_w, label_y + label_h), text_bg_color, -1)
            
            # 绘制标签边框
            cv2.rectangle(frame_copy, (label_x, label_y), 
                         (label_x + label_w, label_y + label_h), (255, 255, 255), 1)
            
            # 绘制文字
            text_x = label_x + 5
            text_y = label_y + main_h + 5
            
            cv2.putText(frame_copy, main_text, (text_x, text_y), 
                       font, font_scale, text_color, thickness)
            text_y += conf_h + 5
            cv2.putText(frame_copy, conf_text, (text_x, text_y), 
                       font, font_scale * 0.8, text_color, max(1, thickness-1))
            text_y += det_h + 5
            cv2.putText(frame_copy, det_text, (text_x, text_y), 
                       font, font_scale * 0.7, text_color, max(1, thickness-1))
        
        # 添加帧信息和统计在左下角
        if frame_info:
            info_lines = [
                frame_info,
                f"Persons detected: {len(persons_results)}"
            ]
            
            # 统计各类别数量
            tshirt_count = sum(1 for p in persons_results if p["class"] == 1)
            other_count = len(persons_results) - tshirt_count
            
            if len(persons_results) > 0:
                info_lines.extend([
                    f"T-shirt: {tshirt_count}, Other: {other_count}"
                ])
            
            info_font_scale = font_scale * 0.8
            info_thickness = max(1, int(info_font_scale * 3))
            
            for i, line in enumerate(info_lines):
                y_pos = h - 20 - (len(info_lines) - 1 - i) * 25
                cv2.putText(frame_copy, line, (10, y_pos), 
                           font, info_font_scale, (255, 255, 255), info_thickness)
        
        return frame_copy
    
    def test_video(self, video_path, skip_frames=5, window_size=(1000, 700)):
        """
        测试视频文件
        
        args:
            video_path: 视频文件路径
            skip_frames: 跳过帧数（每N帧分类一次，减少计算量）
            window_size: 显示窗口大小 (width, height)
        """
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"cannot open video: {video_path}")
            return
        
        # 获取视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"video info: {total_frames} frames, {fps:.1f} fps, {width}x{height}")
        
        # 创建窗口
        window_name = "T-shirt Video Classifier"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, window_size[0], window_size[1])
        
        # 统计信息
        stats = {"other": 0, "tshirt": 0, "no_person": 0, "total_classified": 0}
        
        frame_count = 0
        last_persons_results = []
        
        # 计算播放延迟
        frame_delay = max(1, int(1000 / fps)) if fps > 0 else 33  # 毫秒
        
        logger.info(f"starting video playback (press 'q' to quit, 'space' to pause)")
        
        paused = False
        last_time = time.time()
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    logger.info("reached end of video")
                    break
                
                # 每skip_frames帧进行一次检测和分类
                if frame_count % skip_frames == 0:
                    persons_results = self.detect_and_classify_persons(frame)
                    
                    # 更新统计信息
                    for person in persons_results:
                        class_name = person["class_name"]
                        stats[class_name] += 1
                        stats["total_classified"] += 1
                    
                    if not persons_results:
                        stats["no_person"] = stats.get("no_person", 0) + 1
                    
                    last_persons_results = persons_results
                    
                    if frame_count % (skip_frames * 10) == 0:  # 每50帧打印一次
                        if persons_results:
                            for i, person in enumerate(persons_results):
                                logger.info(f"frame {frame_count} person{i}: {person['class_name']} ({person['class']}), confidence: {person['confidence']:.1%}")
                        else:
                            logger.info(f"frame {frame_count}: no persons detected")
                else:
                    # 使用最近的检测结果
                    persons_results = last_persons_results if 'last_persons_results' in locals() else []
                
                # 帧信息
                frame_info = f"Frame: {frame_count}/{total_frames} | FPS: {fps:.1f}"
                
                # 绘制检测结果
                display_frame = self.draw_detections(frame, persons_results, frame_info)
                
                frame_count += 1
            else:
                # 暂停时使用当前帧和结果
                persons_results = last_persons_results if 'last_persons_results' in locals() else []
                display_frame = self.draw_detections(frame, persons_results, f"PAUSED - Frame: {frame_count}/{total_frames}")
            
            # 显示帧
            cv2.imshow(window_name, display_frame)
            
            # 控制播放速度和处理按键
            current_time = time.time()
            elapsed = (current_time - last_time) * 1000  # 转换为毫秒
            wait_time = max(1, int(frame_delay - elapsed))
            
            key = cv2.waitKey(wait_time) & 0xFF
            last_time = time.time()
            
            if key == ord('q') or key == 27:  # q键或ESC键退出
                break
            elif key == 32:  # 空格键暂停/继续
                paused = not paused
                if paused:
                    logger.info("video paused (press space to continue)")
                else:
                    logger.info("video resumed")
        
        # 清理
        cap.release()
        cv2.destroyAllWindows()
        
        # 显示最终统计
        self.print_video_statistics(stats, frame_count, total_frames)
    
    def print_video_statistics(self, stats, processed_frames, total_frames):
        """打印视频统计信息"""
        logger.info("=== video test statistics ===")
        logger.info(f"total frames: {total_frames}")
        logger.info(f"processed frames: {processed_frames}")
        logger.info(f"total person detections: {stats['total_classified']}")
        logger.info(f"tshirt detections: {stats['tshirt']}")
        logger.info(f"other detections: {stats['other']}")
        logger.info(f"frames with no persons: {stats.get('no_person', 0)}")
        
        if stats['total_classified'] > 0:
            tshirt_rate = stats['tshirt'] / stats['total_classified']
            other_rate = stats['other'] / stats['total_classified']
            logger.info(f"tshirt rate: {tshirt_rate:.1%}")
            logger.info(f"other rate: {other_rate:.1%}")


def main():
    parser = argparse.ArgumentParser(description="test video classification model with real-time visualization")
    parser.add_argument("video", help="video file path")
    parser.add_argument("--pose-model", default="../models/yolo11m-pose.pt", 
                       help="pose detection model path")
    parser.add_argument("--cls-model", default="../models/tshirt_cls/weights/tshirt_cls_v2.pt", 
                       help="classification model path")
    parser.add_argument("--size", nargs=2, type=int, default=[1000, 700], 
                       help="display window size (width height)")
    parser.add_argument("--skip", type=int, default=5,
                       help="classify every N frames (default: 5)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        logger.error(f"video not found: {args.video}")
        return 1
    
    if not os.path.exists(args.pose_model):
        logger.error(f"pose model not found: {args.pose_model}")
        return 1
        
    if not os.path.exists(args.cls_model):
        logger.error(f"classification model not found: {args.cls_model}")
        return 1
    
    try:
        tester = VideoClassifierTester(args.pose_model, args.cls_model)
        tester.test_video(args.video, args.skip, tuple(args.size))
    except Exception as e:
        logger.error(f"video testing failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
