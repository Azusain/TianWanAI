import os
import re

# api
from enum import Enum
from uuid import uuid4 as uuid4

# model
from ultralytics import YOLO
import torch
from transformers import YolosImageProcessor, YolosForObjectDetection

# temporal fall detection - moved to this file
import sys
import time
from collections import defaultdict, deque
import numpy as np
import cv2

# logger  
from loguru import logger

VERSION_API = '0.0.1'

class ServiceStatus(Enum):
    SUCCESS = 0
    INVALID_CONTENT_TYPE = -1
    MISSING_IMAGE_DATA = -2
    INVALID_IMAGE_FORMAT = -3
    NO_OBJECT_DETECTED = -4
    
    def stringify(errno):
        if errno == ServiceStatus.SUCCESS.value:
            return 'SUCCESS'
        elif errno == ServiceStatus.INVALID_CONTENT_TYPE.value:
            return'INVALID_CONTENT_TYPE'
        elif errno == ServiceStatus.MISSING_IMAGE_DATA.value: 
            return 'MISSING_IMAGE_DATA'    
        elif errno == ServiceStatus.INVALID_IMAGE_FORMAT.value:
            return'INVALID_IMAGE_FORMAT'
        elif errno == ServiceStatus.NO_OBJECT_DETECTED.value:
            return 'NO_OBJECT_DETECTED'

class YoloDetectionService():
    def __init__(self, model_path, imgsz) -> None:
        self.version = "0.0.1"
        self.path = model_path
        self.imgsz = imgsz
        
        if not os.path.exists(model_path):
            logger.error(f"model path doesn't exist: {model_path}")
            exit(1)
        
        # loading model
        logger.debug("loading model...")
        self.model = YOLO(self.path)
        self.device = (
            torch.device('cuda') 
            if torch.cuda.is_available() 
            else torch.device('cpu')
        )
        self.model.to(self.device)
        self.is_half = (self.device == 'cpu')
        logger.info(f"device: {self.device.type}")

        # model pre-heating
        logger.debug("model preheating...")
        rand_data = torch.randn(1, 3, 640, 640)
        rand_data = (rand_data - torch.min(rand_data)) / (torch.max(rand_data) - torch.min(rand_data))
        self.model.predict(
            source=rand_data,
            verbose=False
        )
        
    def Predict(self, img):
      results = self.model.predict(
          source=img, 
          imgsz=self.imgsz,
          half=self.is_half,
          verbose=False
      )
      # containing multiple boxes' coordinates
      boxes = results[0].boxes 
      score = None
      xyxyn = None
      # if target exists.
      if boxes.cls.numel() != 0:  
        if len(boxes.conf) == 1:
          score = float(boxes.conf)
        else:
          score = boxes.conf
        xyxyn = boxes.xyxyn
      return score, xyxyn, boxes.cls

    def Response(self, errno, score=None, xyxyn=None):
        if score is not None:
            errno = ServiceStatus.SUCCESS.value
        elif errno is None: 
            errno = ServiceStatus.NO_OBJECT_DETECTED.value
              
        err_msg = ServiceStatus.stringify(errno)
        if errno != 0:
            logger.error(err_msg)
        else:
            logger.success(err_msg + f" - score: {score}")
        
        left, top, width, height = None, None, None, None
        if score is not None and xyxyn is not None:
            xyxyn = xyxyn[0].tolist() 
            left    = xyxyn[0]
            top     = xyxyn[1]
            width   = xyxyn[2] - xyxyn[0]
            height  = xyxyn[3] - xyxyn[1]
            
        return {
            "log_id": uuid4(),
            "errno": errno,
            "err_msg": err_msg,
            "api_version": VERSION_API,
            "model_version": self.version,
            "results": [{
                "score": score,
                "location": {
                    "left": left,
                    "top": top,
                    "width": width,
                    "height": height
                }
            }]
        }
        
class TshirtDetectionService():
    def __init__(self, model_dir_path, imgsz) -> None:
      self.version = "0.0.1"
      self.path = model_dir_path
      self.imgsz = imgsz
      self.feature_extractor = YolosImageProcessor.from_pretrained(
        model_dir_path, local_files_only=True, size={"shortest_edge": imgsz, "longest_edge": imgsz}  
      )
      self.model = YolosForObjectDetection.from_pretrained(model_dir_path, local_files_only=True)
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      self.model.to(self.device)
      logger.info(f"device: {self.device.type}")
      
    def Predict(self, img):
      encoding = self.feature_extractor(images=img, return_tensors="pt")
      encoding = {k: v.to(self.device) for k, v in encoding.items()}
      with torch.no_grad():
          outputs = self.model(**encoding)
      scores = outputs.logits.softmax(-1)[0, :, :-1]  
      labels = scores.argmax(-1)
      scores = scores.max(-1).values
      boxes = outputs.pred_boxes[0]
      threshold = 0.2
      selected = torch.where(scores > threshold)
      selected_scores = scores[selected]
      selected_labels = labels[selected]
      selected_boxes = boxes[selected]
      h, w = img.shape[:2]
      pixel_boxes = []
      for box in selected_boxes:
          cx, cy, bw, bh = box
          x1 = (cx - bw / 2) * w
          x2 = (cx + bw / 2) * w
          y1 = (cy - bh / 2) * h
          y2 = (cy + bh / 2) * h
          pixel_boxes.append([x1.item(), y1.item(), x2.item(), y2.item()])
      return selected_scores, selected_labels, pixel_boxes


# Pose utilities for fall detection
def normalize_points_with_size(xy, width, height, flip=False):
    """Normalize scale points in image with size of image to (0-1).
    xy : (frames, parts, xy) or (parts, xy)
    """
    if xy.ndim == 2:
        xy = np.expand_dims(xy, 0)
    xy[:, :, 0] /= width
    xy[:, :, 1] /= height
    if flip:
        xy[:, :, 0] = 1 - xy[:, :, 0]
    return xy


def scale_pose(xy):
    """Normalize pose points by scale with max/min value of each pose.
    xy : (frames, parts, xy) or (parts, xy)
    """
    if xy.ndim == 2:
        xy = np.expand_dims(xy, 0)
    xy_min = np.nanmin(xy, axis=1)
    xy_max = np.nanmax(xy, axis=1)
    for i in range(xy.shape[0]):
        xy[i] = ((xy[i] - xy_min[i]) / (xy_max[i] - xy_min[i])) * 2 - 1
    return xy.squeeze()


def convert_coco_to_coco_cut(coco_keypoints):
    """Convert COCO 17-point keypoints to coco_cut 14-point format."""
    if coco_keypoints.ndim == 2:
        # Single frame: [17, 3] -> [14, 3]
        coco_cut = np.zeros((14, 3))
        
        # Map COCO indices to coco_cut indices (skip eyes and ears)
        coco_to_cut_mapping = {
            0: 0,   # nose -> nose
            5: 1,   # left_shoulder -> left_shoulder  
            6: 2,   # right_shoulder -> right_shoulder
            7: 3,   # left_elbow -> left_elbow
            8: 4,   # right_elbow -> right_elbow
            9: 5,   # left_wrist -> left_wrist
            10: 6,  # right_wrist -> right_wrist
            11: 7,  # left_hip -> left_hip
            12: 8,  # right_hip -> right_hip
            13: 9,  # left_knee -> left_knee
            14: 10, # right_knee -> right_knee
            15: 11, # left_ankle -> left_ankle
            16: 12, # right_ankle -> right_ankle
        }
        
        # Copy mapped keypoints
        for coco_idx, cut_idx in coco_to_cut_mapping.items():
            coco_cut[cut_idx] = coco_keypoints[coco_idx]
        
        # Calculate neck center from shoulders (index 13)
        left_shoulder = coco_keypoints[5]   # COCO left_shoulder
        right_shoulder = coco_keypoints[6]  # COCO right_shoulder
        
        if left_shoulder[2] > 0.1 and right_shoulder[2] > 0.1:  # Both shoulders visible
            coco_cut[13] = (left_shoulder + right_shoulder) / 2
        elif left_shoulder[2] > 0.1:  # Only left shoulder visible
            coco_cut[13] = left_shoulder
        elif right_shoulder[2] > 0.1:  # Only right shoulder visible
            coco_cut[13] = right_shoulder
        else:  # Neither shoulder visible
            coco_cut[13] = np.array([0, 0, 0])
            
        return coco_cut
        
    elif coco_keypoints.ndim == 3:
        # Multiple frames: [T, 17, 3] -> [T, 14, 3]
        T = coco_keypoints.shape[0]
        coco_cut_sequence = np.zeros((T, 14, 3))
        
        for t in range(T):
            coco_cut_sequence[t] = convert_coco_to_coco_cut(coco_keypoints[t])
            
        return coco_cut_sequence
    
    else:
        raise ValueError(f"Unsupported keypoints shape: {coco_keypoints.shape}")


class TemporalFallDetectionService:
    """Temporal Fall Detection Service using original project's complete model chain"""
    
    def __init__(self, model_path=None, device=None):
        self.version = "0.0.1"
        
        # Add original project to path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'fall-detection'))
        
        # Device setup
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        else:
            self.device = device
            
        logger.info(f"Fall detection device: {self.device}")
        
        # Initialize tracking attributes
        self.model_path = model_path
        self.model = None  # ST-GCN model (optional)
        self.sequence_length = 30  # Frames needed for temporal detection
        self.confidence_threshold = 0.6  # Minimum confidence for fall detection
        self.max_persons = 5  # Maximum persons to track per frame
        self.alert_cooldown = 10.0  # Seconds before re-alerting for same person
        
        # Tracking structures
        self.person_sequences = defaultdict(lambda: defaultdict(deque))
        self.person_timestamps = defaultdict(dict)
        self.fall_alerts = defaultdict(list)
        
        # Import original project modules
        from DetectorLoader import TinyYOLOv3_onecls
        from PoseEstimateLoader import SPPE_FastPose
        from ActionsEstLoader import TSSTG
        from Track.Tracker import Tracker, Detection
        
        # Define kpt2bbox function (from original main.py)
        def kpt2bbox(kpt, ex=20):
            """Get bbox that hold on all of the keypoints (x,y)"""
            return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                           kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))
        self.kpt2bbox = kpt2bbox
        
        # Initialize all models following main.py
        self.inp_dets = 384  # Detection input size
        # Use paths relative to fall-detection directory
        config_path = os.path.join('fall-detection', 'Models', 'yolo-tiny-onecls', 'yolov3-tiny-onecls.cfg')
        weight_path = os.path.join('fall-detection', 'Models', 'yolo-tiny-onecls', 'best-model.pth')
        self.detect_model = TinyYOLOv3_onecls(
            input_size=self.inp_dets,
            config_file=config_path,
            weight_file=weight_path,
            device=self.device
        )
        logger.info("Detection model loaded")
        
        # Pose model - same as main.py line 70
        self.pose_model = SPPE_FastPose('resnet50', 224, 160, device=self.device)
        logger.info("Pose model loaded")
        
        # Tracker - same as main.py line 74
        self.tracker = Tracker(max_age=30, n_init=3)
        logger.info("Tracker initialized")
        
        # Action model - same as main.py line 77
        self.action_model = TSSTG(device=self.device)
        logger.info("Action model loaded")
        
        # Store necessary imports
        self.Detection = Detection
        
        # Class names from action model
        self.class_names = ['Standing', 'Walking', 'Sitting', 'Lying Down',
                          'Stand up', 'Sit down', 'Fall Down']
        
    def _load_model(self):
        """Load ST-GCN model for fall detection"""
        try:
            # Add fall-detection submodule to path
            sys.path.append(os.path.join(os.path.dirname(__file__), 'fall-detection'))
            
            from Actionsrecognition.Models import TwoStreamSpatialTemporalGraph
            
            if not os.path.exists(self.model_path):
                logger.warning(f"ST-GCN model not found at {self.model_path}")
                logger.info("Using fallback pose-based detection")
                self.model = None
                return
                
            # Load ST-GCN model with correct parameters
            graph_args = {'strategy': 'spatial'}  # Use default coco_cut layout
            num_class = 7  # Based on ActionsEstLoader: Standing, Walking, Sitting, Lying Down, Stand up, Sit down, Fall Down
            
            self.model = TwoStreamSpatialTemporalGraph(
                graph_args=graph_args,
                num_class=num_class
            )
            
            # Load weights
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
                
            self.model.to(self.device)
            self.model.eval()
            logger.info("ST-GCN model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load ST-GCN model: {e}")
            logger.info("Using fallback pose-based detection")
            self.model = None
    
    def extract_pose_keypoints(self, image):
        """Extract pose keypoints from image using original project models"""
        try:
            # Step 1: Detect humans using TinyYOLO (it handles resizing internally)
            detected = self.detect_model.detect(image, need_resize=True, expand_bb=10)
            
            persons = []
            if detected is not None and len(detected) > 0:
                # Step 2: Extract poses for each detection
                poses = self.pose_model.predict(image, detected[:, 0:4], detected[:, 4])
                
                # Step 3: Create person data for each pose
                for i, ps in enumerate(poses):
                    # Extract keypoints (14 points + scores)
                    kpts = ps['keypoints'].numpy()  # Shape: [14, 2]
                    scores = ps['kp_score'].numpy()  # Shape: [14, 1]
                    
                    # Combine keypoints with scores for compatibility
                    keypoints = np.concatenate([kpts, scores], axis=1)  # Shape: [14, 3]
                    
                    # Get bbox from keypoints
                    bbox = self.kpt2bbox(kpts)
                    
                    person = {
                        'keypoints': keypoints,  # [14, 3] - x, y, confidence
                        'bbox': bbox,   # [x1, y1, x2, y2]
                        'person_id': i,
                        'raw_pose': ps  # Keep raw pose data
                    }
                    persons.append(person)
                            
            return persons
            
        except Exception as e:
            logger.error(f"Pose extraction failed: {e}")
            return []
    
    def detect_fall_by_pose(self, keypoints):
        """Fallback fall detection using pose analysis"""
        try:
            # Extract key body points (COCO format)
            head_y = keypoints[0, 1]      # nose
            shoulder_y = (keypoints[5, 1] + keypoints[6, 1]) / 2  # shoulders
            hip_y = (keypoints[11, 1] + keypoints[12, 1]) / 2     # hips
            
            # Check if points are valid (confidence > 0.3)
            valid_points = [
                keypoints[0, 2] > 0.3,    # head
                keypoints[5, 2] > 0.3 or keypoints[6, 2] > 0.3,  # shoulders
                keypoints[11, 2] > 0.3 or keypoints[12, 2] > 0.3  # hips
            ]
            
            if not all(valid_points):
                return False, 0.0
            
            # Calculate body orientation
            body_height = abs(head_y - hip_y)
            if body_height < 10:  # Too small to analyze
                return False, 0.0
                
            # Fall detection logic
            # 1. Head lower than hips (person lying down)
            head_below_hips = head_y > hip_y
            
            # 2. Body is more horizontal than vertical
            shoulder_hip_diff = abs(shoulder_y - hip_y)
            horizontal_ratio = shoulder_hip_diff / body_height if body_height > 0 else 0
            
            is_horizontal = horizontal_ratio < 0.3  # Body is quite horizontal
            
            # Combine conditions
            fall_detected = head_below_hips and is_horizontal
            confidence = 0.8 if fall_detected else 0.2
            
            return fall_detected, confidence
            
        except Exception as e:
            logger.error(f"Pose-based fall detection failed: {e}")
            return False, 0.0
    
    def process_frame(self, image, camera_id):
        """Process single frame for fall detection"""
        current_time = time.time()
        
        # Extract pose keypoints
        persons = self.extract_pose_keypoints(image)
        
        results = {
            'camera_id': camera_id,
            'timestamp': current_time,
            'persons_detected': len(persons),
            'fall_detected': False,
            'alerts': []
        }
        
        if not persons:
            return results
        
        # Process each detected person
        for person in persons[:self.max_persons]:
            person_id = f"person_{person['person_id']}"
            keypoints = person['keypoints']
            
            # Add to sequence
            self.person_sequences[camera_id][person_id].append({
                'keypoints': keypoints,
                'timestamp': current_time,
                'bbox': person['bbox']
            })
            
            self.person_timestamps[camera_id][person_id] = current_time
            
            # Check for fall detection
            fall_detected, confidence = self._detect_fall_for_person(
                camera_id, person_id, keypoints
            )
            
            if fall_detected and confidence > self.confidence_threshold:
                # Check cooldown to avoid spam
                if self._can_alert(camera_id, person_id, current_time):
                    alert = {
                        'person_id': person_id,
                        'confidence': confidence,
                        'bbox': person['bbox'].tolist(),
                        'timestamp': current_time,
                        'alert_type': 'FALL_DETECTED'
                    }
                    
                    results['alerts'].append(alert)
                    results['fall_detected'] = True
                    
                    # Record alert
                    self.fall_alerts[camera_id].append({
                        'person_id': person_id,
                        'timestamp': current_time
                    })
                    
                    logger.warning(f"FALL DETECTED - Camera: {camera_id}, Person: {person_id}, Confidence: {confidence:.2f}")
        
        # Cleanup old sequences and alerts
        self._cleanup_old_data(camera_id, current_time)
        
        return results
    
    def _detect_fall_for_person(self, camera_id, person_id, current_keypoints):
        """Detect fall for a specific person"""
        sequence = self.person_sequences[camera_id][person_id]
        
        # If we have the ST-GCN model and enough frames
        if self.model and len(sequence) >= self.sequence_length:
            # Get image size from the first frame in sequence
            image_size = (640, 480)  # Default size, could be extracted from actual image
            return self._detect_fall_stgcn(sequence, image_size)
        
        # Fallback to pose-based detection
        return self.detect_fall_by_pose(current_keypoints)
    
    def _detect_fall_stgcn(self, sequence, image_size=(640, 480)):
        """Use ST-GCN model for fall detection - exactly copying original ActionsEstLoader.predict method"""
        try:
            # 完全按照原始 Human-Falling-Detect-Tracks 项目中的实现
            # main.py 中创建 Detection 对象，然后 tracker.keypoints_list 包含这些 Detection 对象的关键点
            # 在 ActionsEstLoader.py 的 predict 函数中直接处理这些关键点
            
            # 获取关键点序列
            keypoints_sequence = []
            for frame_data in sequence:
                keypoints = frame_data['keypoints']  # [17, 3] COCO format from YOLO
                keypoints_sequence.append(keypoints)
            
            # 转换为numpy数组 - 与 main.py 的 line 150 匹配: pts = np.array(track.keypoints_list, dtype=np.float32)
            pts = np.array(keypoints_sequence, dtype=np.float32)  # [30, 17, 3]
            
            # 严格按照 ActionsEstLoader.predict() 处理:
            
            # 1. 对关键点坐标进行归一化
            pts[:, :, :2] = normalize_points_with_size(pts[:, :, :2], image_size[0], image_size[1])
            
            # 2. 对关键点进行缩放
            pts[:, :, :2] = scale_pose(pts[:, :, :2])
            
            # 3. 添加颈部中心点 - 使用左肩和右肩的平均值 (索引5和6是COCO格式中的左肩和右肩)
            # 注意: 原始代码使用的是pts[:, 1, :] + pts[:, 2, :]，但这是基于不同的关键点格式
            # 在COCO格式中，索引5和6是左肩和右肩
            pts = np.concatenate((pts, np.expand_dims((pts[:, 5, :] + pts[:, 6, :]) / 2, 1)), axis=1)  # [30, 18, 3]
            
            # 4. 转换为PyTorch张量并调整维度顺序
            pts = torch.tensor(pts, dtype=torch.float32)
            pts = pts.permute(2, 0, 1)[None, :]  # [1, 3, 30, 18]
            
            # 5. 准备运动流 - 计算帧间差异
            mot = pts[:, :2, 1:, :] - pts[:, :2, :-1, :]  # [1, 2, 29, 18]
            mot = mot.to(self.device)
            pts = pts.to(self.device)
            
            # 6. 模型推理
            out = self.model((pts, mot))
            
            # 7. 转换为概率并获取跌倒检测结果
            probabilities = torch.softmax(out, dim=1)
            fall_prob = probabilities[0, 6].item()  # 类别6是'Fall Down'
            
            fall_detected = fall_prob > 0.5
            return fall_detected, fall_prob
            
        except Exception as e:
            logger.error(f"ST-GCN inference failed: {e}")
            # 回退到基于姿态的检测
            if sequence:
                return self.detect_fall_by_pose(sequence[-1]['keypoints'])
            return False, 0.0
    
    def _can_alert(self, camera_id, person_id, current_time):
        """Check if we can send an alert (cooldown logic)"""
        alerts = self.fall_alerts.get(camera_id, [])
        
        for alert in reversed(alerts):  # Check recent alerts first
            if alert['person_id'] == person_id:
                if current_time - alert['timestamp'] < self.alert_cooldown:
                    return False
                break
                
        return True
    
    def _cleanup_old_data(self, camera_id, current_time):
        """Clean up old tracking data"""
        timeout = 10.0  # Remove person data after 10 seconds of inactivity
        
        # Clean up person sequences
        inactive_persons = []
        for person_id, last_seen in self.person_timestamps[camera_id].items():
            if current_time - last_seen > timeout:
                inactive_persons.append(person_id)
        
        for person_id in inactive_persons:
            if person_id in self.person_sequences[camera_id]:
                del self.person_sequences[camera_id][person_id]
            del self.person_timestamps[camera_id][person_id]
        
        # Clean up old alerts (keep only last hour)
        alert_timeout = 3600.0
        if camera_id in self.fall_alerts:
            self.fall_alerts[camera_id] = [
                alert for alert in self.fall_alerts[camera_id]
                if current_time - alert['timestamp'] < alert_timeout
            ]
    
    def Predict(self, img):
        """Predict method to match interface with other services"""
        return self.detect(img)
    
    def detect(self, image):
        """Main detection function using original project's workflow"""
        try:
            # Detect humans in frame (detection model handles resizing internally)
            detected = self.detect_model.detect(image, need_resize=True, expand_bb=10)
            
            if detected is not None:
                logger.debug(f"Detected {len(detected)} person(s) in frame")
            else:
                logger.debug("No persons detected in frame")
            
            # Predict tracker
            self.tracker.predict()
            
            # Merge tracker predictions with detections
            for track in self.tracker.tracks:
                det = torch.tensor([track.to_tlbr().tolist() + [0.5, 1.0, 0.0]], dtype=torch.float32)
                detected = torch.cat([detected, det], dim=0) if detected is not None else det
            
            detections = []  # List of Detection objects
            
            if detected is not None:
                # Predict skeleton pose for each bbox
                poses = self.pose_model.predict(image, detected[:, 0:4], detected[:, 4])
                logger.debug(f"Extracted {len(poses)} pose(s)")
                
                # Create Detection objects  
                detections = [self.Detection(self.kpt2bbox(ps['keypoints'].numpy()),
                                           np.concatenate((ps['keypoints'].numpy(),
                                                         ps['kp_score'].numpy()), axis=1),
                                           ps['kp_score'].mean().numpy()) for ps in poses]
            
            # Update tracker
            self.tracker.update(detections)
            logger.debug(f"Tracking {len(self.tracker.tracks)} person(s)")
            
            # Predict actions for each tracked person
            results = []
            for track in self.tracker.tracks:
                if not track.is_confirmed():
                    logger.debug(f"Track {track.track_id} not confirmed yet")
                    continue
                    
                track_id = track.track_id
                frames_collected = len(track.keypoints_list)
                logger.debug(f"Track {track_id}: {frames_collected}/30 frames collected")
                
                # Need 30 frames for prediction
                if frames_collected == 30:
                    pts = np.array(track.keypoints_list, dtype=np.float32)
                    out = self.action_model.predict(pts, image.shape[:2])
                    
                    action_idx = out[0].argmax()
                    action_name = self.class_names[action_idx]
                    confidence = out[0].max()
                    
                    logger.info(f"Track {track_id} - Action: {action_name} (confidence: {confidence:.2f})")
                    
                    # Check if fall detected
                    fall_detected = (action_name == 'Fall Down')
                    if fall_detected:
                        logger.warning(f"FALL DETECTED! Track {track_id} with confidence {confidence:.2f}")
                    
                    results.append({
                        'track_id': track_id,
                        'action': action_name,
                        'confidence': float(confidence),
                        'fall_detected': fall_detected,
                        'bbox': track.to_tlbr().tolist()
                    })
            
            # Return results
            fall_detected = any(r['fall_detected'] for r in results)
            if fall_detected:
                # Return highest confidence fall detection
                fall_results = [r for r in results if r['fall_detected']]
                best_result = max(fall_results, key=lambda x: x['confidence'])
                
                # Convert pixel bbox to normalized coordinates
                bbox = best_result['bbox']  # [x1, y1, x2, y2] in pixels
                H, W = image.shape[:2]
                
                # Normalize coordinates
                x1_norm = bbox[0] / W
                y1_norm = bbox[1] / H
                x2_norm = bbox[2] / W
                y2_norm = bbox[3] / H
                
                normalized_bbox = [x1_norm, y1_norm, x2_norm, y2_norm]
                
                return best_result['confidence'], [normalized_bbox], None
            elif results:
                # Return any action detected (no bbox for non-fall actions)
                return 0.0, [], None
            else:
                # No detection
                return None, None, None
                
        except Exception as e:
            logger.error(f"Fall detection failed: {e}")
            return None, None, None
    
    def Response(self, errno, score=None, xyxyn=None):
        """Response method to match other services"""
        if score is not None:
            errno = ServiceStatus.SUCCESS.value
        elif errno is None: 
            errno = ServiceStatus.NO_OBJECT_DETECTED.value
              
        err_msg = ServiceStatus.stringify(errno)
        if errno != 0:
            logger.error(err_msg)
        else:
            logger.success(err_msg + f" - score: {score}")
        
        left, top, width, height = None, None, None, None
        if score is not None and xyxyn is not None:
            if isinstance(xyxyn, list) and len(xyxyn) > 0:
                # Handle list of bounding boxes
                bbox = xyxyn[0]  # Take first detection
                left = bbox[0]
                top = bbox[1]
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
            
        return {
            "log_id": uuid4(),
            "errno": errno,
            "err_msg": err_msg,
            "api_version": VERSION_API,
            "model_version": self.version,
            "results": [{
                "score": score,
                "location": {
                    "left": left,
                    "top": top,
                    "width": width,
                    "height": height
                }
            }]
        }
