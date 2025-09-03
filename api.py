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
          
          
class YoloClassificationService():
    def __init__(self, model_path, imgsz=224) -> None:
        self.version = "0.0.1"
        self.path = model_path
        self.imgsz = imgsz
        
        if not os.path.exists(model_path):
            logger.error(f"model path doesn't exist: {model_path}")
            exit(1)
        
        logger.debug("loading model...")
        self.model = YOLO(self.path)
        self.device = (
            torch.device('cuda') 
            if torch.cuda.is_available() 
            else torch.device('cpu')
        )
        self.model.to(self.device)
        self.is_half = (self.device.type != 'cpu')
        logger.info(f"device: {self.device.type}")
        
        logger.debug("model preheating...")
        rand_data = torch.randn(1, 3, self.imgsz, self.imgsz)
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
        
        result = results[0]
        probs = result.probs
        
        top1_class = int(probs.top1)
        top1_prob = float(probs.top1conf)
        all_probs = probs.data.cpu().numpy() if hasattr(probs.data, 'cpu') else probs.data
        top5_indices = probs.top5
        top5_confs = probs.top5conf
        top5_classes = [(int(idx), float(conf)) for idx, conf in zip(top5_indices, top5_confs)]
        
        return top1_prob, top1_class, all_probs, top5_classes
    
    
    def GetClassNames(self):
        return self.model.names if hasattr(self.model, 'names') else None



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
