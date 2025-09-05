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

