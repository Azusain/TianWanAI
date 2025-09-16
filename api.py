import os
import re

# api
from enum import Enum
from uuid import uuid4 as uuid4

# model
from ultralytics import YOLO
import torch

# temporal fall detection - moved to this file
import sys
import time
from collections import defaultdict, deque
import numpy as np
import cv2

# logger  
from loguru import logger
import sys

# Configure async logging
logger.remove()  # Remove default handler
logger.add(
    "logs/runtime_{time}.log",
    rotation="2 GB",
    retention="7 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    enqueue=True,  # Enable async logging
    backtrace=True,
    diagnose=True
)
# Keep console output with proper colors
logger.add(
    sys.stderr,
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}:{function}:{line}</cyan> | {message}",
    level="DEBUG",
    enqueue=True
)

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
        
    def Predict(self, img, classes=None):
      results = self.model.predict(
          source=img, 
          imgsz=self.imgsz,
          half=self.is_half,
          verbose=False,
          classes=classes  # filter to specific class indices if provided
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
    def __init__(self) -> None:
        self.version = "0.0.1"
        
    def Predict(self, img, pose_service, classifier_service):
        """detect persons using pose model, extract upper body, classify tshirt"""
        # step 1: detect persons using pose model
        pose_results = pose_service.model.predict(
            source=img,
            imgsz=640,
            verbose=False,
            # person bbox conf
            conf=0.6
        )
        
        result = pose_results[0]
        persons_results = []
        
        if result.keypoints is not None and result.boxes is not None:
            keypoints = result.keypoints.data  # [N, 17, 3] for COCO format
            boxes = result.boxes.data  # [N, 6] - xyxy + conf + cls
            H, W = img.shape[:2]
            
            for i in range(len(keypoints)):
                person_keypoints = keypoints[i].cpu().numpy()  # [17, 3]
                person_box = boxes[i].cpu().numpy()  # [6]
                
                # extract shoulder and hip keypoints (COCO format)
                left_shoulder = person_keypoints[5]   # [x, y, conf]
                right_shoulder = person_keypoints[6]  # [x, y, conf] 
                left_hip = person_keypoints[11]       # [x, y, conf]
                right_hip = person_keypoints[12]      # [x, y, conf]
                
                # find the highest and lowest points among shoulders and hips
                # this handles both normal and inverted poses correctly
                all_y_coords = [left_shoulder[1], right_shoulder[1], left_hip[1], right_hip[1]]
                upper_y = min(all_y_coords)  # highest point (smallest Y coordinate)
                lower_y = max(all_y_coords)  # lowest point (largest Y coordinate)
                
                # use person bbox for x boundaries  
                x1, y1, x2, y2 = person_box[:4]
                
                # calculate upper body region with margin
                margin_y = int((lower_y - upper_y) * 0.1)
                crop_upper_y = max(0, int(upper_y - margin_y))
                crop_lower_y = min(H, int(lower_y + margin_y))
                
                x1, y1, x2, y2 = int(x1), int(crop_upper_y), int(x2), int(crop_lower_y)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(W, x2)
                y2 = min(H, y2)
                
                if x2 > x1 and y2 > y1:
                    # check aspect ratio to filter out abnormal detections
                    width_px = x2 - x1
                    height_px = y2 - y1
                    aspect_ratio = width_px / height_px
                    
                    # filter out extreme aspect ratios (too wide or too tall)
                    # normal human upper body should have reasonable proportions
                    if aspect_ratio > 2.0 or aspect_ratio < 0.5:
                        logger.debug(f"skipping detection with abnormal aspect ratio: {aspect_ratio:.2f} (width={width_px}, height={height_px})")
                        continue
                    
                    # crop upper body region
                    upper_body_region = img[y1:y2, x1:x2]
                    
                    # classify tshirt
                    top1_prob, top1_class, _, _ = classifier_service.Predict(upper_body_region)
                    
                    # calculate normalized coordinates
                    cx = x1 + width_px / 2
                    cy = y1 + height_px / 2
                    cxn = cx / W
                    cyn = cy / H
                    width_n = width_px / W
                    height_n = height_px / H
                    left_n = cxn - width_n / 2
                    top_n = cyn - height_n / 2
                    
                    persons_results.append({
                        "det_score": float(person_box[4]),  # person detection confidence
                        "cls_score": top1_prob if top1_class == 1 else 0.0,  # only output confidence when tshirt is detected
                        "location": {
                            "left": left_n,
                            "top": top_n,
                            "width": width_n,
                            "height": height_n
                        }
                    })
                    
        return persons_results


class PersonDetector:
    def __init__(self):
        self.model = None
        self.device = (
            torch.device('cuda') 
            if torch.cuda.is_available() 
            else torch.device('cpu')
        )
        self._load_model()
    
    def _load_model(self):
        try:
            model_path = "models/yolo11s.pt"
            if not os.path.exists(model_path):
                logger.error(f"person detection model not found: {model_path}")
                raise FileNotFoundError(f"model file not found: {model_path}")
                
            logger.info(f"loading person detection model: {model_path}")
            self.model = YOLO(model_path)
            
            # move model to appropriate device
            self.model.to(self.device)
            logger.info(f"person detection model loaded successfully on {self.device.type}")
        except Exception as e:
            logger.error(f"failed to load person detection model: {e}")
            raise e
    
    def detect_persons(self, img, conf_threshold=0.5):
        """detect persons in image, returns list of person bounding boxes"""
        try:
            # only detect person class (class 0 in COCO dataset)
            results = self.model.predict(
                img, 
                conf=conf_threshold, 
                classes=[0],  # only person class
                verbose=False
            )
            persons = []
            
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        persons.append({
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "confidence": float(conf)
                        })
            
            return persons
        except Exception as e:
            logger.error(f"person detection error: {e}")
            return []

