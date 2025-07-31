import os
import re

# api
from enum import Enum
from uuid import uuid4 as uuid4

# model
from ultralytics import YOLO
import torch
from transformers import YolosImageProcessor, YolosForObjectDetection


# logger
from loguru import logger

VERSION_API = '0.0.1'

class ServiceStatus(Enum):
    SUCCESS = 0
    INVALID_CONTENT_TYPE = -1
    MISSING_IMAGE_DATA = -2
    INVALID_IMAGE_FORMAT = -3
    NO_OBJECT_DETECTED = -4
    
    def stringify(err_no):
        if err_no == ServiceStatus.SUCCESS.value:
            return 'SUCCESS'
        elif err_no == ServiceStatus.INVALID_CONTENT_TYPE.value:
            return'INVALID_CONTENT_TYPE'
        elif err_no == ServiceStatus.MISSING_IMAGE_DATA.value: 
            return 'MISSING_IMAGE_DATA'    
        elif err_no == ServiceStatus.INVALID_IMAGE_FORMAT.value:
            return'INVALID_IMAGE_FORMAT'
        elif err_no == ServiceStatus.NO_OBJECT_DETECTED.value:
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
      box = results[0].boxes 
      score = None
      xyxyn = None
      # if target exists.
      if box.cls.numel() != 0:  
        score = float(box.conf)
        xyxyn = box.xyxyn
      return score, xyxyn

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
            "err_no": errno,
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
        
    def Predict(self, img):
      encoding = self.feature_extractor(images=img, return_tensors="pt")
      with torch.no_grad():
          outputs = self.model(**encoding)
      scores = outputs.logits.softmax(-1)[0, :, :-1]  
      labels = scores.argmax(-1)
      scores = scores.max(-1).values
      boxes = outputs.pred_boxes[0]
      threshold = 0.5
      selected = torch.where(scores > threshold)
      selected_scores = scores[selected]
      selected_labels = labels[selected]
      selected_boxes = boxes[selected]
      w, h = img.size
      pixel_boxes = []
      for box in selected_boxes:
          cx, cy, bw, bh = box
          x1 = (cx - bw / 2) * w
          x2 = (cx + bw / 2) * w
          y1 = (cy - bh / 2) * h
          y2 = (cy + bh / 2) * h
          pixel_boxes.append([x1.item(), y1.item(), x2.item(), y2.item()])
      return selected_scores, selected_labels, pixel_boxes

    def Response(self, errno, score=None, xyxyn=None):
        if score is not None:
            errno = ServiceStatus.SUCCESS.value
        elif errno is None: 
            errno = ServiceStatus.NO_OBJECT_DETECTED.value
              
        err_msg = ServiceStatus.stringify(errno)
        if errno != 0:
            logger.error(err_msg)
        else:
            logger.success(err_msg + f" - tshirt: {score}")
        
        left, top, width, height = None, None, None, None
        if score is not None and xyxyn is not None:
            xyxyn = xyxyn[0].tolist() 
            left    = xyxyn[0]
            top     = xyxyn[1]
            width   = xyxyn[2] - xyxyn[0]
            height  = xyxyn[3] - xyxyn[1]
            
        return {
            "log_id": uuid4(),
            "err_no": errno,
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