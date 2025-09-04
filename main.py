# api packages
import binascii
from enum import Enum
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '__SmokeFire')))

from __SmokeFire.smoke import SmokeFileDetector
import torch

from flask import Flask, request
from uuid import uuid4 as uuid4
import base64

# image processing
import numpy as np
import cv2

# api
from uuid import uuid4 as uuid4

# project 
from api import ServiceStatus, YoloClassificationService, YoloDetectionService, TshirtDetectionService, TemporalFallDetectionService

# logger
from loguru import logger


class ModelType(Enum):
    GESTURE = "gesture"
    PONDING = "ponding"
    SMOKE = "smoke"
    TSHIRT = "tshirt"
    MOUSE = "mouse"
    FALL = "fall"
    CIGAR = "cigar"

# return image and errno.
def validate_img_format():
    req = None
    try:
        req = request.json              
    except Exception:         
        return None, ServiceStatus.INVALID_CONTENT_TYPE.value
        
    # check if the json data contains certain key: 'image'.
    if not req.__contains__('image') or req['image'] == '':         
        return None, ServiceStatus.MISSING_IMAGE_DATA.value
    
    # check whether the image data has a valid format.
    bin_data = None
    try:
        if type(req['image']) is not str:
            raise binascii.Error
        bin_data = base64.b64decode(req['image'])     
        np_arr = np.frombuffer(bin_data, np.int8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)   
        if img is None:
            raise binascii.Error
        return img, None
    except binascii.Error: 
        return None, ServiceStatus.INVALID_IMAGE_FORMAT.value

# TODO: move this to configuration file.
def get_service(model_type: str):
  if model_type == ModelType.GESTURE.value:
    return YoloDetectionService("models/gesture/weights/gesture_v2.pt", 640), None
  if model_type == ModelType.PONDING.value:
    return YoloDetectionService("models/ponding/weights/best.pt", 640), None
  if model_type == ModelType.MOUSE.value:
    return YoloDetectionService("models/mouse/weights/mouse_v4.pt", 640), None
  if model_type == ModelType.CIGAR.value:
    return YoloDetectionService("models/cigar/weights/cigar_v1.pt"), None
  if model_type == ModelType.SMOKE .value:
    return SmokeFileDetector("__SmokeFire/weights/smoke.pt"), None
  if model_type == ModelType.TSHIRT.value:
    return TshirtDetectionService("models/fashionpedia", 640), YoloClassificationService("models/tshirt_cls/weights/tshirt_cls_v1.pt")
  if model_type == ModelType.FALL.value:
    return TemporalFallDetectionService(), None

# model:
#   - gesture
#   - tshirt
#   - mouse
#   - ponding
#   - smoke
#   - fall
#   - cigar
def app():
    model_type = os.environ.get('MODEL')
    if model_type:
      print(f"Using model: {model_type}")
    else:
      print("Environment variable `MODEL` not set")
      return 
  
    # runtime initiaization
    log_path = f"logs/{model_type}/" + "runtime_{time}.log"
    logger.add(
        log_path,
        rotation="2 GB", 
        retention="7 days"
    )
    
    app = Flask(__name__)

    service, base = get_service(model_type)
    logger.info("server is up!")
    
    # router settings, no trailing slash so that:
    #   /GeneralClassifyService == /GeneralClassifyService/
    @app.route('/gesture', methods=['POST'])
    @app.route('/ponding', methods=['POST'])
    @app.route('/mouse', methods=['POST'])
    @app.route('/cigar', methods=['POST'])
    def YoloDetect():
        img, errno = validate_img_format()
        if img is None:
            return service.Response(errno=errno)
        # inference.
        # each result represents a classification of a single image,
        # containing multiple boxes' coordinates
        score, xyxyn, _ = service.Predict(img)
        return service.Response(
            errno=errno,
            score=score,
            xyxyn=xyxyn
        )
    
    # Temporal Fall Detection using ST-GCN - Now follows unified design pattern
    @app.route('/fall', methods=['POST'])
    def FallDetect():
        img, errno = validate_img_format()
        if img is None:
            return service.Response(errno=errno)
        
        # Use the same pattern as other detection services
        # The detect method handles the full pipeline
        score, xyxyn, _ = service.Predict(img)
        
        return service.Response(
            errno=errno,
            score=score,
            xyxyn=xyxyn
        )

    @app.route('/tshirt', methods=['POST'])
    def TshirtDetect():
        img, errno = validate_img_format()
        if img is None:
            return {
              "log_id": uuid4(),
              "errno": errno,
              "err_msg": ServiceStatus.stringify(errno),
            }

        H, W = img.shape[:2]
        selected_scores, selected_labels, pixel_boxes = service.Predict(img)
        results = []

        tshirt_boxes = []
        for score, label_idx, bbox in zip(selected_scores, selected_labels, pixel_boxes):
            label_name = service.model.config.id2label[label_idx.item()]
            if "t-shirt" in label_name.lower():
                tshirt_boxes.append((float(score), bbox))

        def boxes_iou(box1, box2):
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            inter_area = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union_area = area1 + area2 - inter_area
            if union_area == 0:
                return 0
            return inter_area / union_area

        tshirt_boxes.sort(key=lambda x: x[0], reverse=True)
        filtered_boxes = []
        for score, bbox in tshirt_boxes:
            keep = True
            for _, kept_bbox in filtered_boxes:
                if boxes_iou(bbox, kept_bbox) > 0:
                    keep = False
                    break
            if keep:
                filtered_boxes.append((score, bbox))

        for score, bbox in filtered_boxes:
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(W, x2)
            y2 = min(H, y2)
            
            if x2 > x1 and y2 > y1:
                cropped_region = img[y1:y2, x1:x2]
                top1_prob, top1_class, _, _ = base.Predict(cropped_region)
                width_px = x2 - x1
                height_px = y2 - y1
                cx = x1 + width_px / 2
                cy = y1 + height_px / 2
                cxn = cx / W
                cyn = cy / H
                width_n = width_px / W
                height_n = height_px / H
                left_n = cxn - width_n / 2
                top_n = cyn - height_n / 2
                                
                results.append({
                    "det_score": score,
                    # index '1' stands for 'tshirt'
                    "cls_score": top1_prob if top1_class == 1 else 1 - top1_prob,
                    "location": {
                        "left": left_n,
                        "top": top_n,
                        "width": width_n,
                        "height": height_n
                    }
                })
                logger.success(f"detection score: {score}, classification score: {top1_prob}")


        if len(results) > 0:
          errno = ServiceStatus.SUCCESS.value
        else:
          errno = ServiceStatus.NO_OBJECT_DETECTED.value
        return {
          "log_id": uuid4(),
          "errno": errno,
          "err_msg": ServiceStatus.stringify(errno),
          "api_version": "0.0.1",
          "model_version": "0.0.1",
          "results": results
        }

    @app.route('/smoke', methods=['POST'])
    def SmokeDetect():
      img, errno = validate_img_format()
      if img is None:
          return {
            "log_id": uuid4(),
            "errno": errno,
            "err_msg": ServiceStatus.stringify(errno),
          }
      
      # inference.
      errno = ServiceStatus.SUCCESS.value
      # input cv2 format image.
      batch_results = service.Inference([img])
      if len(batch_results) == 0:
        errno = ServiceStatus.NO_OBJECT_DETECTED
        
      return {
        "log_id": uuid4(),
        "errno": errno,
        "err_msg": ServiceStatus.stringify(errno),
        "api_version": "0.0.1",
        "model_version": "0.0.1",
        "results": batch_results[0]
      }

    return app

# comment this if not testing on Windows.
if __name__ == "__main__":
  os.environ["MODEL"] = "fall"
  app = app()
  if app:
    app.run(port=8091, debug=True, host='0.0.0.0')
