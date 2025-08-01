# api packages
from ast import mod
import binascii
from enum import Enum
from sre_constants import SUCCESS
from flask import Flask, request
from uuid import uuid4 as uuid4
import base64

# image processing
import numpy as np
import cv2

# api
from uuid import uuid4 as uuid4

# project 
from __SmokeFire.smoke import SmokeFileDetector
from api import ServiceStatus, YoloDetectionService, TshirtDetectionService

# logger
from loguru import logger

class ModelType(Enum):
    GESTURE = "gesture"
    PONDING = "ponding"
    SMOKE = "smoke"
    TSHIRT = "tshirt"
    MOUSE = "mouse"
    FALL = "fall"

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
    return YoloDetectionService("models/gesture/weights/best.pt", 640), None
  if model_type == ModelType.PONDING.value:
    return YoloDetectionService("models/ponding/weights/best.pt", 640), None
  if model_type == ModelType.MOUSE.value:
    return YoloDetectionService("models/mouse/weights/best.pt", 640), None
  if model_type == ModelType.SMOKE .value:
    return YoloDetectionService("__SmokeFire/weights/smoke.pt", 640), None
  if model_type == ModelType.TSHIRT.value:
    return TshirtDetectionService("models/fashionpedia", 640), None
  if model_type == ModelType.FALL.value:
    return YoloDetectionService("models/fall.pt", 640), YoloDetectionService("models/yolo11s.pt", 640)

# model:
#   - gesture
#   - tshirt
#   - mouse
#   - ponding
#   - smoke
#   - fall
def create_app(model_type, imgsz=640):
    # runtime initiaization
    log_path = f"logs/{model_type}/" + "runtime_{time}.log"
    logger.add(
        log_path,
        rotation="2 GB", 
        retention="7 days"
    )
    
    app = Flask(__name__)

    service, base = get_service(model_type)
  
    # router settings, no trailing slash so that:
    #   /GeneralClassifyService == /GeneralClassifyService/
    @app.route('/gesture', methods=['POST'])
    @app.route('/ponding', methods=['POST'])
    @app.route('/mouse', methods=['POST'])
    def YoloDetect():
        img, errno = validate_img_format()
        if img is None:
            return service.Response(err_no=errno)
        # inference.
        # each result represents a classification of a single image,
        # containing multiple boxes' coordinates
        score, xyxyn, _ = service.Predict(img)
        return service.Response(
            err_no=errno,
            score=score,
            xyxyn=xyxyn
        )
    
    @app.route('/fall', methods=['POST'])
    def FallDetect():
      img, errno = validate_img_format()
      if img is None:
          return {
            "log_id": uuid4(),
            "err_no": errno,
            "err_msg": ServiceStatus.stringify(errno),
          }
  
      # inference.
      # detect persons.
      H, W = img.shape[:2]
      p_scores, p_xyxyn, p_cls = base.Predict(img)
      results = []
      for p_score, (cx, cy, w, h), pcls in zip(p_scores, p_xyxyn, p_cls):
        # index '0' stands for persons.
        if int(pcls) != 0:
            continue
        person_x1 = max(int((cx - w/2) * W), 0)
        person_y1 = max(int((cy - h/2) * H), 0)
        person_x2 = min(int((cx + w/2) * W), W - 1)
        person_y2 = min(int((cy + h/2) * H), H - 1)
        person_img = img[person_y1:person_y2, person_x1:person_x2]
        s_scores, s_xyxyn, s_cls = service.Predict(person_img)
        sw = person_x2 - person_x1
        sh = person_y2 - person_y1
        for s_score, (scx, scy, sw_, sh_), scls in zip(s_scores, s_xyxyn, s_cls):
          # TODO: check the index for class 'fall':
          if int(scls) != 0:
              continue
          # normalization.
          pixel_w = person_x2 - person_x1
          pixel_h = person_y2 - person_y1
          pixel_cx = person_x1 + pixel_w / 2
          pixel_cy = person_y1 + pixel_h / 2
          norm_cx = pixel_cx / W
          norm_cy = pixel_cy / H
          norm_w = pixel_w / W
          norm_h = pixel_h / H           
          results.append({
            "score": s_score,
            "location": {
              "left": norm_cx,
              "top": norm_cy,
              "width": norm_w,
              "height": norm_h
            }
          })
      
      # return results.
      if len(results) > 0:
        errno = ServiceStatus.SUCCESS.value
      else:
        errno = ServiceStatus.NO_OBJECT_DETECTED.value
      return {
        "log_id": uuid4(),
        "err_no": errno,
        "err_msg": ServiceStatus.stringify(errno),
        "api_version": "0.0.1",
        "model_version": "0.0.1",
        "results": results
      }

    @app.route('/tshirt', methods=['POST'])
    def TshirtDetect():
      img, errno = validate_img_format()
      if img is None:
          return {
            "log_id": uuid4(),
            "err_no": errno,
            "err_msg": ServiceStatus.stringify(errno),
          }

      H, W = img.shape[:2]
      selected_scores, selected_labels, pixel_boxes = service.Predict(img)
      results = []
      for score, label_idx, bbox in zip(selected_scores, selected_labels, pixel_boxes):
          label_name = model_type.config.id2label[label_idx.item()]
          if "t-shirt" in label_name:
              x1, y1, x2, y2 = bbox
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
                  "score": float(score),
                  "location": {
                      "left": left_n,
                      "top": top_n,
                      "width": width_n,
                      "height": height_n
                  }
              })
      
      if len(results) > 0:
        errno = ServiceStatus.SUCCESS.value
      else:
        errno = ServiceStatus.NO_OBJECT_DETECTED.value

      return {
        "log_id": uuid4(),
        "err_no": errno,
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
            "err_no": errno,
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
        "err_no": errno,
        "err_msg": ServiceStatus.stringify(errno),
        "api_version": "0.0.1",
        "model_version": "0.0.1",
        "results": batch_results[0]
      }

    return app

# comment it if in production environment
# create_app(model='models/dropdoor_0_1_0.pt').run(port=8901, debug=True, host='0.0.0.0')
