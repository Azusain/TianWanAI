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
    return SmokeFileDetector("__SmokeFire/weights/smoke.pt"), None
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
    
    # TODO: refactor with person detection.
    @app.route('/fall', methods=['POST'])
    def FallDetect():
        img, errno = validate_img_format()
        if img is None:
            return {
                "log_id": uuid4(),
                "errno": errno,
                "err_msg": ServiceStatus.stringify(errno),
            }

        def ensure_iterable(x):
            if x is None:
                return []
            if isinstance(x, torch.Tensor):
                return x.cpu().tolist()
            if isinstance(x, (float, int)):
                return [x]
            if hasattr(x, '__iter__'):
                return list(x)
            return [x]

        H, W = img.shape[:2]
        p_scores, p_xyxyn, p_cls = base.Predict(img)

        p_scores = ensure_iterable(p_scores)
        p_xyxyn = ensure_iterable(p_xyxyn)
        p_cls = ensure_iterable(p_cls)

        results = []
        for p_score, (cx, cy, w, h), pcls in zip(p_scores, p_xyxyn, p_cls):
            if int(pcls) != 0:
                continue

            person_x1 = max(int((cx - w / 2) * W), 0)
            person_y1 = max(int((cy - h / 2) * H), 0)
            person_x2 = min(int((cx + w / 2) * W), W - 1)
            person_y2 = min(int((cy + h / 2) * H), H - 1)

            if person_y2 <= person_y1 or person_x2 <= person_x1:
                continue  # 防止无效切片

            person_img = img[person_y1:person_y2, person_x1:person_x2]

            s_scores, s_xyxyn, s_cls = service.Predict(person_img)

            s_scores = ensure_iterable(s_scores)
            s_xyxyn = ensure_iterable(s_xyxyn)
            s_cls = ensure_iterable(s_cls)

            for s_score, (scx, scy, sw_, sh_), scls in zip(s_scores, s_xyxyn, s_cls):
                if int(scls) != 0:
                    continue

                pixel_w = person_x2 - person_x1
                pixel_h = person_y2 - person_y1
                pixel_cx = person_x1 + pixel_w / 2
                pixel_cy = person_y1 + pixel_h / 2
                norm_cx = pixel_cx / W
                norm_cy = pixel_cy / H
                norm_w = pixel_w / W
                norm_h = pixel_h / H
                logger.success(f"SUCCESS - score: {s_score}")
                results.append({
                    "score": s_score,
                    "location": {
                        "left": norm_cx,
                        "top": norm_cy,
                        "width": norm_w,
                        "height": norm_h
                    }
                })

        errno = ServiceStatus.SUCCESS.value if results else ServiceStatus.NO_OBJECT_DETECTED.value
        return {
            "log_id": uuid4(),
            "errno": errno,
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
            "errno": errno,
            "err_msg": ServiceStatus.stringify(errno),
          }

      H, W = img.shape[:2]
      selected_scores, selected_labels, pixel_boxes = service.Predict(img)
      results = []

      # 收集所有 t-shirt 检测框
      tshirt_boxes = []
      for score, label_idx, bbox in zip(selected_scores, selected_labels, pixel_boxes):
          label_name = service.model.config.id2label[label_idx.item()]
          if "t-shirt" in label_name.lower():
              tshirt_boxes.append((float(score), bbox))

      # TODO: move to utils.
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

      # 过滤重叠框，只保留高分的
      iou_threshold = 0.5
      tshirt_boxes.sort(key=lambda x: x[0], reverse=True)  # 按分数降序
      filtered_boxes = []
      for score, bbox in tshirt_boxes:
          keep = True
          for _, kept_bbox in filtered_boxes:
              if boxes_iou(bbox, kept_bbox) > 0:
                  keep = False
                  break
          if keep:
              filtered_boxes.append((score, bbox))

      # 归一化输出
      for score, bbox in filtered_boxes:
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
          logger.success(f"SUCCESS - score: {score}")
          results.append({
              "score": score,
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

# test on Windows.
# if __name__ == "__main__":
#   os.environ["MODEL"] = "smoke"
#   app = app()
#   if app:
#     app.run(port=8080, debug=True, host='0.0.0.0')
