# api packages
import binascii
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
from api import ServiceStatus, YoloDetectionService

# logger
from loguru import logger

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

# model:
#   - gesture
#   - tshirt
#   - mouse
#   - ponding
#   - smoke
#   - fall
def create_app(model, imgsz=640):
    # runtime initiaization
    model_name = model.split('/')[-1].split('.')[0]
    log_path = f"logs/{model_name}/" + "runtime_{time}.log"
    logger.add(
        log_path,
        rotation="2 GB", 
        retention="7 days"
    )
    
    app = Flask(__name__)

    yolo_service = YoloDetectionService(model, imgsz)
    smoke_service = SmokeFileDetector(model='__SmokeFire/weights/smoke.pt')
    
  
    # router settings, no trailing slash so that:
    #   /GeneralClassifyService == /GeneralClassifyService/
    @app.route('/gesture', methods=['POST'])
    @app.route('/ponding', methods=['POST'])
    @app.route('/mouse', methods=['POST'])
    def YoloDetect():
        img, errno = validate_img_format()
        if img is None:
            return yolo_service.Response(err_no=errno)
        # inference.
        # each result represents a classification of a single image,
        # containing multiple boxes' coordinates
        score, xyxyn = yolo_service.Predict(img)
        return yolo_service.Response(
            err_no=errno,
            score=score,
            xyxyn=xyxyn
        )
    
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
      selected_scores, selected_labels, pixel_boxes = yolo_service.Predict(img)
      results = []
      for score, label_idx, bbox in zip(selected_scores, selected_labels, pixel_boxes):
          label_name = model.config.id2label[label_idx.item()]
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
      batch_results = smoke_service.Inference([img])
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
