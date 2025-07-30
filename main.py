# api packages
import binascii
from flask import Flask, request
from uuid import uuid4 as uuid4
import base64

# image processing
import numpy as np
import cv2

# api
from enum import Enum
from uuid import uuid4 as uuid4

# project 
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

    service = YoloDetectionService(model, imgsz)

    # router settings, no trailing slash so that:
    #   /GeneralClassifyService == /GeneralClassifyService/
    @app.route('/gesture', methods=['POST'])
    @app.route('/ponding', methods=['POST'])
    @app.route('/mouse', methods=['POST'])
    def detect():
        img, errno = validate_img_format()
        if img is None:
            return service.Response(err_no=errno)
        # inference.
        # each result represents a classification of a single image,
        # containing multiple boxes' coordinates
        score, xyxyn = service.Predict(img)
        return service.Response(
            err_no=errno,
            score=score,
            xyxyn=xyxyn
        )
    










    return app

# comment it if in production environment
# create_app(model='models/dropdoor_0_1_0.pt').run(port=8901, debug=True, host='0.0.0.0')
