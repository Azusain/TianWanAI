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
from api import ServiceStatus, GestureService

# logger
from loguru import logger

# return image and errno.
def validate_img_format():
    req = None
    try:
        req = request.json              
    except Exception:         
        return None, ServiceStatus.INVALID_CONTENT_TYPE.value
        
    # check if the json data contains certain key: 'image'
    if not req.__contains__('image') or req['image'] == '':         
        return None, ServiceStatus.MISSING_IMAGE_DATA.value
    
    # check whether the image data has a valid format 
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

def get_service(model_name, imgsz):
    if model_name == "gesture":
        return GestureService(
        model_path="", 
        imgsz=imgsz
    )

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

    service = get_service(model, imgsz)

    # router settings, no trailing slash so that:
    #   /GeneralClassifyService == /GeneralClassifyService/
    @app.route('/Gesture', methods=['POST'])
    def detect():
        img, errno = validate_img_format()
        if img is None:
            service.Response(err_no=errno)

        # inference 
        results = service.Predict(img)  # each result represents a classification of a single image,
        box = results[0].boxes          # containing multiple boxes' coordinates
        score = None
        xyxyn = None
        if box.cls.numel() != 0:  # if target exists.
            score = float(box.conf)
            xyxyn = box.xyxyn

        # send results back       
        if score is not None:
            err_no = ServiceStatus.SUCCESS.value
        else: 
            err_no = ServiceStatus.NO_OBJECT_DETECTED.value                
        return service.Response(
            err_no=err_no,
            score=score,
            xyxyn=xyxyn
        )










    return app

# comment it if in production environment
# create_app(model='models/dropdoor_0_1_0.pt').run(port=8901, debug=True, host='0.0.0.0')
