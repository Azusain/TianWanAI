import binascii
import os
import sys
# git submodule.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '__SmokeFire')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '__Fall')))
from __Fall import FallDetector, ResultHandler
from __SmokeFire.smoke import SmokeFileDetector
from flask import Flask, request
from uuid import uuid4 as uuid4
import base64
import numpy as np
import cv2
from uuid import uuid4 as uuid4
from api import ServiceStatus, YoloClassificationService, YoloDetectionService, TshirtDetectionService
from loguru import logger
import threading
from flask import Flask, request, jsonify

# Global model instances (initialized at startup)
g_gesture_service = None
g_ponding_service = None
g_mouse_service = None
g_cigar_service = None
g_smoke_service = None
g_pose_service = None
g_tshirt_service = None
g_tshirt_classifier = None

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

def initialize_models():
    """Initialize all models at startup (except Fall which has special handling)"""
    global g_gesture_service, g_ponding_service, g_mouse_service, g_cigar_service
    global g_smoke_service, g_pose_service, g_tshirt_service, g_tshirt_classifier
    
    logger.info("starting model initialization...")
    
    try:
        # Initialize YOLO detection models
        logger.info("loading gesture detection model...")
        g_gesture_service = YoloDetectionService("models/gesture/weights/gesture_v2.pt", 640)
        
        logger.info("loading ponding detection model...")
        g_ponding_service = YoloDetectionService("models/ponding/weights/best.pt", 640)
        
        logger.info("loading mouse detection model...")
        g_mouse_service = YoloDetectionService("models/mouse/weights/mouse_v5.pt", 640)
        
        logger.info("loading cigar detection model...")
        g_cigar_service = YoloDetectionService("models/cigar/weights/cigar_v1.pt", 640)
        
        logger.info("loading smoke detection model...")
        g_smoke_service = SmokeFileDetector("__SmokeFire/weights/smoke.pt")
        
        # Initialize pose detection model for tshirt service
        logger.info("loading pose detection model...")
        g_pose_service = YoloDetectionService("models/yolo11m-pose.pt", 640)
        
        # Initialize tshirt service and classifier
        logger.info("loading tshirt service...")
        g_tshirt_service = TshirtDetectionService()
        
        logger.info("loading tshirt classification model...")
        g_tshirt_classifier = YoloClassificationService("models/tshirt_cls/weights/tshirt_cls_v1.pt")
        
        logger.success("all models initialized successfully!")
        
    except Exception as e:
        logger.error(f"failed to initialize models: {e}")
        raise e

# model:
#   - gesture
#   - tshirt
#   - mouse
#   - ponding
#   - smoke
#   - fall
#   - cigar
def app():
    # runtime initiaization
    # TODO: differed by model.
    log_path = f"logs/" + "runtime_{time}.log"
    logger.add(
        log_path,
        rotation="2 GB", 
        retention="7 days"
    )
    
    app = Flask(__name__)

    # Initialize all models at startup
    initialize_models()
    logger.info("server is up!")
    
    # router settings, no trailing slash so that:
    #   /GeneralClassifyService == /GeneralClassifyService/
    @app.route('/gesture', methods=['POST'])
    def GestureDetect():
        img, errno = validate_img_format()
        if img is None:
            return g_gesture_service.Response(errno=errno)
        # inference.
        score, xyxyn, _ = g_gesture_service.Predict(img)
        return g_gesture_service.Response(
            errno=errno,
            score=score,
            xyxyn=xyxyn
        )
    
    @app.route('/ponding', methods=['POST'])
    def PondingDetect():
        img, errno = validate_img_format()
        if img is None:
            return g_ponding_service.Response(errno=errno)
        # inference.
        score, xyxyn, _ = g_ponding_service.Predict(img)
        return g_ponding_service.Response(
            errno=errno,
            score=score,
            xyxyn=xyxyn
        )
    
    @app.route('/mouse', methods=['POST'])
    def MouseDetect():
        img, errno = validate_img_format()
        if img is None:
            return g_mouse_service.Response(errno=errno)
        # inference.
        score, xyxyn, _ = g_mouse_service.Predict(img)
        return g_mouse_service.Response(
            errno=errno,
            score=score,
            xyxyn=xyxyn
        )
    
    @app.route('/cigar', methods=['POST'])
    def CigarDetect():
        img, errno = validate_img_format()
        if img is None:
            return g_cigar_service.Response(errno=errno)
        # inference.
        score, xyxyn, _ = g_cigar_service.Predict(img)
        return g_cigar_service.Response(
            errno=errno,
            score=score,
            xyxyn=xyxyn
        )
    
    # Temporal Fall Detection using ST-GCN - Now follows unified design pattern
    @app.route('/fall/start', methods=['POST'])
    def StartFallDetection():
        # check args.
        req = None
        try:
            req = request.json              
        except Exception:         
            return None, ServiceStatus.INVALID_CONTENT_TYPE.value
        if not req.__contains__('rtsp_address') or req['rtsp_address'] == '':         
            return {
                "err_msg": "missing json field: rtsp_address",
            }, 400
        
        # Auto-detect device
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # spawn a new thread.
        task_id = str(uuid4())
        with FallDetector.g_tasks_lock:
            FallDetector.g_tasks[task_id] = False
        threading.Thread(
            target=FallDetector.fall_detection_task, 
            args=(task_id, req['rtsp_address'], device)
        ).start()
        logger.warning(f"spawn new worker thread for task {task_id} using device: {device}")
        return {
            "err_msg": "success",
            "task_id": task_id
        }
      
    @app.route('/fall/stop', methods=['POST'])
    def StopFallDetectionById():
        data = request.get_json()
        task_id = data.get("task_id") if data else None
        if not task_id:
            return {
                "err_msg": "missing json field: task_id"
            }, 400

        with FallDetector.g_tasks_lock:
            if (not task_id in FallDetector.g_tasks) or FallDetector.g_tasks[task_id] is None or FallDetector.g_tasks[task_id] is True:
                return {
                    "err_msg": f"task {task_id} not found"
                }, 400
            FallDetector.g_tasks[task_id] = True
            return {
                "err_msg": f"successfully stop task {task_id}"
            }
    
    # TODO: error and err_msg.
    @app.route('/fall/result', methods=['POST'])
    def GetDetectionResultById():
        data = request.get_json()
        task_id = data.get("task_id")
        # if the json field 'limit' is not set, return all result.
        limit = data.get("limit", None) 

        if not task_id:
            return jsonify({"error": "task_id is required"}), 400

        with ResultHandler.g_images_lock:
            if task_id not in ResultHandler.g_images or len(ResultHandler.g_images[task_id]) == 0:
                return jsonify({"results": []})
            
            if limit is not None:
                try:
                    limit = int(limit)
                    if limit <= 0:
                        return jsonify({"error": "limit must be positive"}), 400
                except ValueError:
                    return jsonify({"error": "limit must be an integer"}), 400

                results_list = []
                for _ in range(min(limit, len(ResultHandler.g_images[task_id]))):
                    results_list.append(ResultHandler.g_images[task_id].popleft())
            else:
                results_list = list(ResultHandler.g_images[task_id])
                ResultHandler.g_images[task_id].clear()

        resp = []
        for item in results_list:
            import cv2, base64
            _, buffer = cv2.imencode('.jpg', item["image"])
            img_b64 = base64.b64encode(buffer).decode('utf-8')
            resp.append({
                "image": img_b64,
                "results": item["results"]
            })

        return jsonify({"results": resp})
        
    @app.route('/tshirt', methods=['POST'])
    def TshirtDetect():
        img, errno = validate_img_format()
        if img is None:
            return {
              "log_id": uuid4(),
              "errno": errno,
              "err_msg": ServiceStatus.stringify(errno),
            }

        # use new pose-based detection logic
        results = g_tshirt_service.Predict(img)
        
        # log results
        for result in results:
            logger.success(f"detection score: {result['det_score']}, classification score: {result['cls_score']}")

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
      batch_results = g_smoke_service.Inference([img])
      if len(batch_results) == 0:
        errno = ServiceStatus.NO_OBJECT_DETECTED
        
      return {
        "log_id": uuid4(),
        "errno": errno,
        "err_msg": ServiceStatus.stringify(errno),
        # TODO: version fields are not used.
        "api_version": "0.0.1",
        "model_version": "0.0.1",
        "results": batch_results[0]
      }

    return app

# comment this if not testing on Windows.
if __name__ == "__main__":
  app = app()
  if app:
    app.run(port=8901, debug=True, host='0.0.0.0')
