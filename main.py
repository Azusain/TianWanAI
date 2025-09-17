import binascii
import os
import sys
# git submodule.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '__SmokeFire')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '__Fall')))
import FallDetector
import ResultHandler
import shared_state
from __SmokeFire.smoke import SmokeFileDetector
from flask import Flask, request
from uuid import uuid4 as uuid4
import base64
import numpy as np
import cv2
from uuid import uuid4 as uuid4
from api import ServiceStatus, YoloClassificationService, YoloDetectionService, TshirtDetectionService, PersonDetector
from loguru import logger
# loguru configuration is handled in api.py
import threading
from flask import Flask, request, jsonify

# Global model instances (initialized at startup)
g_unified_detection_service = None  # unified YOLO detection service for multiple classes
g_smoke_service = None
g_pose_service = None
g_tshirt_service = None
g_tshirt_classifier = None
g_person_detector = None
g_person_classifier = None

# class indices for unified detection model
# names: ['other', 'gesture', 'ponding', 'smoke', 'mouse', 'tshirt', 'cigar', 'helmet']
CLASS_INDICES = {
    'other': 0,
    'gesture': 1,
    'ponding': 2, 
    'smoke': 3,
    'mouse': 4,
    'tshirt': 5,
    'cigar': 6,
    'helmet': 7
}

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
    global g_gesture_service, g_ponding_service, g_mouse_service, g_cigar_service, g_helmet_service
    global g_smoke_service, g_pose_service, g_tshirt_service, g_tshirt_classifier, g_person_detector, g_person_classifier
    
    logger.info("starting model initialization...")
    
    try:
        # Initialize separate unified YOLO detection models for each endpoint to avoid concurrency issues
        # TODO: update with your actual unified model path
        unified_model_path = "models/unified/weights/unified_model.pt"
        
        logger.info("loading unified gesture detection model...")
        g_gesture_service = YoloDetectionService(unified_model_path, 640)
        
        logger.info("loading unified ponding detection model...")
        g_ponding_service = YoloDetectionService(unified_model_path, 640)
        
        logger.info("loading unified mouse detection model...")
        g_mouse_service = YoloDetectionService(unified_model_path, 640)
        
        logger.info("loading unified cigar detection model...")
        g_cigar_service = YoloDetectionService(unified_model_path, 640)
        
        logger.info("loading unified helmet detection model...")
        g_helmet_service = YoloDetectionService(unified_model_path, 640)
        
        logger.info("loading smoke detection model...")
        g_smoke_service = SmokeFileDetector("__SmokeFire/weights/smoke.pt")
        
        # Initialize pose detection model for tshirt service
        logger.info("loading pose detection model...")
        g_pose_service = YoloDetectionService("models/yolo11m-pose.pt", 640)
        
        # Initialize person detection model for gesture service
        logger.info("loading person detection model...")
        g_person_detector = PersonDetector()
        
        # Initialize tshirt service and classifier
        logger.info("loading tshirt service...")
        g_tshirt_service = TshirtDetectionService()
        
        logger.info("loading tshirt classification model...")
        g_tshirt_classifier = YoloClassificationService("models/tshirt_cls/weights/tshirt_cls_v2.pt")
        
        # initialize person detection model for fall detection
        logger.info("loading person detection model for fall verification...")
        g_person_classifier = YoloDetectionService("models/yolo11n.pt", 640)
        
        # set global variables for ResultHandler (using detection model instead of classification)
        ResultHandler.g_person_classifier = g_person_classifier
        ResultHandler.g_person_class_index = 0  # person class is index 0 in COCO dataset
        # also set in shared state for consistency
        shared_state.set_person_classifier(g_person_classifier, 0)
        logger.info("person detection model loaded for fall verification (COCO class 0: person)")
        
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
    # runtime initialization
    # Async logging is already configured in api.py
    
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
            return {
                "log_id": uuid4(),
                "errno": errno,
                "err_msg": ServiceStatus.stringify(errno),
                "api_version": "0.0.1",
                "model_version": "0.0.1",
                "results": []
            }
        
        persons = g_person_detector.detect_persons(img, conf_threshold=0.3)
        logger.info(f"detected {len(persons)} persons")
        
        if not persons:
            return {
                "log_id": uuid4(),
                "errno": ServiceStatus.NO_OBJECT_DETECTED.value,
                "err_msg": ServiceStatus.stringify(ServiceStatus.NO_OBJECT_DETECTED.value),
                "api_version": "0.0.1",
                "model_version": "0.0.1",
                "results": []
            }
        
        # step 2: detect gestures in each person region
        gesture_results = []
        img_height, img_width = img.shape[:2]
        
        for person_idx, person in enumerate(persons):
            person_bbox = person["bbox"]
            person_conf = person["confidence"]
            logger.info(f"processing person {person_idx}: bbox={person_bbox}, conf={person_conf:.3f}")
            
            # crop person region from image with padding (like 9.3 project)
            x1, y1, x2, y2 = person_bbox
            # add padding (10% of bbox size)
            padding_x = int((x2 - x1) * 0.1)
            padding_y = int((y2 - y1) * 0.1)
            crop_x1 = max(0, x1 - padding_x)
            crop_y1 = max(0, y1 - padding_y)
            crop_x2 = min(img_width, x2 + padding_x)
            crop_y2 = min(img_height, y2 + padding_y)
            
            # crop the person region
            person_crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
            logger.info(f"cropped person {person_idx} region: ({crop_x1},{crop_y1}) to ({crop_x2},{crop_y2})")
            
            if person_crop.size == 0:
                logger.warning(f"empty crop for person {person_idx}, skipping")
                continue
            
            # detect gestures in the cropped person region using unified model with gesture class (index 1)
            gesture_score, gesture_xyxyn, _ = g_gesture_service.Predict(person_crop, classes=[CLASS_INDICES['gesture']])
            
            if gesture_score is not None and gesture_xyxyn is not None:
                # convert relative coordinates back to original image coordinates
                if hasattr(gesture_xyxyn, 'cpu'):
                    gesture_xyxyn = gesture_xyxyn.cpu()
                if len(gesture_xyxyn) > 0:
                    # get first detection (best score)
                    rel_coords = gesture_xyxyn[0].tolist()
                    
                    # convert from relative coordinates in person region to absolute coordinates in original image
                    crop_width = crop_x2 - crop_x1
                    crop_height = crop_y2 - crop_y1
                    gesture_x1 = crop_x1 + rel_coords[0] * crop_width
                    gesture_y1 = crop_y1 + rel_coords[1] * crop_height
                    gesture_x2 = crop_x1 + rel_coords[2] * crop_width
                    gesture_y2 = crop_y1 + rel_coords[3] * crop_height
                    
                    # calculate gesture normalized coordinates
                    gesture_left_n = gesture_x1 / img_width
                    gesture_top_n = gesture_y1 / img_height
                    gesture_width_n = (gesture_x2 - gesture_x1) / img_width
                    gesture_height_n = (gesture_y2 - gesture_y1) / img_height
                    
                    gesture_results.append({
                        "score": float(gesture_score) if not isinstance(gesture_score, list) else float(gesture_score[0]),
                        "location": {
                            "left": gesture_left_n,
                            "top": gesture_top_n,
                            "width": gesture_width_n,
                            "height": gesture_height_n
                        }
                    })
                    
                    logger.info(f"gesture detected in person {person_idx} with score {gesture_score}")
                else:
                    logger.debug(f"no gesture detected in person {person_idx}")
        
        # determine response status
        if len(gesture_results) > 0:
            errno = ServiceStatus.SUCCESS.value
            logger.success(f"detected {len(gesture_results)} gesture(s) from {len(persons)} person(s)")
        else:
            errno = ServiceStatus.NO_OBJECT_DETECTED.value
            logger.debug(f"no gestures detected from {len(persons)} person(s)")
            
        return {
            "log_id": uuid4(),
            "errno": errno,
            "err_msg": ServiceStatus.stringify(errno),
            "api_version": "0.0.1",
            "model_version": "0.0.1",
            "results": gesture_results
        }
    
    @app.route('/ponding', methods=['POST'])
    def PondingDetect():
        img, errno = validate_img_format()
        if img is None:
            return g_ponding_service.Response(errno=errno)
        # inference using unified model with ponding class (index 2)
        score, xyxyn, _ = g_ponding_service.Predict(img, classes=[CLASS_INDICES['ponding']])
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
        # inference using unified model with mouse class (index 4)
        score, xyxyn, _ = g_mouse_service.Predict(img, classes=[CLASS_INDICES['mouse']])
        return g_mouse_service.Response(
            errno=errno,
            score=score,
            xyxyn=xyxyn
        )

    @app.route('/helmet', methods=['POST'])
    def HelmetDetect():
        img, errno = validate_img_format()
        if img is None:
            return {
                "log_id": uuid4(),
                "errno": errno,
                "err_msg": ServiceStatus.stringify(errno),
                "api_version": "0.0.1",
                "model_version": "0.0.1",
                "results": []
            }
        # TODO: configure threshold somewhere else. 
        # step 1: detect persons using person detector
        persons = g_person_detector.detect_persons(img, conf_threshold=0.5)
        
        if not persons:
            return {
                "log_id": uuid4(),
                "errno": ServiceStatus.NO_OBJECT_DETECTED.value,
                "err_msg": ServiceStatus.stringify(ServiceStatus.NO_OBJECT_DETECTED.value),
                "api_version": "0.0.1",
                "model_version": "0.0.1",
                "results": []
            }
        
        # step 2: detect helmets in each person region
        helmet_results = []
        img_height, img_width = img.shape[:2]
        
        for person_idx, person in enumerate(persons):
            person_bbox = person["bbox"]
            person_conf = person["confidence"]

            # crop person region from image with padding
            x1, y1, x2, y2 = person_bbox
            # add padding (10% of bbox size)
            padding_x = int((x2 - x1) * 0.1)
            padding_y = int((y2 - y1) * 0.1)
            crop_x1 = max(0, x1 - padding_x)
            crop_y1 = max(0, y1 - padding_y)
            crop_x2 = min(img_width, x2 + padding_x)
            crop_y2 = min(img_height, y2 + padding_y)
            
            # crop the person region
            person_crop = img[crop_y1:crop_y2, crop_x1:crop_x2]

            if person_crop.size == 0:
                continue
            
            # detect helmets in the cropped person region using unified model with helmet class (index 7)
            helmet_score, helmet_xyxyn, _ = g_helmet_service.Predict(person_crop, classes=[CLASS_INDICES['helmet']])
            
            # calculate person coordinates in normalized format (return person bbox, not helmet bbox)
            person_left_n = x1 / img_width
            person_top_n = y1 / img_height
            person_width_n = (x2 - x1) / img_width
            person_height_n = (y2 - y1) / img_height
            
            if helmet_score is not None and helmet_xyxyn is not None:
                # helmet detected - dangerous score = 1 - helmet_confidence (lower helmet conf = higher danger)
                helmet_confidence = float(helmet_score) if not isinstance(helmet_score, list) else float(helmet_score[0])
                dangerous_score = 1.0 - helmet_confidence  # higher helmet confidence = lower danger
                
                helmet_results.append({
                    "score": dangerous_score,
                    "location": {
                        "left": person_left_n,
                        "top": person_top_n,
                        "width": person_width_n,
                        "height": person_height_n
                    }
                })
                

            else:
                # no helmet detected - maximum danger (dangerous score = 1.0)
                helmet_results.append({
                    "score": 1.0,  # maximum dangerous score when no helmet detected
                    "location": {
                        "left": person_left_n,
                        "top": person_top_n,
                        "width": person_width_n,
                        "height": person_height_n
                    }
                })
                
        
        # determine response status
        if len(helmet_results) > 0:
            errno = ServiceStatus.SUCCESS.value
        else:
            errno = ServiceStatus.NO_OBJECT_DETECTED.value
            
        return {
            "log_id": uuid4(),
            "errno": errno,
            "err_msg": ServiceStatus.stringify(errno),
            "api_version": "0.0.1",
            "model_version": "0.0.1",
            "results": helmet_results
        }
    
    @app.route('/cigar', methods=['POST'])
    def CigarDetect():
        img, errno = validate_img_format()
        if img is None:
            return g_cigar_service.Response(errno=errno)
        # inference using unified model with cigar class (index 6)
        score, xyxyn, _ = g_cigar_service.Predict(img, classes=[CLASS_INDICES['cigar']])
        return g_cigar_service.Response(
            errno=errno,
            score=score,
            xyxyn=xyxyn
        )
    
    # Temporal Fall Detection using ST-GCN + Alpha Pose + Tiny YOLO.
    @app.route('/fall/start', methods=['POST'])
    def StartFallDetection():
        # check args.
        req = None
        try:
            req = request.json              
        except Exception:         
            return {}, 400
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

        # use shared state instead of module-local variables
        g_images = shared_state.get_images_dict()
        g_images_lock = shared_state.get_images_lock()
        
        logger.info(f"querying results for task_id: {task_id}")
        
        with g_images_lock:
            logger.info(f"available task IDs in shared g_images: {list(g_images.keys())}")
            logger.info(f"available task IDs in ResultHandler.g_images: {list(ResultHandler.g_images.keys())}")
            
            # try both shared state and module state for compatibility
            if task_id in g_images and len(g_images[task_id]) > 0:
                logger.info(f"found {len(g_images[task_id])} results in shared state for task_id {task_id}")
                images_dict = g_images
            elif task_id in ResultHandler.g_images and len(ResultHandler.g_images[task_id]) > 0:
                logger.info(f"found {len(ResultHandler.g_images[task_id])} results in module state for task_id {task_id}")
                images_dict = ResultHandler.g_images
            else:
                logger.warning(f"result no found for {task_id} in either shared or module state")
                return jsonify({"results": []})
            
            if limit is not None:
                try:
                    limit = int(limit)
                    if limit <= 0:
                        return jsonify({"error": "limit must be positive"}), 400
                except ValueError:
                    return jsonify({"error": "limit must be an integer"}), 400

                results_list = []
                for _ in range(min(limit, len(images_dict[task_id]))):
                    results_list.append(images_dict[task_id].popleft())
            else:
                results_list = list(images_dict[task_id])
                images_dict[task_id].clear()

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
        results = g_tshirt_service.Predict(img, g_pose_service, g_tshirt_classifier)
        
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
