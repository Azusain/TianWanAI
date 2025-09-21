import grpc
from concurrent import futures
import threading
from uuid import uuid4
import base64
import cv2
import numpy as np
from loguru import logger
import sys
import os

# Import generated gRPC code
sys.path.append('generated-python')
from api.v1 import model_service_pb2
from api.v1 import model_service_pb2_grpc

# Import existing service classes
from api import (
    ServiceStatus, YoloClassificationService, YoloDetectionService, 
    TshirtDetectionService, PersonDetector
)
from __SmokeFire.smoke import SmokeFileDetector
from __Fall import FallDetector, ResultHandler
import shared_state

class ModelInferenceServicer(model_service_pb2_grpc.ModelInferenceServiceServicer):
    def __init__(self):
        self.sessions = {}  # session_id -> session_data
        self.session_lock = threading.Lock()
        
        # Initialize models (same as in main.py)
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all models at startup"""
        logger.info("initializing models for grpc server...")
        
        try:
            unified_model_path = "models/all_v1.pt"
            
            # Initialize detection services
            logger.info("loading unified detection models...")
            self.g_gesture_service = YoloDetectionService(unified_model_path, 640)
            self.g_ponding_service = YoloDetectionService(unified_model_path, 640)
            self.g_mouse_service = YoloDetectionService(unified_model_path, 640)
            self.g_cigar_service = YoloDetectionService(unified_model_path, 640)
            self.g_helmet_service = YoloDetectionService(unified_model_path, 640)
            
            # Smoke detection
            logger.info("loading smoke detection model...")
            self.g_smoke_service = SmokeFileDetector("__SmokeFire/weights/smoke.pt")
            
            # Pose and person detection
            logger.info("loading pose and person detection models...")
            self.g_pose_service = YoloDetectionService("models/yolo11m-pose.pt", 640)
            self.g_person_detector = PersonDetector()
            
            # Tshirt service and classifier
            logger.info("loading tshirt service...")
            self.g_tshirt_service = TshirtDetectionService()
            self.g_tshirt_classifier = YoloClassificationService("models/tshirt_cls/weights/tshirt_cls_v2.pt")
            
            # Person classifier for fall detection
            logger.info("loading person classifier for fall detection...")
            self.g_person_classifier = YoloDetectionService("models/yolo11n.pt", 640)
            
            # Set global variables for ResultHandler
            ResultHandler.g_person_classifier = self.g_person_classifier
            ResultHandler.g_person_class_index = 0
            shared_state.set_person_classifier(self.g_person_classifier, 0)
            
            # Class indices mapping
            self.CLASS_INDICES = {
                'other': 0, 'gesture': 1, 'ponding': 2, 'smoke': 3,
                'mouse': 4, 'tshirt': 5, 'cigar': 6, 'helmet': 7
            }
            
            logger.success("all models initialized successfully for grpc server!")
            
        except Exception as e:
            logger.error(f"failed to initialize models: {e}")
            raise e
    
    def ListModels(self, request, context):
        """Return list of available models"""
        models = [
            model_service_pb2.ModelInfo(
                type=model_service_pb2.MODEL_TYPE_GESTURE,
                name="Gesture Detection",
                description="Detect hand gestures in images",
                version="0.0.1"
            ),
            model_service_pb2.ModelInfo(
                type=model_service_pb2.MODEL_TYPE_PONDING,
                name="Ponding Detection", 
                description="Detect water ponding in images",
                version="0.0.1"
            ),
            model_service_pb2.ModelInfo(
                type=model_service_pb2.MODEL_TYPE_SMOKE,
                name="Smoke Detection",
                description="Detect smoke in images",
                version="0.0.1"
            ),
            model_service_pb2.ModelInfo(
                type=model_service_pb2.MODEL_TYPE_MOUSE,
                name="Mouse Detection",
                description="Detect mice in images", 
                version="0.0.1"
            ),
            model_service_pb2.ModelInfo(
                type=model_service_pb2.MODEL_TYPE_TSHIRT,
                name="T-shirt Detection",
                description="Detect inappropriate clothing",
                version="0.0.1"
            ),
            model_service_pb2.ModelInfo(
                type=model_service_pb2.MODEL_TYPE_CIGAR,
                name="Cigar Detection",
                description="Detect cigarettes in images",
                version="0.0.1"
            ),
            model_service_pb2.ModelInfo(
                type=model_service_pb2.MODEL_TYPE_HELMET,
                name="Helmet Detection",
                description="Detect safety helmets",
                version="0.0.1"
            ),
            model_service_pb2.ModelInfo(
                type=model_service_pb2.MODEL_TYPE_FALL,
                name="Fall Detection",
                description="Detect person falls (stateful)",
                version="0.0.1"
            )
        ]
        
        return model_service_pb2.ListModelsResponse(
            models=models,
            server_version="1.0.0"
        )
    
    def Inference(self, request_iterator, context):
        """Handle bidirectional streaming inference"""
        session_id = None
        model_type = None
        
        try:
            for request in request_iterator:
                session_id = request.session_id
                
                if request.HasField('init'):
                    # Handle stream initialization
                    model_type = request.init.model_type
                    logger.info(f"initializing session {session_id} for model {model_type}")
                    
                    with self.session_lock:
                        self.sessions[session_id] = {
                            'model_type': model_type,
                            'parameters': dict(request.init.parameters)
                        }
                    
                    # Send init response
                    yield model_service_pb2.InferenceResponse(
                        init=model_service_pb2.StreamInitResponse(
                            session_id=session_id,
                            status=model_service_pb2.SERVICE_STATUS_SUCCESS
                        )
                    )
                    
                elif request.HasField('frame'):
                    # Handle frame inference
                    if session_id not in self.sessions:
                        yield model_service_pb2.InferenceResponse(
                            error=model_service_pb2.StreamErrorResponse(
                                task_id=session_id,
                                status=model_service_pb2.SERVICE_STATUS_STREAM_ERROR,
                                error_message="session not initialized"
                            )
                        )
                        continue
                    
                    model_type = self.sessions[session_id]['model_type']
                    image_data = request.frame.image_data
                    
                    # Decode image
                    try:
                        np_arr = np.frombuffer(image_data, np.uint8)
                        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        if img is None:
                            raise ValueError("invalid image data")
                    except Exception as e:
                        yield model_service_pb2.InferenceResponse(
                            error=model_service_pb2.StreamErrorResponse(
                                task_id=session_id,
                                status=model_service_pb2.SERVICE_STATUS_INVALID_IMAGE_FORMAT,
                                error_message=str(e)
                            )
                        )
                        continue
                    
                    # Process based on model type
                    results = self._process_frame(img, model_type, session_id)
                    
                    # Send detection response
                    yield model_service_pb2.InferenceResponse(
                        detection=results
                    )
                    
                elif request.HasField('close'):
                    # Handle stream close
                    logger.info(f"closing session {session_id}")
                    with self.session_lock:
                        if session_id in self.sessions:
                            del self.sessions[session_id]
                    
                    yield model_service_pb2.InferenceResponse(
                        close=model_service_pb2.StreamCloseResponse(
                            task_id=session_id,
                            message="session closed"
                        )
                    )
                    break
                    
        except Exception as e:
            logger.error(f"error in streaming inference: {e}")
            yield model_service_pb2.InferenceResponse(
                error=model_service_pb2.StreamErrorResponse(
                    task_id=session_id or "unknown",
                    status=model_service_pb2.SERVICE_STATUS_STREAM_ERROR,
                    error_message=str(e)
                )
            )
    
    def _process_frame(self, img, model_type, session_id):
        """Process a single frame based on model type"""
        try:
            if model_type == model_service_pb2.MODEL_TYPE_GESTURE:
                return self._process_gesture(img, session_id)
            elif model_type == model_service_pb2.MODEL_TYPE_PONDING:
                return self._process_ponding(img, session_id)
            elif model_type == model_service_pb2.MODEL_TYPE_SMOKE:
                return self._process_smoke(img, session_id)
            elif model_type == model_service_pb2.MODEL_TYPE_MOUSE:
                return self._process_mouse(img, session_id)
            elif model_type == model_service_pb2.MODEL_TYPE_TSHIRT:
                return self._process_tshirt(img, session_id)
            elif model_type == model_service_pb2.MODEL_TYPE_CIGAR:
                return self._process_cigar(img, session_id)
            elif model_type == model_service_pb2.MODEL_TYPE_HELMET:
                return self._process_helmet(img, session_id)
            elif model_type == model_service_pb2.MODEL_TYPE_FALL:
                return self._process_fall(img, session_id)
            else:
                return model_service_pb2.DetectionResponse(
                    task_id=session_id,
                    status=model_service_pb2.SERVICE_STATUS_MODEL_NOT_FOUND,
                    message="unsupported model type",
                    results=[],
                    timestamp=0,
                    api_version="1.0.0",
                    model_version="0.0.1"
                )
        except Exception as e:
            logger.error(f"error processing frame: {e}")
            return model_service_pb2.DetectionResponse(
                task_id=session_id,
                status=model_service_pb2.SERVICE_STATUS_STREAM_ERROR,
                message=str(e),
                results=[],
                timestamp=0,
                api_version="1.0.0",
                model_version="0.0.1"
            )
    
    def _process_gesture(self, img, session_id):
        """Process gesture detection"""
        persons = self.g_person_detector.detect_persons(img, conf_threshold=0.3)
        
        if not persons:
            return model_service_pb2.DetectionResponse(
                task_id=session_id,
                status=model_service_pb2.SERVICE_STATUS_NO_OBJECT_DETECTED,
                message="no persons detected",
                results=[],
                timestamp=0,
                api_version="1.0.0",
                model_version="0.0.1"
            )
        
        gesture_results = []
        img_height, img_width = img.shape[:2]
        
        for person in persons:
            person_bbox = person["bbox"]
            x1, y1, x2, y2 = person_bbox
            
            # Add padding
            padding_x = int((x2 - x1) * 0.1)
            padding_y = int((y2 - y1) * 0.1)
            crop_x1 = max(0, x1 - padding_x)
            crop_y1 = max(0, y1 - padding_y)
            crop_x2 = min(img_width, x2 + padding_x)
            crop_y2 = min(img_height, y2 + padding_y)
            
            person_crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
            if person_crop.size == 0:
                continue
            
            # Detect gestures in person region
            gesture_score, gesture_xyxyn, _ = self.g_gesture_service.Predict(
                person_crop, classes=[self.CLASS_INDICES['gesture']]
            )
            
            if gesture_score is not None and gesture_xyxyn is not None:
                if hasattr(gesture_xyxyn, 'cpu'):
                    gesture_xyxyn = gesture_xyxyn.cpu()
                if len(gesture_xyxyn) > 0:
                    rel_coords = gesture_xyxyn[0].tolist()
                    
                    # Convert to absolute coordinates
                    crop_width = crop_x2 - crop_x1
                    crop_height = crop_y2 - crop_y1
                    gesture_x1 = crop_x1 + rel_coords[0] * crop_width
                    gesture_y1 = crop_y1 + rel_coords[1] * crop_height
                    gesture_x2 = crop_x1 + rel_coords[2] * crop_width
                    gesture_y2 = crop_y1 + rel_coords[3] * crop_height
                    
                    # Normalize
                    gesture_left_n = gesture_x1 / img_width
                    gesture_top_n = gesture_y1 / img_height
                    gesture_width_n = (gesture_x2 - gesture_x1) / img_width
                    gesture_height_n = (gesture_y2 - gesture_y1) / img_height
                    
                    gesture_results.append(
                        model_service_pb2.DetectionResult(
                            score=float(gesture_score) if not isinstance(gesture_score, list) else float(gesture_score[0]),
                            location=model_service_pb2.Location(
                                left=gesture_left_n,
                                top=gesture_top_n,
                                width=gesture_width_n,
                                height=gesture_height_n
                            )
                        )
                    )
        
        status = (model_service_pb2.SERVICE_STATUS_SUCCESS if gesture_results 
                 else model_service_pb2.SERVICE_STATUS_NO_OBJECT_DETECTED)
        
        return model_service_pb2.DetectionResponse(
            task_id=session_id,
            status=status,
            message="gesture detection completed",
            results=gesture_results,
            timestamp=0,
            api_version="1.0.0",
            model_version="0.0.1"
        )
    
    def _process_ponding(self, img, session_id):
        """Process ponding detection"""
        score, xyxyn, _ = self.g_ponding_service.Predict(img, classes=[self.CLASS_INDICES['ponding']])
        return self._create_simple_detection_response(score, xyxyn, session_id, "ponding detection")
    
    def _process_mouse(self, img, session_id):
        """Process mouse detection"""
        score, xyxyn, _ = self.g_mouse_service.Predict(img, classes=[self.CLASS_INDICES['mouse']])
        return self._create_simple_detection_response(score, xyxyn, session_id, "mouse detection")
    
    def _process_cigar(self, img, session_id):
        """Process cigar detection"""
        score, xyxyn, _ = self.g_cigar_service.Predict(img, classes=[self.CLASS_INDICES['cigar']])
        return self._create_simple_detection_response(score, xyxyn, session_id, "cigar detection")
    
    def _create_simple_detection_response(self, score, xyxyn, session_id, description):
        """Helper method to create simple detection response"""
        results = []
        
        if score is not None and xyxyn is not None:
            # Convert tensor to native Python types
            if hasattr(score, 'cpu'):
                score = score.cpu()
            if hasattr(score, 'numpy'):
                score = score.numpy()
            if hasattr(score, 'tolist'):
                score = score.tolist()
            if isinstance(score, (list, np.ndarray)) and len(score) == 1:
                score = float(score[0])
            elif not isinstance(score, (float, int)):
                score = float(score)
                
            if hasattr(xyxyn, 'cpu'):
                xyxyn = xyxyn.cpu()
            
            if hasattr(xyxyn[0], 'tolist'):
                coords = xyxyn[0].tolist()
            else:
                coords = xyxyn[0]
            
            left = float(coords[0])
            top = float(coords[1])
            width = float(coords[2] - coords[0])
            height = float(coords[3] - coords[1])
            
            results.append(
                model_service_pb2.DetectionResult(
                    score=score,
                    location=model_service_pb2.Location(
                        left=left,
                        top=top,
                        width=width,
                        height=height
                    )
                )
            )
        
        status = (model_service_pb2.SERVICE_STATUS_SUCCESS if results 
                 else model_service_pb2.SERVICE_STATUS_NO_OBJECT_DETECTED)
        
        return model_service_pb2.DetectionResponse(
            task_id=session_id,
            status=status,
            message=f"{description} completed",
            results=results,
            timestamp=0,
            api_version="1.0.0",
            model_version="0.0.1"
        )
    
    # TODO: Implement other model processing methods
    def _process_smoke(self, img, session_id):
        """Process smoke detection - placeholder"""
        # TODO: Implement smoke detection logic
        return self._create_simple_detection_response(None, None, session_id, "smoke detection")
    
    def _process_tshirt(self, img, session_id):
        """Process tshirt detection - placeholder"""
        # TODO: Implement tshirt detection logic
        return self._create_simple_detection_response(None, None, session_id, "tshirt detection")
    
    def _process_helmet(self, img, session_id):
        """Process helmet detection - placeholder"""
        # TODO: Implement helmet detection logic
        return self._create_simple_detection_response(None, None, session_id, "helmet detection")
    
    def _process_fall(self, img, session_id):
        """Process fall detection - placeholder"""
        # TODO: Implement stateful fall detection logic
        return self._create_simple_detection_response(None, None, session_id, "fall detection")


def serve():
    """Start the gRPC server"""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    model_service_pb2_grpc.add_ModelInferenceServiceServicer_to_server(
        ModelInferenceServicer(), server
    )
    
    listen_addr = '[::]:8901'
    server.add_insecure_port(listen_addr)
    
    logger.info(f"starting grpc server on {listen_addr}")
    server.start()
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("shutting down grpc server...")
        server.stop(0)


if __name__ == '__main__':
    serve()
