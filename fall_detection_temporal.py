"""
Temporal Fall Detection Service for TianWan
Integrates Human-Falling-Detect-Tracks ST-GCN model for time-series fall detection
"""

import sys
import os
import json
import time
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
import torch
from loguru import logger

# Add fall-detection submodule to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'fall-detection'))

try:
    from ActionsEstLoader import TwoStreamSpatialTemporalGraph
    import torch_directml  # For DirectML support on Windows
except ImportError as e:
    logger.warning(f"Could not import fall detection modules: {e}")
    logger.info("Make sure you have initialized the fall-detection submodule and installed dependencies")

class TemporalFallDetectionService:
    """
    Temporal Fall Detection Service using ST-GCN model
    Maintains pose sequences per person for accurate fall detection
    """
    
    def __init__(self, model_path: str = None, device: str = None):
        """
        Initialize temporal fall detection service
        
        Args:
            model_path: Path to ST-GCN model weights
            device: Device to run inference on ('cpu', 'cuda', 'dml')
        """
        self.version = "1.0.0"
        self.sequence_length = 30  # Required frames for ST-GCN
        self.max_persons = 5  # Maximum persons to track
        
        # Person tracking data - per camera
        self.person_sequences = defaultdict(lambda: defaultdict(lambda: deque(maxlen=self.sequence_length)))
        self.person_timestamps = defaultdict(lambda: defaultdict(float))
        self.fall_alerts = defaultdict(list)  # Recent fall detections per camera
        
        # Model configuration
        self.model_path = model_path or os.path.join("fall-detection", "Models", "TSSTG-model.pth")
        self.confidence_threshold = 0.7
        self.alert_cooldown = 3.0  # Seconds between alerts for same person
        
        # Device setup
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch, 'directml') and torch.directml.is_available():
                self.device = torch.device('dml')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
            
        logger.info(f"Fall detection device: {self.device}")
        
        # Load model
        self._load_model()
        
    def _load_model(self):
        """Load ST-GCN model for fall detection"""
        try:
            if not os.path.exists(self.model_path):
                logger.warning(f"ST-GCN model not found at {self.model_path}")
                logger.info("Using fallback pose-based detection")
                self.model = None
                return
                
            # Load ST-GCN model
            self.model = TwoStreamSpatialTemporalGraph(
                in_channels=3,
                num_class=2,  # Fall/No-Fall
                graph_cfg={'layout': 'openpose', 'strategy': 'spatial'}
            )
            
            # Load weights
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
                
            self.model.to(self.device)
            self.model.eval()
            logger.info("ST-GCN model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load ST-GCN model: {e}")
            logger.info("Using fallback pose-based detection")
            self.model = None
    
    def extract_pose_keypoints(self, image: np.ndarray) -> List[Dict]:
        """
        Extract pose keypoints from image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of detected persons with keypoints
        """
        try:
            # Use YOLOv8 pose model for keypoint extraction
            # This is a simplified version - in practice you'd use the actual pose estimation
            from ultralytics import YOLO
            
            if not hasattr(self, 'pose_model'):
                self.pose_model = YOLO('yolov8n-pose.pt')
                
            results = self.pose_model(image, verbose=False)
            
            persons = []
            if results and len(results) > 0:
                result = results[0]
                if result.keypoints is not None:
                    keypoints = result.keypoints.data.cpu().numpy()  # Shape: [N, 17, 3]
                    boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
                    
                    for i, kpts in enumerate(keypoints):
                        if i < len(boxes):
                            person = {
                                'keypoints': kpts,  # [17, 3] - x, y, confidence
                                'bbox': boxes[i],   # [x1, y1, x2, y2]
                                'person_id': i
                            }
                            persons.append(person)
                            
            return persons
            
        except Exception as e:
            logger.error(f"Pose extraction failed: {e}")
            return []
    
    def detect_fall_by_pose(self, keypoints: np.ndarray) -> Tuple[bool, float]:
        """
        Fallback fall detection using pose analysis
        
        Args:
            keypoints: Pose keypoints [17, 3]
            
        Returns:
            Tuple of (is_fall, confidence)
        """
        try:
            # Extract key body points (COCO format)
            head_y = keypoints[0, 1]      # nose
            shoulder_y = (keypoints[5, 1] + keypoints[6, 1]) / 2  # shoulders
            hip_y = (keypoints[11, 1] + keypoints[12, 1]) / 2     # hips
            
            # Check if points are valid (confidence > 0.3)
            valid_points = [
                keypoints[0, 2] > 0.3,    # head
                keypoints[5, 2] > 0.3 or keypoints[6, 2] > 0.3,  # shoulders
                keypoints[11, 2] > 0.3 or keypoints[12, 2] > 0.3  # hips
            ]
            
            if not all(valid_points):
                return False, 0.0
            
            # Calculate body orientation
            body_height = abs(head_y - hip_y)
            if body_height < 10:  # Too small to analyze
                return False, 0.0
                
            # Fall detection logic
            # 1. Head lower than hips (person lying down)
            head_below_hips = head_y > hip_y
            
            # 2. Body is more horizontal than vertical
            shoulder_hip_diff = abs(shoulder_y - hip_y)
            horizontal_ratio = shoulder_hip_diff / body_height if body_height > 0 else 0
            
            is_horizontal = horizontal_ratio < 0.3  # Body is quite horizontal
            
            # Combine conditions
            fall_detected = head_below_hips and is_horizontal
            confidence = 0.8 if fall_detected else 0.2
            
            return fall_detected, confidence
            
        except Exception as e:
            logger.error(f"Pose-based fall detection failed: {e}")
            return False, 0.0
    
    def process_frame(self, image: np.ndarray, camera_id: str) -> Dict:
        """
        Process single frame for fall detection
        
        Args:
            image: Input frame (BGR format)
            camera_id: Unique camera identifier
            
        Returns:
            Detection results with fall alerts
        """
        current_time = time.time()
        
        # Extract pose keypoints
        persons = self.extract_pose_keypoints(image)
        
        results = {
            'camera_id': camera_id,
            'timestamp': current_time,
            'persons_detected': len(persons),
            'fall_detected': False,
            'alerts': []
        }
        
        if not persons:
            return results
        
        # Process each detected person
        for person in persons[:self.max_persons]:
            person_id = f"person_{person['person_id']}"
            keypoints = person['keypoints']
            
            # Add to sequence
            self.person_sequences[camera_id][person_id].append({
                'keypoints': keypoints,
                'timestamp': current_time,
                'bbox': person['bbox']
            })
            
            self.person_timestamps[camera_id][person_id] = current_time
            
            # Continue processing (sequence length tracking removed for simplicity)
            
            # Check for fall detection
            fall_detected, confidence = self._detect_fall_for_person(
                camera_id, person_id, keypoints
            )
            
            if fall_detected and confidence > self.confidence_threshold:
                # Check cooldown to avoid spam
                if self._can_alert(camera_id, person_id, current_time):
                    alert = {
                        'person_id': person_id,
                        'confidence': confidence,
                        'bbox': person['bbox'].tolist(),
                        'timestamp': current_time,
                        'alert_type': 'FALL_DETECTED'
                    }
                    
                    results['alerts'].append(alert)
                    results['fall_detected'] = True
                    
                    # Record alert
                    self.fall_alerts[camera_id].append({
                        'person_id': person_id,
                        'timestamp': current_time
                    })
                    
                    logger.warning(f"FALL DETECTED - Camera: {camera_id}, Person: {person_id}, Confidence: {confidence:.2f}")
        
        # Cleanup old sequences and alerts
        self._cleanup_old_data(camera_id, current_time)
        
        return results
    
    def _detect_fall_for_person(self, camera_id: str, person_id: str, current_keypoints: np.ndarray) -> Tuple[bool, float]:
        """
        Detect fall for a specific person
        
        Args:
            camera_id: Camera identifier
            person_id: Person identifier
            current_keypoints: Current frame keypoints
            
        Returns:
            Tuple of (fall_detected, confidence)
        """
        sequence = self.person_sequences[camera_id][person_id]
        
        # If we have the ST-GCN model and enough frames
        if self.model and len(sequence) >= self.sequence_length:
            return self._detect_fall_stgcn(sequence)
        
        # Fallback to pose-based detection
        return self.detect_fall_by_pose(current_keypoints)
    
    def _detect_fall_stgcn(self, sequence: deque) -> Tuple[bool, float]:
        """
        Use ST-GCN model for fall detection
        
        Args:
            sequence: Sequence of pose frames
            
        Returns:
            Tuple of (fall_detected, confidence)
        """
        try:
            # Prepare input tensor
            keypoints_sequence = []
            for frame_data in sequence:
                keypoints_sequence.append(frame_data['keypoints'][:, :2])  # Only x, y coordinates
            
            # Convert to tensor [T, V, C] -> [1, C, T, V, M]
            input_tensor = np.array(keypoints_sequence)  # [T=30, V=17, C=2]
            input_tensor = input_tensor.transpose(2, 0, 1)  # [C=2, T=30, V=17]
            input_tensor = np.expand_dims(input_tensor, axis=(0, -1))  # [1, C=2, T=30, V=17, M=1]
            
            # Normalize
            input_tensor = (input_tensor - input_tensor.mean()) / (input_tensor.std() + 1e-8)
            
            # Convert to torch tensor
            input_tensor = torch.FloatTensor(input_tensor).to(self.device)
            
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                fall_prob = probabilities[0, 1].item()  # Probability of fall class
                
            fall_detected = fall_prob > 0.5
            return fall_detected, fall_prob
            
        except Exception as e:
            logger.error(f"ST-GCN inference failed: {e}")
            # Fallback to last frame pose detection
            if sequence:
                return self.detect_fall_by_pose(sequence[-1]['keypoints'])
            return False, 0.0
    
    def _can_alert(self, camera_id: str, person_id: str, current_time: float) -> bool:
        """Check if we can send an alert (cooldown logic)"""
        alerts = self.fall_alerts.get(camera_id, [])
        
        for alert in reversed(alerts):  # Check recent alerts first
            if alert['person_id'] == person_id:
                if current_time - alert['timestamp'] < self.alert_cooldown:
                    return False
                break
                
        return True
    
    def _cleanup_old_data(self, camera_id: str, current_time: float):
        """Clean up old tracking data"""
        timeout = 10.0  # Remove person data after 10 seconds of inactivity
        
        # Clean up person sequences
        inactive_persons = []
        for person_id, last_seen in self.person_timestamps[camera_id].items():
            if current_time - last_seen > timeout:
                inactive_persons.append(person_id)
        
        for person_id in inactive_persons:
            if person_id in self.person_sequences[camera_id]:
                del self.person_sequences[camera_id][person_id]
            del self.person_timestamps[camera_id][person_id]
        
        # Clean up old alerts (keep only last hour)
        alert_timeout = 3600.0
        if camera_id in self.fall_alerts:
            self.fall_alerts[camera_id] = [
                alert for alert in self.fall_alerts[camera_id]
                if current_time - alert['timestamp'] < alert_timeout
            ]


# Service instance (singleton pattern)
_fall_detection_service = None

def get_fall_detection_service() -> TemporalFallDetectionService:
    """Get or create fall detection service instance"""
    global _fall_detection_service
    if _fall_detection_service is None:
        _fall_detection_service = TemporalFallDetectionService()
    return _fall_detection_service
