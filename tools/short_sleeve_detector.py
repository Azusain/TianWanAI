#!/usr/bin/env python3
import os
import sys
import argparse
import shutil
from pathlib import Path

# add parent directory to path to import api modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import cv2
import numpy as np
from api import YoloDetectionService, YoloClassificationService
from loguru import logger

# configure logger
logger.remove()
logger.add(
    sys.stderr,
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}:{function}:{line}</cyan> | {message}",
    level="INFO"
)

class ShortSleeveDetector:
    def __init__(self, pose_model_path="models/yolo11m-pose.pt", 
                 tshirt_model_path="models/tshirt_cls/weights/tshirt_cls_v3.pt"):
        """
        Initialize the short sleeve detector
        
        Args:
            pose_model_path: path to pose detection model
            tshirt_model_path: path to tshirt classification model
        """
        self.pose_model_path = pose_model_path
        self.tshirt_model_path = tshirt_model_path
        self.pose_service = None
        self.tshirt_classifier = None
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
    def load_models(self):
        """load pose detection and tshirt classification models"""
        try:
            logger.info("loading pose detection model...")
            if not os.path.exists(self.pose_model_path):
                logger.error(f"pose model not found: {self.pose_model_path}")
                raise FileNotFoundError(f"pose model not found: {self.pose_model_path}")
            
            self.pose_service = YoloDetectionService(self.pose_model_path, 640)
            logger.success("pose detection model loaded successfully")
            
            logger.info("loading tshirt classification model...")
            if not os.path.exists(self.tshirt_model_path):
                logger.error(f"tshirt classification model not found: {self.tshirt_model_path}")
                raise FileNotFoundError(f"tshirt classification model not found: {self.tshirt_model_path}")
            
            self.tshirt_classifier = YoloClassificationService(self.tshirt_model_path, 224)
            logger.success("tshirt classification model loaded successfully")
            
        except Exception as e:
            logger.error(f"failed to load models: {e}")
            raise e
    
    def extract_upper_body(self, img, pose_result):
        """
        extract upper body region using pose keypoints
        
        Args:
            img: input image
            pose_result: pose detection result from YOLO model
            
        Returns:
            list of cropped upper body regions with their bounding boxes
        """
        upper_body_crops = []
        
        if pose_result.keypoints is not None and pose_result.boxes is not None:
            keypoints = pose_result.keypoints.data  # [N, 17, 3] for COCO format
            boxes = pose_result.boxes.data  # [N, 6] - xyxy + conf + cls
            H, W = img.shape[:2]
            
            for i in range(len(keypoints)):
                person_keypoints = keypoints[i].cpu().numpy()  # [17, 3]
                person_box = boxes[i].cpu().numpy()  # [6]
                
                # extract shoulder and hip keypoints (COCO format)
                left_shoulder = person_keypoints[5]   # [x, y, conf]
                right_shoulder = person_keypoints[6]  # [x, y, conf] 
                left_hip = person_keypoints[11]       # [x, y, conf]
                right_hip = person_keypoints[12]      # [x, y, conf]
                
                # check if key points are detected with sufficient confidence
                min_keypoint_conf = 0.3
                if (left_shoulder[2] < min_keypoint_conf or right_shoulder[2] < min_keypoint_conf or
                    left_hip[2] < min_keypoint_conf or right_hip[2] < min_keypoint_conf):
                    logger.warning(f"skipping person {i}: insufficient keypoint confidence")
                    continue
                
                # find the highest and lowest points among shoulders and hips
                all_y_coords = [left_shoulder[1], right_shoulder[1], left_hip[1], right_hip[1]]
                upper_y = min(all_y_coords)  # highest point (smallest Y coordinate)
                lower_y = max(all_y_coords)  # lowest point (largest Y coordinate)
                
                # use person bbox for x boundaries  
                x1, y1, x2, y2 = person_box[:4]
                
                # calculate upper body region with margin
                margin_y = int((lower_y - upper_y) * 0.1)
                crop_upper_y = max(0, int(upper_y - margin_y))
                crop_lower_y = min(H, int(lower_y + margin_y))
                
                x1, y1, x2, y2 = int(x1), int(crop_upper_y), int(x2), int(crop_lower_y)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(W, x2)
                y2 = min(H, y2)
                
                if x2 > x1 and y2 > y1:
                    # check aspect ratio to filter out abnormal detections
                    width_px = x2 - x1
                    height_px = y2 - y1
                    aspect_ratio = width_px / height_px
                    
                    # filter out extreme aspect ratios
                    if aspect_ratio > 2.0 or aspect_ratio < 0.5:
                        logger.warning(f"skipping person {i}: abnormal aspect ratio {aspect_ratio:.2f}")
                        continue
                    
                    # crop upper body region
                    upper_body_crop = img[y1:y2, x1:x2]
                    
                    if upper_body_crop.size > 0:
                        upper_body_crops.append({
                            'crop': upper_body_crop,
                            'bbox': (x1, y1, x2, y2),
                            'person_conf': float(person_box[4])
                        })
                        
        return upper_body_crops
    
    def detect_person_poses(self, img):
        """detect persons using pose model"""
        try:
            pose_results = self.pose_service.model.predict(
                source=img,
                imgsz=640,
                verbose=False,
                conf=0.6  # person detection confidence threshold
            )
            
            if pose_results and len(pose_results) > 0:
                return pose_results[0]
            return None
            
        except Exception as e:
            logger.error(f"pose detection error: {e}")
            return None
    
    def classify_tshirt(self, upper_body_crop):
        """classify if the upper body region contains short sleeves"""
        try:
            top1_prob, top1_class, _, _ = self.tshirt_classifier.Predict(upper_body_crop)
            
            # assume class 1 = short sleeve, class 0 = not short sleeve
            # this may need to be adjusted based on actual model training
            is_short_sleeve = (top1_class == 1)
            confidence = top1_prob
            
            return is_short_sleeve, confidence
            
        except Exception as e:
            logger.error(f"tshirt classification error: {e}")
            return False, 0.0
    
    def process_image(self, image_path, output_short_dir, output_not_short_dir, confidence_threshold=0.5):
        """
        process a single image for short sleeve detection
        
        Args:
            image_path: path to input image
            output_short_dir: directory for short sleeve images
            output_not_short_dir: directory for non-short sleeve images
            confidence_threshold: minimum confidence for classification
            
        Returns:
            dict with processing results
        """
        try:
            # load image
            img = cv2.imread(str(image_path))
            if img is None:
                logger.error(f"failed to load image: {image_path}")
                return {'status': 'failed', 'error': 'failed to load image'}
            
            # detect poses
            pose_result = self.detect_person_poses(img)
            if pose_result is None:
                logger.warning(f"no persons detected in {image_path}")
                return {'status': 'no_person', 'persons': 0}
            
            # extract upper body regions
            upper_body_crops = self.extract_upper_body(img, pose_result)
            if not upper_body_crops:
                logger.warning(f"no valid upper body regions found in {image_path}")
                return {'status': 'no_upper_body', 'persons': 0}
            
            # classify each person
            short_sleeve_detected = False
            max_confidence = 0.0
            person_results = []
            
            for i, crop_data in enumerate(upper_body_crops):
                is_short_sleeve, confidence = self.classify_tshirt(crop_data['crop'])
                
                person_results.append({
                    'person_id': i,
                    'is_short_sleeve': is_short_sleeve,
                    'confidence': confidence,
                    'person_conf': crop_data['person_conf']
                })
                
                if is_short_sleeve and confidence >= confidence_threshold:
                    short_sleeve_detected = True
                    max_confidence = max(max_confidence, confidence)
            
            # copy image to appropriate output directory
            image_name = Path(image_path).name
            
            if short_sleeve_detected:
                output_path = output_short_dir / image_name
                shutil.copy2(image_path, output_path)
                logger.success(f"short sleeve detected in {image_name} (conf: {max_confidence:.3f})")
                status = 'short_sleeve'
            else:
                output_path = output_not_short_dir / image_name
                shutil.copy2(image_path, output_path)
                logger.info(f"no short sleeve detected in {image_name}")
                status = 'not_short_sleeve'
            
            return {
                'status': status,
                'persons': len(person_results),
                'max_confidence': max_confidence,
                'person_results': person_results,
                'output_path': str(output_path)
            }
            
        except Exception as e:
            logger.error(f"error processing {image_path}: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def process_folder(self, input_folder, output_folder, confidence_threshold=0.5):
        """
        process all images in a folder
        
        Args:
            input_folder: path to input folder containing images
            output_folder: path to output folder (will create subfolders)
            confidence_threshold: minimum confidence for classification
        """
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        
        if not input_path.exists() or not input_path.is_dir():
            logger.error(f"input folder does not exist: {input_folder}")
            return
        
        # create output directories
        output_short_dir = output_path / "short_sleeve"
        output_not_short_dir = output_path / "not_short_sleeve"
        
        output_short_dir.mkdir(parents=True, exist_ok=True)
        output_not_short_dir.mkdir(parents=True, exist_ok=True)
        
        # find all image files
        image_files = []
        for ext in self.supported_formats:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            logger.warning(f"no supported image files found in {input_folder}")
            return
        
        logger.info(f"found {len(image_files)} images to process")
        
        # process statistics
        stats = {
            'total': len(image_files),
            'short_sleeve': 0,
            'not_short_sleeve': 0,
            'no_person': 0,
            'errors': 0
        }
        
        # process each image
        for i, image_file in enumerate(image_files, 1):
            logger.info(f"processing [{i}/{len(image_files)}]: {image_file.name}")
            
            result = self.process_image(
                image_file, 
                output_short_dir, 
                output_not_short_dir, 
                confidence_threshold
            )
            
            if result['status'] == 'short_sleeve':
                stats['short_sleeve'] += 1
            elif result['status'] == 'not_short_sleeve':
                stats['not_short_sleeve'] += 1
            elif result['status'] in ['no_person', 'no_upper_body']:
                stats['no_person'] += 1
            else:
                stats['errors'] += 1
        
        # print final statistics
        logger.success("processing completed!")
        logger.info(f"total images: {stats['total']}")
        logger.info(f"short sleeve detected: {stats['short_sleeve']}")
        logger.info(f"not short sleeve: {stats['not_short_sleeve']}")
        logger.info(f"no person detected: {stats['no_person']}")
        logger.info(f"errors: {stats['errors']}")
        
        logger.info(f"results saved to:")
        logger.info(f"  short sleeve: {output_short_dir}")
        logger.info(f"  not short sleeve: {output_not_short_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="detect short sleeves in images using pose detection and classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python short_sleeve_detector.py input_folder output_folder
  python short_sleeve_detector.py input_folder output_folder --confidence 0.7
  python short_sleeve_detector.py input_folder output_folder --pose-model models/custom_pose.pt
        """
    )
    
    parser.add_argument(
        "input_folder",
        help="path to folder containing input images"
    )
    
    parser.add_argument(
        "output_folder", 
        help="path to output folder (will create subfolders for results)"
    )
    
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="minimum confidence threshold for short sleeve classification (default: 0.5)"
    )
    
    parser.add_argument(
        "--pose-model",
        default="models/yolo11m-pose.pt",
        help="path to pose detection model (default: models/yolo11m-pose.pt)"
    )
    
    parser.add_argument(
        "--tshirt-model",
        default="models/tshirt_cls/weights/tshirt_cls_v3.pt",
        help="path to tshirt classification model (default: models/tshirt_cls/weights/tshirt_cls_v3.pt)"
    )
    
    args = parser.parse_args()
    
    # validate input arguments
    if not os.path.exists(args.input_folder):
        logger.error(f"input folder does not exist: {args.input_folder}")
        sys.exit(1)
    
    if args.confidence < 0.0 or args.confidence > 1.0:
        logger.error(f"confidence must be between 0.0 and 1.0, got: {args.confidence}")
        sys.exit(1)
    
    # initialize detector
    try:
        detector = ShortSleeveDetector(
            pose_model_path=args.pose_model,
            tshirt_model_path=args.tshirt_model
        )
        
        # load models
        detector.load_models()
        
        # process folder
        detector.process_folder(
            args.input_folder,
            args.output_folder,
            args.confidence
        )
        
    except KeyboardInterrupt:
        logger.warning("processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
