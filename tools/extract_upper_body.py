import os
import cv2
import numpy as np
import argparse
from pathlib import Path
from ultralytics import YOLO
import torch
from loguru import logger

class UpperBodyExtractor:
    def __init__(self, pose_model_path, output_dir="upper_body_crops"):
        """
        initialize upper body extractor
        
        Args:
            pose_model_path: path to yolo pose model
            output_dir: directory to save cropped images
        """
        self.output_dir = output_dir
        self.pose_model_path = pose_model_path
        
        # create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # load pose model
        logger.info(f"loading pose model from {pose_model_path}")
        self.pose_model = YOLO(pose_model_path)
        
        # set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pose_model.to(self.device)
        logger.info(f"using device: {self.device}")
        
        # model warmup
        logger.info("warming up pose model...")
        dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        self.pose_model.predict(source=dummy_img, verbose=False)
        logger.success("pose model loaded and warmed up!")
        
    def extract_upper_body_from_person(self, img, person_data, margin_ratio=0.1):
        """
        extract upper body region based on shoulder and hip keypoints
        
        Args:
            img: input image
            person_data: dict containing keypoints and bbox
            margin_ratio: margin ratio for cropping
            
        Returns:
            upper body image or None if invalid
        """
        H, W = img.shape[:2]
        
        # extract keypoints
        keypoints = person_data['keypoints']
        
        # get shoulder and hip keypoints (COCO format)
        left_shoulder = keypoints[5]   # [x, y, conf]
        right_shoulder = keypoints[6]  # [x, y, conf]
        left_hip = keypoints[11]       # [x, y, conf]
        right_hip = keypoints[12]      # [x, y, conf]
        
        # find shoulder top and hip bottom
        shoulder_y_min = min(left_shoulder[1], right_shoulder[1])
        hip_y_max = max(left_hip[1], right_hip[1])
        
        # use person bbox for x boundaries
        bbox = person_data['bbox']
        x1, y1, x2, y2 = bbox
        
        # calculate upper body region with margin
        margin_y = int((hip_y_max - shoulder_y_min) * margin_ratio)
        upper_y = max(0, int(shoulder_y_min - margin_y))
        lower_y = min(H, int(hip_y_max + margin_y))
        
        # ensure boundaries are within image and bbox
        x1 = max(0, int(x1))
        y1 = max(0, upper_y)
        x2 = min(W, int(x2))
        y2 = min(H, lower_y)
        
        if x2 <= x1 or y2 <= y1:
            logger.warning("invalid upper body region boundaries")
            return None
            
        # crop upper body region
        upper_body_region = img[y1:y2, x1:x2]
        
        # check minimum size
        if upper_body_region.shape[0] < 50 or upper_body_region.shape[1] < 50:
            logger.warning("upper body region too small, skipping")
            return None
            
        return upper_body_region
        
    def detect_persons_in_frame(self, frame):
        """
        detect persons in frame using pose model
        
        Args:
            frame: input frame
            
        Returns:
            list of person data
        """
        results = self.pose_model.predict(
            source=frame,
            imgsz=640,
            verbose=False
        )
        
        result = results[0]
        persons_data = []
        
        if result.keypoints is not None and result.boxes is not None:
            keypoints = result.keypoints.data  # [N, 17, 3] for COCO format
            boxes = result.boxes.data  # [N, 6] - xyxy + conf + cls
            
            for i in range(len(keypoints)):
                person_keypoints = keypoints[i].cpu().numpy()  # [17, 3]
                person_box = boxes[i].cpu().numpy()  # [6]
                
                persons_data.append({
                    'bbox': person_box[:4],  # [x1, y1, x2, y2]
                    'confidence': person_box[4],  # detection confidence
                    'keypoints': person_keypoints
                })
                
        return persons_data
        
    def process_video(self, video_path, frame_interval=30, min_confidence=0.5, 
                     prefix="upper_body", save_debug_frames=False):
        """
        process video and extract upper body regions
        
        Args:
            video_path: path to input video
            frame_interval: interval between frames to process (1 = every frame)
            min_confidence: minimum person detection confidence
            prefix: prefix for saved images
            save_debug_frames: whether to save debug frames with pose annotations
        """
        if not os.path.exists(video_path):
            logger.error(f"video file not found: {video_path}")
            return
            
        # open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"failed to open video: {video_path}")
            return
            
        # get video info
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"video info: {total_frames} frames, {fps:.2f} fps, {duration:.2f}s")
        logger.info(f"processing every {frame_interval} frame(s)")
        
        # create debug output directory if needed
        if save_debug_frames:
            debug_dir = os.path.join(self.output_dir, "debug_frames")
            Path(debug_dir).mkdir(exist_ok=True)
        
        frame_count = 0
        processed_count = 0
        saved_count = 0
        
        video_name = Path(video_path).stem
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # process frame at specified interval
            if frame_count % frame_interval == 0:
                processed_count += 1
                logger.info(f"processing frame {frame_count}/{total_frames} ({processed_count})")
                
                # detect persons
                persons_data = self.detect_persons_in_frame(frame)
                
                # create debug frame if needed
                debug_frame = frame.copy() if save_debug_frames else None
                
                # process each person
                for person_idx, person_data in enumerate(persons_data):
                    if person_data['confidence'] < min_confidence:
                        continue
                        
                    # extract upper body
                    upper_body_img = self.extract_upper_body_from_person(frame, person_data)
                    
                    if upper_body_img is not None:
                        # save upper body image
                        filename = f"{prefix}_{video_name}_frame{frame_count:06d}_person{person_idx}.jpg"
                        save_path = os.path.join(self.output_dir, filename)
                        cv2.imwrite(save_path, upper_body_img)
                        saved_count += 1
                        
                        logger.success(f"saved upper body: {filename} (size: {upper_body_img.shape})")
                    
                    # draw debug annotations
                    if save_debug_frames and debug_frame is not None:
                        # draw person bbox
                        bbox = person_data['bbox'].astype(int)
                        cv2.rectangle(debug_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                        
                        # draw keypoints
                        keypoints = person_data['keypoints']
                        for kpt in keypoints:
                            if kpt[2] > 0.5:  # confidence threshold
                                cv2.circle(debug_frame, (int(kpt[0]), int(kpt[1])), 3, (0, 0, 255), -1)
                        
                        # highlight shoulder and hip keypoints
                        shoulder_hip_indices = [5, 6, 11, 12]  # left_shoulder, right_shoulder, left_hip, right_hip
                        for idx in shoulder_hip_indices:
                            kpt = keypoints[idx]
                            if kpt[2] > 0.5:
                                cv2.circle(debug_frame, (int(kpt[0]), int(kpt[1])), 6, (255, 0, 0), -1)
                
                # save debug frame
                if save_debug_frames and debug_frame is not None:
                    debug_filename = f"debug_{video_name}_frame{frame_count:06d}.jpg"
                    debug_save_path = os.path.join(debug_dir, debug_filename)
                    cv2.imwrite(debug_save_path, debug_frame)
            
            frame_count += 1
            
        cap.release()
        
        logger.success(f"processing complete!")
        logger.info(f"total frames: {total_frames}")
        logger.info(f"processed frames: {processed_count}")
        logger.info(f"saved upper body images: {saved_count}")
        logger.info(f"output directory: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description='extract upper body regions from video using pose detection')
    parser.add_argument('video', help='path to input video file')
    parser.add_argument('--pose-model', default='models/yolo11m-pose.pt', 
                       help='path to pose detection model (default: models/yolo11m-pose.pt)')
    parser.add_argument('--output', '-o', default='upper_body_crops', 
                       help='output directory (default: upper_body_crops)')
    parser.add_argument('--interval', '-i', type=int, default=30,
                       help='frame interval to process (1=every frame, 30=every 30th frame, default: 30)')
    parser.add_argument('--confidence', '-c', type=float, default=0.5,
                       help='minimum person detection confidence (default: 0.5)')
    parser.add_argument('--prefix', '-p', default='upper_body',
                       help='prefix for saved images (default: upper_body)')
    parser.add_argument('--debug', action='store_true',
                       help='save debug frames with pose annotations')
    parser.add_argument('--margin', '-m', type=float, default=0.1,
                       help='margin ratio for cropping (default: 0.1)')
    
    args = parser.parse_args()
    
    # check if video file exists
    if not os.path.exists(args.video):
        logger.error(f"video file not found: {args.video}")
        return
        
    # check if pose model exists
    if not os.path.exists(args.pose_model):
        logger.error(f"pose model not found: {args.pose_model}")
        return
    
    # create extractor
    extractor = UpperBodyExtractor(args.pose_model, args.output)
    
    # process video
    extractor.process_video(
        video_path=args.video,
        frame_interval=args.interval,
        min_confidence=args.confidence,
        prefix=args.prefix,
        save_debug_frames=args.debug
    )

if __name__ == "__main__":
    main()
