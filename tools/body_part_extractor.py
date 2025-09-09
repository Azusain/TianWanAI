#!/usr/bin/env python3
"""
Enhanced Multi-threaded Body Part Extractor

Supports extraction of different body regions:
- Upper body (shoulders to hips)
- Head region (head and neck)
- Lower body (hips to knees/ankles)
- Full body (entire person)

Based on YOLO pose estimation with multi-threading for performance
"""

import os
import cv2
import numpy as np
import argparse
import glob
import time
from pathlib import Path
from ultralytics import YOLO
import torch
from loguru import logger
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional
from enum import Enum

class BodyRegion(Enum):
    """Supported body regions for extraction"""
    UPPER_BODY = "upper_body"
    HEAD = "head"
    LOWER_BODY = "lower_body"
    FULL_BODY = "full_body"

class FrameData:
    """Container for frame information"""
    def __init__(self, frame: np.ndarray, frame_idx: int, source_name: str):
        self.frame = frame
        self.frame_idx = frame_idx
        self.source_name = source_name

class ProcessedResult:
    """Container for processing results"""
    def __init__(self, frame_data: FrameData, extractions: List[Tuple[np.ndarray, str, Dict]], 
                 debug_frame: Optional[np.ndarray] = None):
        self.frame_data = frame_data
        self.extractions = extractions  # [(extracted_image, filename, metadata), ...]
        self.debug_frame = debug_frame

class MultiThreadBodyPartExtractor:
    def __init__(self, pose_model_path: str, output_dir: str = "body_part_crops",
                 num_workers: int = 4, batch_size: int = 4):
        """
        Initialize multi-threaded body part extractor
        
        Args:
            pose_model_path: path to YOLO pose model
            output_dir: directory to save cropped images
            num_workers: number of worker threads for inference
            batch_size: batch size for inference
        """
        self.output_dir = Path(output_dir)
        self.pose_model_path = pose_model_path
        self.num_workers = num_workers
        self.batch_size = batch_size
        
        # create output directory structure
        for region in BodyRegion:
            (self.output_dir / region.value).mkdir(parents=True, exist_ok=True)
        
        # thread synchronization
        self.frame_queue = queue.Queue(maxsize=50)
        self.result_queue = queue.Queue(maxsize=50)
        self.stop_event = threading.Event()
        
        # statistics
        self.stats = {
            'frames_read': 0,
            'frames_processed': 0,
            'images_saved': 0,
            'regions_extracted': {region.value: 0 for region in BodyRegion},
            'start_time': None,
            'lock': threading.Lock()
        }
        
        # load pose model
        logger.info(f"loading pose model from {pose_model_path}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"using device: {self.device}")
        
        # create model pool for workers
        self.model_pool = []
        for i in range(num_workers):
            model = YOLO(pose_model_path)
            model.to(self.device)
            self.model_pool.append(model)
            
        # warmup models
        logger.info("warming up pose models...")
        dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        for model in self.model_pool:
            model.predict(source=dummy_img, verbose=False)
        logger.success(f"loaded and warmed up {num_workers} pose model instances!")
        
    def update_stats(self, **kwargs):
        """thread-safe stats update"""
        with self.stats['lock']:
            for key, value in kwargs.items():
                if key == 'regions_extracted':
                    for region, count in value.items():
                        self.stats['regions_extracted'][region] += count
                elif key in self.stats:
                    self.stats[key] += value
                    
    def get_stats(self) -> Dict:
        """get current statistics"""
        with self.stats['lock']:
            return self.stats.copy()
    
    def extract_body_region(self, img: np.ndarray, person_data: Dict, 
                          region: BodyRegion, margin_ratio: float = 0.1) -> Optional[np.ndarray]:
        """
        Extract specific body region based on keypoints
        
        Args:
            img: input image
            person_data: person detection data with keypoints
            region: body region to extract
            margin_ratio: margin to add around the region
        """
        H, W = img.shape[:2]
        keypoints = person_data['keypoints']
        bbox = person_data['bbox']
        
        # COCO keypoint indices
        # 0: nose, 1-2: eyes, 3-4: ears, 5-6: shoulders, 7-8: elbows, 9-10: wrists
        # 11-12: hips, 13-14: knees, 15-16: ankles
        
        if region == BodyRegion.HEAD:
            # head region: from top to shoulders
            nose = keypoints[0]
            left_eye, right_eye = keypoints[1], keypoints[2]
            left_ear, right_ear = keypoints[3], keypoints[4]
            left_shoulder, right_shoulder = keypoints[5], keypoints[6]
            
            # find head boundaries
            head_points = [nose, left_eye, right_eye, left_ear, right_ear]
            valid_head_points = [p for p in head_points if p[2] > 0.3]  # confidence > 0.3
            
            if len(valid_head_points) < 2:
                return None
                
            head_y_coords = [p[1] for p in valid_head_points]
            shoulder_y = min(left_shoulder[1], right_shoulder[1])
            
            upper_y = max(0, min(head_y_coords))
            lower_y = shoulder_y
            
        elif region == BodyRegion.UPPER_BODY:
            # upper body: shoulders to hips
            left_shoulder, right_shoulder = keypoints[5], keypoints[6]
            left_hip, right_hip = keypoints[11], keypoints[12]
            
            all_y_coords = [left_shoulder[1], right_shoulder[1], left_hip[1], right_hip[1]]
            upper_y = min(all_y_coords)
            lower_y = max(all_y_coords)
            
        elif region == BodyRegion.LOWER_BODY:
            # lower body: hips to ankles
            left_hip, right_hip = keypoints[11], keypoints[12]
            left_knee, right_knee = keypoints[13], keypoints[14]
            left_ankle, right_ankle = keypoints[15], keypoints[16]
            
            # use hips as upper bound
            hip_y = min(left_hip[1], right_hip[1])
            
            # find lowest point (prefer ankles, fallback to knees)
            lower_points = []
            if left_ankle[2] > 0.3:
                lower_points.append(left_ankle[1])
            if right_ankle[2] > 0.3:
                lower_points.append(right_ankle[1])
            
            if not lower_points:  # fallback to knees
                if left_knee[2] > 0.3:
                    lower_points.append(left_knee[1])
                if right_knee[2] > 0.3:
                    lower_points.append(right_knee[1])
            
            if not lower_points:
                return None
                
            upper_y = hip_y
            lower_y = max(lower_points)
            
        elif region == BodyRegion.FULL_BODY:
            # full body: use person bbox with small margin
            upper_y = bbox[1]
            lower_y = bbox[3]
            
        else:
            raise ValueError(f"unsupported body region: {region}")
        
        # add margin
        height_diff = lower_y - upper_y
        margin_y = int(height_diff * margin_ratio)
        crop_upper_y = max(0, int(upper_y - margin_y))
        crop_lower_y = min(H, int(lower_y + margin_y))
        
        # use person bbox for x boundaries with margin
        bbox_width = bbox[2] - bbox[0]
        margin_x = int(bbox_width * margin_ratio * 0.5)  # smaller horizontal margin
        x1 = max(0, int(bbox[0] - margin_x))
        y1 = crop_upper_y
        x2 = min(W, int(bbox[2] + margin_x))
        y2 = crop_lower_y
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        # check aspect ratio to filter out abnormal detections
        width_px = x2 - x1
        height_px = y2 - y1
        aspect_ratio = width_px / height_px
        
        # different aspect ratio constraints for different regions
        if region == BodyRegion.HEAD:
            if aspect_ratio > 2.0 or aspect_ratio < 0.5:
                return None
        elif region == BodyRegion.UPPER_BODY:
            if aspect_ratio > 3.0 or aspect_ratio < 0.3:
                return None
        elif region == BodyRegion.LOWER_BODY:
            if aspect_ratio > 2.5 or aspect_ratio < 0.4:
                return None
        # full body can have wider aspect ratio range
        
        # crop region
        extracted_region = img[y1:y2, x1:x2]
        
        # check minimum size
        min_size = 50 if region != BodyRegion.HEAD else 30  # smaller minimum for heads
        if extracted_region.shape[0] < min_size or extracted_region.shape[1] < min_size:
            return None
            
        return extracted_region
        
    def detect_persons_batch(self, frames: List[FrameData], model: YOLO) -> List[Tuple[FrameData, List[Dict]]]:
        """batch detect persons in multiple frames"""
        if not frames:
            return []
            
        frame_images = [fd.frame for fd in frames]
        
        # batch inference
        results = model.predict(
            source=frame_images,
            imgsz=640,
            verbose=False
        )
        
        batch_results = []
        
        for frame_data, result in zip(frames, results):
            persons_data = []
            
            if result.keypoints is not None and result.boxes is not None:
                keypoints = result.keypoints.data  # [N, 17, 3]
                boxes = result.boxes.data  # [N, 6]
                
                for i in range(len(keypoints)):
                    person_keypoints = keypoints[i].cpu().numpy()  # [17, 3]
                    person_box = boxes[i].cpu().numpy()  # [6]
                    
                    persons_data.append({
                        'bbox': person_box[:4],  # [x1, y1, x2, y2]
                        'confidence': person_box[4],
                        'keypoints': person_keypoints
                    })
                    
            batch_results.append((frame_data, persons_data))
            
        return batch_results
    
    def process_worker(self, worker_id: int, regions: List[BodyRegion], 
                      min_confidence: float, prefix: str, margin_ratio: float):
        """worker thread for processing frames"""
        model = self.model_pool[worker_id]
        batch_frames = []
        
        logger.info(f"worker {worker_id} started, processing regions: {[r.value for r in regions]}")
        
        while not self.stop_event.is_set():
            try:
                # collect batch
                for _ in range(self.batch_size):
                    try:
                        frame_data = self.frame_queue.get(timeout=1.0)
                        if frame_data is None:  # end signal
                            break
                        batch_frames.append(frame_data)
                    except queue.Empty:
                        break
                
                if not batch_frames:
                    continue
                
                # batch process
                batch_results = self.detect_persons_batch(batch_frames, model)
                
                for frame_data, persons_data in batch_results:
                    extractions = []
                    region_counts = {region.value: 0 for region in BodyRegion}
                    
                    for person_idx, person_data in enumerate(persons_data):
                        if person_data['confidence'] < min_confidence:
                            continue
                        
                        # extract all requested regions for this person
                        for region in regions:
                            extracted_region = self.extract_body_region(
                                frame_data.frame, person_data, region, margin_ratio
                            )
                            
                            if extracted_region is not None:
                                # generate filename
                                filename = (f"{prefix}_{region.value}_{frame_data.source_name}_"
                                          f"frame{frame_data.frame_idx:06d}_person{person_idx}.jpg")
                                
                                metadata = {
                                    'region': region.value,
                                    'confidence': person_data['confidence'],
                                    'frame_idx': frame_data.frame_idx,
                                    'person_idx': person_idx,
                                    'shape': extracted_region.shape
                                }
                                
                                extractions.append((extracted_region, filename, metadata))
                                region_counts[region.value] += 1
                    
                    # create result
                    result = ProcessedResult(frame_data, extractions)
                    self.result_queue.put(result)
                    
                    # update stats
                    self.update_stats(
                        frames_processed=1,
                        regions_extracted=region_counts
                    )
                
                batch_frames.clear()
                
            except Exception as e:
                logger.error(f"worker {worker_id} error: {e}")
                continue
        
        logger.info(f"worker {worker_id} finished")
    
    def save_worker(self):
        """worker thread for saving extracted images"""
        logger.info("save worker started")
        
        while not self.stop_event.is_set():
            try:
                result = self.result_queue.get(timeout=1.0)
                if result is None:  # end signal
                    break
                
                saved_count = 0
                for extracted_img, filename, metadata in result.extractions:
                    region_dir = self.output_dir / metadata['region']
                    output_path = region_dir / filename
                    
                    success = cv2.imwrite(str(output_path), extracted_img)
                    if success:
                        saved_count += 1
                    else:
                        logger.warning(f"failed to save: {output_path}")
                
                self.update_stats(images_saved=saved_count)
                
                # log progress
                stats = self.get_stats()
                if stats['frames_processed'] % 100 == 0 or stats['frames_processed'] <= 10:
                    elapsed = time.time() - stats['start_time'] if stats['start_time'] else 0
                    fps = stats['frames_processed'] / elapsed if elapsed > 0 else 0
                    logger.info(f"processed: {stats['frames_processed']} frames, "
                              f"saved: {stats['images_saved']} images, "
                              f"fps: {fps:.1f}")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"save worker error: {e}")
        
        logger.info("save worker finished")
    
    def process_video(self, video_path: str, regions: List[BodyRegion], 
                     interval: int = 30, max_frames: int = None,
                     min_confidence: float = 0.5, margin_ratio: float = 0.1,
                     prefix: str = "extracted"):
        """
        Process single video file
        
        Args:
            video_path: path to video file
            regions: list of body regions to extract
            interval: frame sampling interval
            max_frames: maximum frames to process
            min_confidence: minimum detection confidence
            margin_ratio: margin around extracted regions
            prefix: filename prefix for saved images
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"cannot open video: {video_path}")
            return
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_name = Path(video_path).stem
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"processing video: {video_name}")
        logger.info(f"total frames: {total_frames}, fps: {fps:.2f}")
        logger.info(f"extracting regions: {[r.value for r in regions]}")
        logger.info(f"sampling interval: {interval}, max frames: {max_frames}")
        
        # reset stats
        with self.stats['lock']:
            self.stats['start_time'] = time.time()
            self.stats['frames_read'] = 0
            self.stats['frames_processed'] = 0
            self.stats['images_saved'] = 0
            for region in BodyRegion:
                self.stats['regions_extracted'][region.value] = 0
        
        # start worker threads
        self.stop_event.clear()
        
        # start processing workers
        process_threads = []
        for i in range(self.num_workers):
            thread = threading.Thread(
                target=self.process_worker,
                args=(i, regions, min_confidence, prefix, margin_ratio)
            )
            thread.start()
            process_threads.append(thread)
        
        # start save worker
        save_thread = threading.Thread(target=self.save_worker)
        save_thread.start()
        
        # read and queue frames
        frame_count = 0
        queued_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                self.update_stats(frames_read=1)
                
                if frame_count % interval == 0:
                    if max_frames and queued_count >= max_frames:
                        break
                        
                    frame_data = FrameData(frame, frame_count, video_name)
                    self.frame_queue.put(frame_data)
                    queued_count += 1
                
                frame_count += 1
        
        finally:
            cap.release()
        
        # signal end to workers
        logger.info("signaling workers to stop...")
        for _ in range(self.num_workers):
            self.frame_queue.put(None)
        
        # wait for processing to complete
        for thread in process_threads:
            thread.join()
        
        # signal save worker to stop
        self.result_queue.put(None)
        save_thread.join()
        
        # final stats
        final_stats = self.get_stats()
        elapsed = time.time() - final_stats['start_time']
        
        logger.success(f"video processing complete!")
        logger.info(f"processed {final_stats['frames_processed']} frames in {elapsed:.1f}s")
        logger.info(f"saved {final_stats['images_saved']} extracted images")
        logger.info("extracted regions:")
        for region, count in final_stats['regions_extracted'].items():
            if count > 0:
                logger.info(f"  {region}: {count}")

def main():
    parser = argparse.ArgumentParser(description="extract body parts from videos using pose estimation")
    parser.add_argument("input", help="input video file or directory")
    parser.add_argument("-m", "--model", required=True, help="path to YOLO pose model")
    parser.add_argument("-o", "--output", default="body_part_crops", help="output directory")
    parser.add_argument("-r", "--regions", nargs="+", 
                       choices=[r.value for r in BodyRegion],
                       default=["upper_body"], 
                       help="body regions to extract")
    parser.add_argument("--interval", type=int, default=30, help="frame sampling interval")
    parser.add_argument("--max-frames", type=int, help="maximum frames to process per video")
    parser.add_argument("--confidence", type=float, default=0.5, help="minimum detection confidence")
    parser.add_argument("--margin", type=float, default=0.1, help="margin ratio around extracted regions")
    parser.add_argument("--workers", type=int, default=4, help="number of worker threads")
    parser.add_argument("--batch-size", type=int, default=4, help="batch size for inference")
    parser.add_argument("--prefix", default="extracted", help="filename prefix for saved images")
    
    args = parser.parse_args()
    
    # convert region strings to enums
    regions = [BodyRegion(r) for r in args.regions]
    
    # create extractor
    extractor = MultiThreadBodyPartExtractor(
        pose_model_path=args.model,
        output_dir=args.output,
        num_workers=args.workers,
        batch_size=args.batch_size
    )
    
    # process input
    input_path = Path(args.input)
    
    if input_path.is_file():
        # single video
        extractor.process_video(
            video_path=str(input_path),
            regions=regions,
            interval=args.interval,
            max_frames=args.max_frames,
            min_confidence=args.confidence,
            margin_ratio=args.margin,
            prefix=args.prefix
        )
    elif input_path.is_dir():
        # directory of videos
        video_files = []
        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv']:
            video_files.extend(input_path.glob(ext))
        
        if not video_files:
            logger.error(f"no video files found in {input_path}")
            return
        
        logger.info(f"found {len(video_files)} video files")
        
        for i, video_file in enumerate(video_files, 1):
            logger.info(f"processing video {i}/{len(video_files)}: {video_file.name}")
            extractor.process_video(
                video_path=str(video_file),
                regions=regions,
                interval=args.interval,
                max_frames=args.max_frames,
                min_confidence=args.confidence,
                margin_ratio=args.margin,
                prefix=args.prefix
            )
    else:
        logger.error(f"input path does not exist: {input_path}")

if __name__ == "__main__":
    main()
