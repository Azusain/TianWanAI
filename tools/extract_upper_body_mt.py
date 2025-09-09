import os
import cv2
import numpy as np
import argparse
from pathlib import Path
from ultralytics import YOLO
import torch
from loguru import logger
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional

class FrameData:
    def __init__(self, frame: np.ndarray, frame_idx: int, video_name: str):
        self.frame = frame
        self.frame_idx = frame_idx
        self.video_name = video_name

class ProcessedResult:
    def __init__(self, frame_data: FrameData, persons_data: List[Dict], 
                 upper_body_images: List[Tuple[np.ndarray, str]], debug_frame: Optional[np.ndarray] = None):
        self.frame_data = frame_data
        self.persons_data = persons_data
        self.upper_body_images = upper_body_images  # [(image, filename), ...]
        self.debug_frame = debug_frame

class MultiThreadUpperBodyExtractor:
    def __init__(self, pose_model_path: str, output_dir: str = "upper_body_crops",
                 num_workers: int = 4, batch_size: int = 4):
        """
        initialize multi-threaded upper body extractor
        
        Args:
            pose_model_path: path to yolo pose model
            output_dir: directory to save cropped images
            num_workers: number of worker threads for inference
            batch_size: batch size for inference
        """
        self.output_dir = output_dir
        self.pose_model_path = pose_model_path
        self.num_workers = num_workers
        self.batch_size = batch_size
        
        # create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # thread synchronization
        self.frame_queue = queue.Queue(maxsize=50)  # frame buffer
        self.result_queue = queue.Queue(maxsize=50)  # processed results
        self.stop_event = threading.Event()
        
        # statistics
        self.stats = {
            'frames_read': 0,
            'frames_processed': 0,
            'images_saved': 0,
            'start_time': None,
            'lock': threading.Lock()
        }
        
        # load pose model (shared across workers)
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
                if key in self.stats:
                    self.stats[key] += value
                    
    def get_stats(self) -> Dict:
        """get current statistics"""
        with self.stats['lock']:
            return self.stats.copy()
    
    def extract_upper_body_from_person(self, img: np.ndarray, person_data: Dict, 
                                     margin_ratio: float = 0.1) -> Optional[np.ndarray]:
        """extract upper body region based on shoulder and hip keypoints"""
        H, W = img.shape[:2]
        
        # extract keypoints
        keypoints = person_data['keypoints']
        
        # get shoulder and hip keypoints (COCO format)
        left_shoulder = keypoints[5]   # [x, y, conf]
        right_shoulder = keypoints[6]  # [x, y, conf]
        left_hip = keypoints[11]       # [x, y, conf]
        right_hip = keypoints[12]      # [x, y, conf]
        
        # find the highest and lowest points among shoulders and hips
        # this handles both normal and inverted poses correctly
        all_y_coords = [left_shoulder[1], right_shoulder[1], left_hip[1], right_hip[1]]
        upper_y = min(all_y_coords)  # highest point (smallest Y coordinate)
        lower_y = max(all_y_coords)  # lowest point (largest Y coordinate)
        
        # use person bbox for x boundaries
        bbox = person_data['bbox']
        x1, y1, x2, y2 = bbox
        
        # calculate upper body region with margin
        margin_y = int((lower_y - upper_y) * margin_ratio)
        crop_upper_y = max(0, int(upper_y - margin_y))
        crop_lower_y = min(H, int(lower_y + margin_y))
        
        # ensure boundaries are within image and bbox
        x1 = max(0, int(x1))
        y1 = max(0, crop_upper_y)
        x2 = min(W, int(x2))
        y2 = min(H, crop_lower_y)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        # check aspect ratio to filter out abnormal detections
        width_px = x2 - x1
        height_px = y2 - y1
        aspect_ratio = width_px / height_px
        
        # filter out extreme aspect ratios (too wide or too tall)
        # normal human upper body should have reasonable proportions
        if aspect_ratio > 3.0 or aspect_ratio < 0.3:
            logger.debug(f"skipping extraction with abnormal aspect ratio: {aspect_ratio:.2f} (width={width_px}, height={height_px})")
            return None
            
        # crop upper body region
        upper_body_region = img[y1:y2, x1:x2]
        
        # check minimum size
        if upper_body_region.shape[0] < 50 or upper_body_region.shape[1] < 50:
            return None
            
        return upper_body_region
        
    def detect_persons_batch(self, frames: List[FrameData], model: YOLO) -> List[Tuple[FrameData, List[Dict]]]:
        """batch detect persons in multiple frames"""
        if not frames:
            return []
            
        # prepare batch
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
                    
            batch_results.append((frame_data, persons_data))
            
        return batch_results
    
    def process_worker(self, worker_id: int, min_confidence: float, prefix: str,
                      save_debug_frames: bool, margin_ratio: float):
        """worker thread for processing frames"""
        model = self.model_pool[worker_id]
        batch_frames = []
        
        logger.info(f"worker {worker_id} started")
        
        while not self.stop_event.is_set():
            try:
                # collect batch
                while len(batch_frames) < self.batch_size and not self.stop_event.is_set():
                    try:
                        frame_data = self.frame_queue.get(timeout=0.1)
                        batch_frames.append(frame_data)
                    except queue.Empty:
                        if len(batch_frames) > 0:
                            break  # process partial batch
                        continue
                
                if not batch_frames:
                    continue
                
                # batch inference
                batch_results = self.detect_persons_batch(batch_frames, model)
                
                # process each frame result
                for frame_data, persons_data in batch_results:
                    upper_body_images = []
                    debug_frame = None
                    
                    # create debug frame if needed
                    if save_debug_frames:
                        debug_frame = frame_data.frame.copy()
                    
                    # process each person
                    for person_idx, person_data in enumerate(persons_data):
                        if person_data['confidence'] < min_confidence:
                            continue
                            
                        # extract upper body
                        upper_body_img = self.extract_upper_body_from_person(
                            frame_data.frame, person_data, margin_ratio)
                        
                        if upper_body_img is not None:
                            filename = f"{prefix}_{frame_data.video_name}_frame{frame_data.frame_idx:06d}_person{person_idx}.jpg"
                            upper_body_images.append((upper_body_img, filename))
                        
                        # draw debug annotations
                        if save_debug_frames and debug_frame is not None:
                            self.draw_debug_annotations(debug_frame, person_data)
                    
                    # create processed result
                    result = ProcessedResult(frame_data, persons_data, upper_body_images, debug_frame)
                    
                    # add to result queue
                    self.result_queue.put(result)
                    
                    # update stats
                    self.update_stats(frames_processed=1)
                
                # clear batch
                batch_frames = []
                
            except Exception as e:
                logger.error(f"worker {worker_id} error: {e}")
                batch_frames = []
                
        logger.info(f"worker {worker_id} finished")
    
    def draw_debug_annotations(self, debug_frame: np.ndarray, person_data: Dict):
        """draw debug annotations on frame"""
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
    
    def save_worker(self, save_debug_frames: bool):
        """worker thread for saving images"""
        debug_dir = None
        if save_debug_frames:
            debug_dir = os.path.join(self.output_dir, "debug_frames")
            Path(debug_dir).mkdir(exist_ok=True)
        
        logger.info("save worker started")
        
        # use thread pool for parallel image saving
        with ThreadPoolExecutor(max_workers=4) as executor:
            while not self.stop_event.is_set() or not self.result_queue.empty():
                try:
                    result = self.result_queue.get(timeout=0.1)
                    
                    # submit save tasks
                    save_futures = []
                    
                    # save upper body images
                    for upper_body_img, filename in result.upper_body_images:
                        save_path = os.path.join(self.output_dir, filename)
                        future = executor.submit(cv2.imwrite, save_path, upper_body_img)
                        save_futures.append((future, filename, upper_body_img.shape))
                    
                    # save debug frame
                    if save_debug_frames and result.debug_frame is not None:
                        debug_filename = f"debug_{result.frame_data.video_name}_frame{result.frame_data.frame_idx:06d}.jpg"
                        debug_save_path = os.path.join(debug_dir, debug_filename)
                        debug_future = executor.submit(cv2.imwrite, debug_save_path, result.debug_frame)
                        save_futures.append((debug_future, debug_filename, result.debug_frame.shape))
                    
                    # wait for saves to complete and log results
                    for future, filename, shape in save_futures:
                        if future.result():
                            logger.success(f"saved: {filename} (size: {shape})")
                            self.update_stats(images_saved=1)
                        else:
                            logger.error(f"failed to save: {filename}")
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"save worker error: {e}")
        
        logger.info("save worker finished")
    
    def video_reader(self, video_path: str, frame_interval: int):
        """video reader thread"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"failed to open video: {video_path}")
            return
            
        video_name = Path(video_path).stem
        frame_count = 0
        
        logger.info("video reader started")
        
        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
                
            # process frame at specified interval
            if frame_count % frame_interval == 0:
                frame_data = FrameData(frame, frame_count, video_name)
                
                # add to queue (blocking if queue is full)
                try:
                    self.frame_queue.put(frame_data, timeout=1.0)
                    self.update_stats(frames_read=1)
                except queue.Full:
                    logger.warning("frame queue full, dropping frame")
            
            frame_count += 1
        
        cap.release()
        logger.info(f"video reader finished, read {frame_count} frames")
    
    def print_progress(self):
        """progress monitoring thread"""
        logger.info("progress monitor started")
        
        while not self.stop_event.is_set():
            time.sleep(5)  # print every 5 seconds
            stats = self.get_stats()
            
            if stats['start_time']:
                elapsed = time.time() - stats['start_time']
                fps = stats['frames_processed'] / elapsed if elapsed > 0 else 0
                
                logger.info(f"progress: read={stats['frames_read']}, "
                          f"processed={stats['frames_processed']}, "
                          f"saved={stats['images_saved']}, "
                          f"fps={fps:.1f}, "
                          f"queue_size={self.frame_queue.qsize()}")
        
        logger.info("progress monitor finished")
    
    def process_video(self, video_path: str, frame_interval: int = 30, min_confidence: float = 0.5,
                     prefix: str = "upper_body", save_debug_frames: bool = False, margin_ratio: float = 0.1):
        """process video with multi-threading"""
        if not os.path.exists(video_path):
            logger.error(f"video file not found: {video_path}")
            return
        
        # get video info
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        cap.release()
        
        logger.info(f"video info: {total_frames} frames, {fps:.2f} fps, {duration:.2f}s")
        logger.info(f"processing every {frame_interval} frame(s) with {self.num_workers} workers")
        
        # start statistics
        with self.stats['lock']:
            self.stats['start_time'] = time.time()
        
        # start threads
        threads = []
        
        # video reader thread
        reader_thread = threading.Thread(target=self.video_reader, args=(video_path, frame_interval))
        reader_thread.start()
        threads.append(reader_thread)
        
        # worker threads
        for i in range(self.num_workers):
            worker_thread = threading.Thread(
                target=self.process_worker,
                args=(i, min_confidence, prefix, save_debug_frames, margin_ratio)
            )
            worker_thread.start()
            threads.append(worker_thread)
        
        # save worker thread
        save_thread = threading.Thread(target=self.save_worker, args=(save_debug_frames,))
        save_thread.start()
        threads.append(save_thread)
        
        # progress monitor thread
        progress_thread = threading.Thread(target=self.print_progress)
        progress_thread.start()
        threads.append(progress_thread)
        
        # wait for video reader to finish
        reader_thread.join()
        logger.info("video reading completed")
        
        # wait for processing to finish
        while not self.frame_queue.empty():
            logger.info(f"waiting for processing to finish, {self.frame_queue.qsize()} frames remaining")
            time.sleep(1)
        
        # wait for results to be saved
        while not self.result_queue.empty():
            logger.info(f"waiting for saving to finish, {self.result_queue.qsize()} results remaining")
            time.sleep(1)
        
        # stop all threads
        self.stop_event.set()
        
        # wait for all threads to finish
        for thread in threads:
            thread.join()
        
        # final statistics
        stats = self.get_stats()
        elapsed = time.time() - stats['start_time']
        avg_fps = stats['frames_processed'] / elapsed if elapsed > 0 else 0
        
        logger.success(f"processing complete!")
        logger.info(f"total frames read: {stats['frames_read']}")
        logger.info(f"total frames processed: {stats['frames_processed']}")
        logger.info(f"total images saved: {stats['images_saved']}")
        logger.info(f"total time: {elapsed:.2f}s")
        logger.info(f"average fps: {avg_fps:.2f}")
        logger.info(f"output directory: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description='multi-threaded upper body extraction from video using pose detection')
    parser.add_argument('video', help='path to input video file')
    parser.add_argument('--pose-model', default='models/yolo11m-pose.pt', 
                       help='path to pose detection model (default: models/yolo11m-pose.pt)')
    parser.add_argument('--output', '-o', default='upper_body_crops_mt', 
                       help='output directory (default: upper_body_crops_mt)')
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
    parser.add_argument('--workers', '-w', type=int, default=4,
                       help='number of worker threads (default: 4)')
    parser.add_argument('--batch-size', '-b', type=int, default=4,
                       help='batch size for inference (default: 4)')
    
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
    extractor = MultiThreadUpperBodyExtractor(
        pose_model_path=args.pose_model,
        output_dir=args.output,
        num_workers=args.workers,
        batch_size=args.batch_size
    )
    
    # process video
    extractor.process_video(
        video_path=args.video,
        frame_interval=args.interval,
        min_confidence=args.confidence,
        prefix=args.prefix,
        save_debug_frames=args.debug,
        margin_ratio=args.margin
    )

if __name__ == "__main__":
    main()
