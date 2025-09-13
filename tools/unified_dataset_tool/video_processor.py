import cv2
import os
import logging
from pathlib import Path
from typing import List, Optional
import time

logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self):
        self.supported_formats = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v'}
    
    def extract_frames(self, video_path: str, output_dir: str, 
                      frame_interval: int = 30, max_frames: Optional[int] = None) -> List[str]:
        """
        extract frames from video
        
        args:
            video_path: path to input video file
            output_dir: directory to save extracted frames
            frame_interval: extract every N frames (default: 30)
            max_frames: maximum number of frames to extract (None for all)
        
        returns:
            list of extracted frame file paths
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        
        if not video_path.exists():
            raise FileNotFoundError(f"video file not found: {video_path}")
        
        if video_path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"unsupported video format: {video_path.suffix}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"failed to open video: {video_path}")
        
        # get video info
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"processing video: {video_path.name}")
        logger.info(f"total frames: {total_frames}, fps: {fps:.2f}, duration: {duration:.2f}s")
        
        extracted_files = []
        frame_count = 0
        extracted_count = 0
        
        # base name for frames
        base_name = video_path.stem
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # extract frame at specified interval
            if frame_count % frame_interval == 0:
                if max_frames and extracted_count >= max_frames:
                    break
                
                # save frame
                frame_filename = f"{base_name}_frame_{extracted_count:06d}.jpg"
                frame_path = output_dir / frame_filename
                
                cv2.imwrite(str(frame_path), frame)
                extracted_files.append(str(frame_path))
                extracted_count += 1
                
                if extracted_count % 100 == 0:
                    logger.info(f"extracted {extracted_count} frames...")
            
            frame_count += 1
        
        cap.release()
        
        logger.info(f"extraction complete: {extracted_count} frames saved to {output_dir}")
        return extracted_files
    
    def batch_extract_frames(self, video_dir: str, output_dir: str,
                           frame_interval: int = 30, max_frames: Optional[int] = None) -> dict:
        """
        extract frames from all videos in a directory
        
        returns:
            dict mapping video names to extracted frame lists
        """
        video_dir = Path(video_dir)
        output_dir = Path(output_dir)
        
        if not video_dir.exists():
            raise FileNotFoundError(f"video directory not found: {video_dir}")
        
        # find all video files
        video_files = []
        for ext in self.supported_formats:
            video_files.extend(video_dir.glob(f"*{ext}"))
        
        if not video_files:
            raise ValueError(f"no video files found in: {video_dir}")
        
        logger.info(f"found {len(video_files)} video files")
        
        results = {}
        
        for video_file in video_files:
            try:
                # create subdirectory for each video
                video_output_dir = output_dir / video_file.stem
                
                extracted_frames = self.extract_frames(
                    str(video_file), str(video_output_dir), 
                    frame_interval, max_frames
                )
                
                results[video_file.name] = extracted_frames
                
            except Exception as e:
                logger.error(f"failed to process {video_file.name}: {e}")
                results[video_file.name] = []
        
        return results
    
    def get_video_info(self, video_path: str) -> dict:
        """get basic video information"""
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"video file not found: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"failed to open video: {video_path}")
        
        info = {
            'filename': video_path.name,
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        
        info['duration'] = info['total_frames'] / info['fps'] if info['fps'] > 0 else 0
        info['size_mb'] = video_path.stat().st_size / (1024 * 1024)
        
        cap.release()
        return info
