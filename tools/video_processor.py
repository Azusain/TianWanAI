#!/usr/bin/env python3
"""
Unified Video Frame Processor

Combines all video frame extraction functionalities:
- Basic frame extraction with customizable intervals
- Smart sampling for long videos with timeout protection
- Support for both lossy (JPG) and lossless (PNG) formats
- Batch processing of multiple videos
- Chinese path support
- Robust error handling for problematic videos

Replaces: extract_frames_lossless.py, extract_mouse3_final.py, 
         extract_mouse3_frames.py, extract_mouse_200frames.py, frame-extraction.py
"""

import cv2
import os
import shutil
import time
import argparse
import glob
from pathlib import Path
from typing import Optional, List, Tuple
from loguru import logger

class VideoFrameExtractor:
    def __init__(self, output_format: str = "jpg", quality: int = 95):
        """
        Initialize video frame extractor
        
        Args:
            output_format: 'jpg' for lossy compression, 'png' for lossless
            quality: quality for JPG (1-100) or compression level for PNG (0-9)
        """
        self.output_format = output_format.lower()
        self.quality = quality
        
        # set encoding parameters
        if self.output_format == "jpg":
            self.file_ext = "jpg"
            self.encode_format = ".jpg"
            self.encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif self.output_format == "png":
            self.file_ext = "png" 
            self.encode_format = ".png"
            self.encode_params = [cv2.IMWRITE_PNG_COMPRESSION, min(9, quality)]
        else:
            raise ValueError(f"unsupported format: {output_format}. Use 'jpg' or 'png'")
        
        logger.info(f"initialized extractor: {self.output_format.upper()} format, quality={quality}")
    
    def extract_frames(self, video_path: str, output_folder: str, 
                      prefix: str = "frame", target_frames: int = None,
                      interval: int = None, start_idx: int = 1,
                      max_duration: int = 300, overwrite: bool = True) -> int:
        """
        Extract frames from video with flexible sampling strategies
        
        Args:
            video_path: path to input video
            output_folder: output directory for frames
            prefix: filename prefix for extracted frames
            target_frames: target number of frames to extract (overrides interval)
            interval: fixed frame sampling interval
            start_idx: starting index for output filenames
            max_duration: maximum processing time in seconds (timeout protection)
            overwrite: whether to overwrite existing output folder
            
        Returns:
            number of successfully extracted frames
        """
        video_path = Path(video_path)
        output_folder = Path(output_folder)
        
        if not video_path.exists():
            logger.error(f"video file not found: {video_path}")
            return 0
        
        # prepare output directory
        if output_folder.exists() and overwrite:
            shutil.rmtree(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"output directory: {output_folder}")
        
        # open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"cannot open video: {video_path}")
            return 0
        
        # get video metadata
        total_frames_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_meta = total_frames_meta / fps if fps > 0 else 0
        
        logger.info(f"video: {video_path.name}")
        logger.info(f"metadata - frames: {total_frames_meta}, fps: {fps:.2f}, "
                   f"resolution: {width}x{height}, duration: {duration_meta:.2f}s")
        
        # determine sampling strategy
        if target_frames and interval:
            logger.warning("both target_frames and interval specified, using target_frames")
            interval = None
        
        if target_frames:
            # smart sampling based on target frame count
            if total_frames_meta > 0 and total_frames_meta <= target_frames:
                sample_interval = 1
                effective_target = total_frames_meta
            else:
                # estimate interval for problematic videos
                estimated_total = int(fps * duration_meta) if fps > 0 and duration_meta > 0 else 3000
                sample_interval = max(1, estimated_total // target_frames)
                effective_target = target_frames
            
            logger.info(f"target frames: {target_frames}, estimated interval: {sample_interval}")
        elif interval:
            # fixed interval sampling
            sample_interval = interval
            effective_target = total_frames_meta // interval if total_frames_meta > 0 else None
            logger.info(f"fixed interval: {interval}, estimated output: {effective_target}")
        else:
            # default: extract every 30th frame
            sample_interval = 30
            effective_target = None
            logger.info("using default interval: 30")
        
        # start extraction
        start_time = time.time()
        frame_count = 0
        saved_count = 0
        
        logger.info(f"starting extraction (timeout: {max_duration}s)...")
        
        try:
            while True:
                # check timeout
                elapsed = time.time() - start_time
                if elapsed > max_duration:
                    logger.warning(f"timeout reached ({max_duration}s), stopping extraction")
                    break
                
                # read frame
                ret, frame = cap.read()
                if not ret:
                    logger.info(f"end of video reached at frame {frame_count}")
                    break
                
                frame_count += 1
                
                # check if we should save this frame
                should_save = False
                if target_frames:
                    # target-based sampling
                    if frame_count % sample_interval == 0 or saved_count == 0:
                        should_save = True
                        if saved_count >= target_frames:
                            break
                elif interval:
                    # interval-based sampling
                    if frame_count % sample_interval == 0:
                        should_save = True
                
                if should_save:
                    saved_count += 1
                    filename = f"{prefix}_{saved_count + start_idx - 1:04d}.{self.file_ext}"
                    output_path = output_folder / filename
                    
                    # encode and save frame
                    success, encoded_img = cv2.imencode(self.encode_format, frame, self.encode_params)
                    if success:
                        with open(output_path, 'wb') as f:
                            f.write(encoded_img.tobytes())
                        
                        # log progress
                        if saved_count <= 5 or saved_count % 50 == 0 or saved_count % 25 == 0:
                            file_size = output_path.stat().st_size
                            logger.success(f"frame {saved_count}: {filename} "
                                         f"(video frame {frame_count}, {file_size/1024/1024:.1f}MB, {elapsed:.1f}s)")
                    else:
                        logger.warning(f"failed to encode frame {saved_count}")
                        saved_count -= 1
        
        finally:
            cap.release()
        
        elapsed = time.time() - start_time
        logger.success(f"extraction complete!")
        logger.info(f"processed {frame_count} video frames in {elapsed:.1f}s")
        logger.info(f"saved {saved_count} image frames to {output_folder}")
        
        # verify output
        if output_folder.exists():
            saved_files = list(output_folder.glob(f"*.{self.file_ext}"))
            total_size = sum(f.stat().st_size for f in saved_files)
            avg_size = total_size / len(saved_files) if saved_files else 0
            
            logger.info(f"verification: {len(saved_files)} files, "
                       f"total size: {total_size/1024/1024:.1f}MB, "
                       f"avg size: {avg_size/1024/1024:.2f}MB per frame")
        
        return saved_count
    
    def process_video_list(self, video_configs: List[dict]) -> int:
        """
        Process multiple videos with individual configurations
        
        Args:
            video_configs: list of video configuration dicts, each containing:
                - video_path: str, path to video file
                - output_folder: str, output directory
                - prefix: str, filename prefix (optional)
                - target_frames: int, target frame count (optional)
                - interval: int, sampling interval (optional)
                
        Returns:
            total number of frames extracted across all videos
        """
        total_extracted = 0
        
        for i, config in enumerate(video_configs, 1):
            video_path = config['video_path']
            logger.info(f"\n{'='*60}")
            logger.info(f"processing video {i}/{len(video_configs)}: {Path(video_path).name}")
            logger.info(f"{'='*60}")
            
            if not Path(video_path).exists():
                logger.error(f"video not found: {video_path}")
                continue
            
            extracted = self.extract_frames(
                video_path=video_path,
                output_folder=config['output_folder'],
                prefix=config.get('prefix', 'frame'),
                target_frames=config.get('target_frames'),
                interval=config.get('interval'),
                start_idx=config.get('start_idx', 1),
                max_duration=config.get('max_duration', 300),
                overwrite=config.get('overwrite', True)
            )
            
            total_extracted += extracted
        
        logger.info(f"\n{'='*60}")
        logger.success(f"batch processing complete! total extracted: {total_extracted} frames")
        logger.info(f"{'='*60}")
        
        return total_extracted
    
    def process_directory(self, input_dir: str, output_base: str,
                         target_frames: int = None, interval: int = None,
                         prefix: str = "frame", video_extensions: List[str] = None) -> int:
        """
        Process all videos in a directory
        
        Args:
            input_dir: directory containing video files
            output_base: base output directory (subdirectories created for each video)
            target_frames: target frame count per video
            interval: sampling interval
            prefix: filename prefix
            video_extensions: list of video file extensions to process
            
        Returns:
            total number of frames extracted
        """
        if video_extensions is None:
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        
        input_path = Path(input_dir)
        if not input_path.exists() or not input_path.is_dir():
            logger.error(f"input directory not found: {input_dir}")
            return 0
        
        # find video files
        video_files = []
        for ext in video_extensions:
            video_files.extend(input_path.glob(f"*{ext}"))
            video_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        if not video_files:
            logger.error(f"no video files found in {input_dir}")
            logger.info(f"supported extensions: {video_extensions}")
            return 0
        
        logger.info(f"found {len(video_files)} video files")
        
        # prepare configurations
        video_configs = []
        for video_file in video_files:
            output_folder = Path(output_base) / f"{video_file.stem}_frames"
            video_configs.append({
                'video_path': str(video_file),
                'output_folder': str(output_folder),
                'prefix': f"{prefix}_{video_file.stem}",
                'target_frames': target_frames,
                'interval': interval
            })
        
        return self.process_video_list(video_configs)

def main():
    parser = argparse.ArgumentParser(description="unified video frame extraction tool")
    parser.add_argument("input", help="input video file or directory")
    parser.add_argument("-o", "--output", help="output directory (required for single video)")
    parser.add_argument("-f", "--format", choices=['jpg', 'png'], default='jpg',
                       help="output format: jpg (lossy) or png (lossless)")
    parser.add_argument("-q", "--quality", type=int, default=95,
                       help="quality: 1-100 for jpg, 0-9 for png")
    parser.add_argument("--target-frames", type=int,
                       help="target number of frames to extract")
    parser.add_argument("--interval", type=int,
                       help="frame sampling interval")
    parser.add_argument("--prefix", default="frame",
                       help="filename prefix for extracted frames")
    parser.add_argument("--start-idx", type=int, default=1,
                       help="starting index for output filenames")
    parser.add_argument("--timeout", type=int, default=300,
                       help="maximum processing time per video (seconds)")
    parser.add_argument("--no-overwrite", action="store_true",
                       help="don't overwrite existing output directories")
    
    args = parser.parse_args()
    
    # create extractor
    extractor = VideoFrameExtractor(
        output_format=args.format,
        quality=args.quality
    )
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # single video
        if not args.output:
            logger.error("output directory required for single video")
            return 1
        
        extracted = extractor.extract_frames(
            video_path=str(input_path),
            output_folder=args.output,
            prefix=args.prefix,
            target_frames=args.target_frames,
            interval=args.interval,
            start_idx=args.start_idx,
            max_duration=args.timeout,
            overwrite=not args.no_overwrite
        )
        
        if extracted > 0:
            logger.success(f"successfully extracted {extracted} frames")
        else:
            logger.error("frame extraction failed")
            return 1
            
    elif input_path.is_dir():
        # directory of videos
        output_base = args.output or f"{input_path.name}_extracted_frames"
        
        extracted = extractor.process_directory(
            input_dir=str(input_path),
            output_base=output_base,
            target_frames=args.target_frames,
            interval=args.interval,
            prefix=args.prefix
        )
        
        if extracted > 0:
            logger.success(f"successfully extracted {extracted} frames total")
        else:
            logger.error("no frames extracted")
            return 1
    else:
        logger.error(f"input path does not exist: {input_path}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
