#!/usr/bin/env python3
"""
时间段帧提取工具
从视频的指定时间段提取所有帧
"""

import cv2
import os
import shutil
import argparse
from pathlib import Path
from loguru import logger
import time


def extract_time_range_frames(video_path, start_time_sec, end_time_sec, output_folder, prefix="frame"):
    """
    从视频指定时间段提取所有帧
    
    args:
        video_path: 视频文件路径
        start_time_sec: 开始时间（秒）
        end_time_sec: 结束时间（秒）
        output_folder: 输出文件夹
        prefix: 文件名前缀
    
    returns:
        提取的帧数
    """
    
    # 创建输出目录
    output_path = Path(output_folder)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"created output directory: {output_folder}")
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"cannot open video: {video_path}")
        return 0
    
    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"video info:")
    logger.info(f"  total frames: {total_frames}")
    logger.info(f"  fps: {fps:.2f}")
    logger.info(f"  duration: {duration:.2f} seconds ({duration/60:.1f} minutes)")
    logger.info(f"  resolution: {width}x{height}")
    
    # 验证时间范围
    if start_time_sec >= duration:
        logger.error(f"start time ({start_time_sec}s) exceeds video duration ({duration:.2f}s)")
        cap.release()
        return 0
    
    if end_time_sec > duration:
        logger.warning(f"end time ({end_time_sec}s) exceeds video duration ({duration:.2f}s), will extract until end")
        end_time_sec = duration
    
    # 计算帧范围
    start_frame = int(start_time_sec * fps)
    end_frame = int(end_time_sec * fps)
    expected_frames = end_frame - start_frame
    
    logger.info(f"extraction range:")
    logger.info(f"  start time: {start_time_sec}s (frame {start_frame})")
    logger.info(f"  end time: {end_time_sec}s (frame {end_frame})")
    logger.info(f"  expected frames to extract: {expected_frames}")
    
    # 跳转到开始帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # 提取帧
    extracted_count = 0
    current_frame = start_frame
    extract_start_time = time.time()
    
    logger.info("starting frame extraction...")
    
    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"failed to read frame {current_frame}, stopping extraction")
            break
        
        # 保存帧
        frame_number = current_frame - start_frame + 1
        filename = f"{prefix}_{frame_number:06d}.jpg"
        output_file = output_path / filename
        
        success = cv2.imwrite(str(output_file), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        if success:
            extracted_count += 1
            
            # 每100帧或每秒报告一次进度
            if extracted_count % 100 == 0 or extracted_count % int(fps) == 0:
                elapsed = time.time() - extract_start_time
                progress = (current_frame - start_frame) / expected_frames * 100
                current_time = (current_frame - start_frame) / fps + start_time_sec
                logger.info(f"extracted {extracted_count} frames ({progress:.1f}%) - current time: {current_time:.1f}s - {filename}")
        else:
            logger.warning(f"failed to save frame {current_frame}")
        
        current_frame += 1
    
    # 统计信息
    extraction_time = time.time() - extract_start_time
    cap.release()
    
    logger.success(f"extraction completed!")
    logger.info(f"summary:")
    logger.info(f"  extracted frames: {extracted_count}")
    logger.info(f"  expected frames: {expected_frames}")
    logger.info(f"  success rate: {extracted_count/expected_frames*100:.1f}%")
    logger.info(f"  extraction time: {extraction_time:.2f}s")
    logger.info(f"  average fps: {extracted_count/extraction_time:.1f} frames/sec")
    logger.info(f"  output directory: {output_folder}")
    
    # 检查输出文件
    saved_files = list(output_path.glob("*.jpg"))
    logger.info(f"verification: found {len(saved_files)} jpg files in output directory")
    
    if saved_files:
        # 计算文件大小统计
        total_size = sum(f.stat().st_size for f in saved_files)
        avg_size = total_size / len(saved_files)
        logger.info(f"file statistics:")
        logger.info(f"  total size: {total_size/1024/1024:.1f} MB")
        logger.info(f"  average file size: {avg_size/1024:.1f} KB per frame")
    
    return extracted_count


def main():
    parser = argparse.ArgumentParser(description="extract frames from specific time range of video")
    parser.add_argument("video", help="input video file path")
    parser.add_argument("--start", type=float, required=True, help="start time in seconds")
    parser.add_argument("--end", type=float, required=True, help="end time in seconds")
    parser.add_argument("--output", "-o", default="extracted_frames", help="output directory")
    parser.add_argument("--prefix", default="frame", help="output filename prefix")
    
    args = parser.parse_args()
    
    # 验证输入
    if not os.path.exists(args.video):
        logger.error(f"video file not found: {args.video}")
        return 1
    
    if args.start >= args.end:
        logger.error(f"start time ({args.start}s) must be less than end time ({args.end}s)")
        return 1
    
    if args.start < 0:
        logger.error(f"start time cannot be negative: {args.start}s")
        return 1
    
    logger.info(f"time range frame extraction")
    logger.info(f"video: {args.video}")
    logger.info(f"time range: {args.start}s - {args.end}s ({args.end - args.start}s duration)")
    logger.info(f"output: {args.output}")
    
    try:
        extracted_count = extract_time_range_frames(
            video_path=args.video,
            start_time_sec=args.start,
            end_time_sec=args.end,
            output_folder=args.output,
            prefix=args.prefix
        )
        
        if extracted_count > 0:
            logger.success(f"successfully extracted {extracted_count} frames!")
            return 0
        else:
            logger.error("frame extraction failed!")
            return 1
            
    except Exception as e:
        logger.error(f"extraction failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
