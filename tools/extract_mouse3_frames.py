# -*- coding:utf-8 -*-
import cv2
import os
import shutil

def get_frame_from_video(video_path, output_folder, interval, prefix="frame", start_idx=1, max_frames=1500):
    """
    Extract frames from video with specified parameters
    
    Args:
        video_path: Path to input video file
        output_folder: Output directory for extracted frames
        interval: Frame sampling interval
        prefix: Output filename prefix
        start_idx: Starting index for output files
        max_frames: Maximum number of frames to extract
    """
    # Create output directory
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f'Created output directory: {output_folder}')
    else:
        # Clean existing directory
        shutil.rmtree(output_folder)
        os.makedirs(output_folder)
        print(f'Cleaned and recreated directory: {output_folder}')

    # Open video
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print(f'Error: Could not open video {video_path}')
        return False
    
    # Get video info
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    
    print(f'Video: {os.path.basename(video_path)}')
    print(f'Total frames: {total_frames}')
    print(f'FPS: {fps:.2f}')
    print(f'Duration: {duration:.2f} seconds')
    print(f'Extracting every {interval} frames, max {max_frames} frames')
    
    # Frame extraction
    frame_index = 0  # Current frame in video
    saved_count = 0  # Count of saved frames
    
    while saved_count < max_frames:
        success, frame = video_capture.read()
        if not success:
            print(f'Video reading completed. Extracted {saved_count} frames total.')
            break
        
        frame_index += 1
        
        # Save frame at specified interval
        if frame_index % interval == 0:
            saved_count += 1
            save_name = f"{prefix}_{saved_count:04d}.jpg"
            save_path = os.path.join(output_folder, save_name)
            cv2.imwrite(save_path, frame)
            
            if saved_count % 100 == 0:
                print(f'Saved {saved_count} frames - latest: {save_name}')
    
    video_capture.release()
    print(f'Extraction complete! Saved {saved_count} frames to {output_folder}')
    return True

def calculate_optimal_interval(video_path, target_frames=1500):
    """Calculate optimal sampling interval to get approximately target_frames"""
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        return 1
    
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_capture.release()
    
    if total_frames <= target_frames:
        return 1
    
    interval = max(1, total_frames // target_frames)
    print(f'Video has {total_frames} total frames. Using interval {interval} to get ~{target_frames} frames')
    return interval

def main():
    # Define paths - use absolute paths to avoid issues
    base_path = r"C:\Users\azusaing\Desktop\田湾"
    output_base = r"C:\Users\azusaing\Desktop\田湾\extracted_frames"
    
    videos = [
        {
            'name': 'mouse3_1.mp4',
            'output_folder': os.path.join(output_base, 'mouse3_1_frames'),
            'prefix': 'mouse3_1_frame'
        },
        {
            'name': 'mouse3_2.mp4', 
            'output_folder': os.path.join(output_base, 'mouse3_2_frames'),
            'prefix': 'mouse3_2_frame'
        }
    ]
    
    target_frames = 1500
    
    print(f"Base path: {base_path}")
    print(f"Output base: {output_base}")
    
    for video_info in videos:
        video_path = os.path.join(base_path, video_info['name'])
        
        if not os.path.exists(video_path):
            print(f'Error: Video file not found: {video_path}')
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing: {video_info['name']}")
        print(f"{'='*60}")
        
        # Calculate optimal interval
        interval = calculate_optimal_interval(video_path, target_frames)
        
        # Extract frames
        success = get_frame_from_video(
            video_path=video_path,
            output_folder=video_info['output_folder'],
            interval=interval,
            prefix=video_info['prefix'],
            start_idx=1,
            max_frames=target_frames
        )
        
        if success:
            # Verify extraction
            extracted_files = [f for f in os.listdir(video_info['output_folder']) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"✓ Successfully extracted {len(extracted_files)} frames")
        else:
            print(f"✗ Failed to extract frames from {video_info['name']}")

if __name__ == '__main__':
    main()
