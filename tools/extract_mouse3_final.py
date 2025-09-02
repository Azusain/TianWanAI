# -*- coding:utf-8 -*-
import cv2
import os
import shutil

def extract_frames_from_video(video_path, output_folder, prefix="frame", max_frames=1500):
    """
    Extract frames from video with Chinese character path support
    """
    # Create output directory
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)
    print(f'Created directory: {output_folder}')

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'ERROR: Cannot open video: {video_path}')
        return False
    
    # Get video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    
    print(f'Video: {os.path.basename(video_path)}')
    print(f'Total frames: {total_frames}')
    print(f'FPS: {fps:.2f}')
    print(f'Duration: {duration:.2f} seconds')
    
    # Calculate interval for target frames
    if total_frames <= max_frames:
        interval = 1
        target_frames = total_frames
    else:
        interval = max(1, total_frames // max_frames)
        target_frames = max_frames
    
    print(f'Extracting every {interval} frames, targeting {target_frames} frames')
    
    # Extract frames
    frame_index = 0
    saved_count = 0
    
    while saved_count < target_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_index += 1
        
        # Save frame at specified interval
        if frame_index % interval == 0:
            saved_count += 1
            filename = f"{prefix}_{saved_count:04d}.jpg"
            output_path = os.path.join(output_folder, filename)
            
            # Use cv2.imencode to handle Chinese characters in path
            success, encoded_img = cv2.imencode('.jpg', frame)
            if success:
                with open(output_path, 'wb') as f:
                    f.write(encoded_img.tobytes())
                
                if saved_count % 100 == 0 or saved_count <= 5:
                    file_size = os.path.getsize(output_path)
                    print(f'Saved frame {saved_count}: {filename} ({file_size} bytes)')
            else:
                print(f'Failed to encode frame {saved_count}')
    
    cap.release()
    print(f'✓ Extraction complete! Saved {saved_count} frames')
    return saved_count

def main():
    # Video paths
    base_path = r"C:\Users\azusaing\Desktop\田湾"
    output_base = r"C:\Users\azusaing\Desktop\田湾\extracted_frames"
    
    videos = [
        {
            'file': 'mouse3_1.mp4',
            'output_dir': os.path.join(output_base, 'mouse3_1_frames'),
            'prefix': 'mouse3_1_frame',
            'target': 1500  # Will extract all 574 frames since video is short
        },
        {
            'file': 'mouse3_2.mp4', 
            'output_dir': os.path.join(output_base, 'mouse3_2_frames'),
            'prefix': 'mouse3_2_frame',
            'target': 1500
        }
    ]
    
    total_extracted = 0
    
    for video_info in videos:
        video_path = os.path.join(base_path, video_info['file'])
        
        if not os.path.exists(video_path):
            print(f'ERROR: Video not found: {video_path}')
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing: {video_info['file']}")
        print(f"{'='*60}")
        
        extracted = extract_frames_from_video(
            video_path=video_path,
            output_folder=video_info['output_dir'],
            prefix=video_info['prefix'],
            max_frames=video_info['target']
        )
        
        total_extracted += extracted
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: Extracted {total_extracted} frames total")
    print(f"Files saved to: {output_base}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
