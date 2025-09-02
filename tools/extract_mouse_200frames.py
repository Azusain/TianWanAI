# -*- coding:utf-8 -*-
import cv2
import os
import shutil

def extract_frames_from_video(video_path, output_folder, prefix="frame", target_frames=200):
    """
    Extract specified number of frames from video with Chinese character path support
    Handles problematic videos with incorrect metadata
    """
    import time
    
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
    
    # Get video info (may be incorrect for problematic videos)
    total_frames_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration_meta = total_frames_meta / fps if fps > 0 else 0
    
    print(f'Video: {os.path.basename(video_path)}')
    print(f'Metadata - Total frames: {total_frames_meta}, FPS: {fps:.2f}, Duration: {duration_meta:.2f} seconds')
    print(f'⚠️ Note: Metadata may be incorrect, will read actual frames')
    
    # Use sequential reading with target frame count and timeout
    print(f'Target: Extract {target_frames} frames from video (assuming ~2-3 minutes)')
    
    # Sequential reading approach for problematic videos
    # Read frames sequentially with smart sampling
    saved_count = 0
    frame_count = 0
    start_time = time.time()
    max_duration = 180  # 3 minutes timeout
    
    # Estimate sampling interval (assuming 2-3 minute video at 25-30 fps = ~3000-5000 frames)
    estimated_total = int(fps * 150) if fps > 0 else 3000  # assume 2.5 minutes
    sample_interval = max(1, estimated_total // target_frames)
    
    print(f'Using sequential reading with estimated interval: {sample_interval}')
    print(f'Timeout: {max_duration} seconds')
    
    while saved_count < target_frames:
        current_time = time.time()
        if current_time - start_time > max_duration:
            print(f'⏰ Timeout reached ({max_duration}s), stopping extraction')
            break
            
        ret, frame = cap.read()
        if not ret:
            print(f'📹 End of video reached at frame {frame_count}')
            break
        
        frame_count += 1
        
        # Sample frame at interval
        if frame_count % sample_interval == 0 or saved_count == 0:  # Always save first frame
            saved_count += 1
            filename = f"{prefix}_{saved_count:04d}.jpg"
            output_path = os.path.join(output_folder, filename)
            
            # Use cv2.imencode to handle Chinese characters in path
            success, encoded_img = cv2.imencode('.jpg', frame)
            if success:
                with open(output_path, 'wb') as f:
                    f.write(encoded_img.tobytes())
                
                if saved_count % 50 == 0 or saved_count <= 5 or saved_count % 25 == 0:
                    file_size = os.path.getsize(output_path)
                    elapsed = current_time - start_time
                    print(f'✅ Frame {saved_count}: {filename} (video frame {frame_count}, {file_size} bytes, {elapsed:.1f}s)')
            else:
                print(f'❌ Failed to encode frame {saved_count}')
                saved_count -= 1
    
    print(f'📊 Final stats: Read {frame_count} video frames, saved {saved_count} image frames')
    
    cap.release()
    print(f'✓ Extraction complete! Saved {saved_count} frames')
    return saved_count

def main():
    """Extract 200 frames from mouse video"""
    print("🎬 老鼠视频抽帧工具 - 抽取200帧")
    print("=" * 50)
    
    # Video path
    video_path = r"C:\Users\azusaing\Desktop\tianwan_dataset\老鼠 5M0DTW102TV.mp4"
    output_folder = r"C:\Users\azusaing\Desktop\tianwan_dataset\mouse_200frames"
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f'❌ 视频文件不存在: {video_path}')
        return
    
    print(f"📹 输入视频: {video_path}")
    print(f"📁 输出目录: {output_folder}")
    print(f"🎯 目标帧数: 200")
    
    # Extract frames
    extracted_count = extract_frames_from_video(
        video_path=video_path,
        output_folder=output_folder,
        prefix="mouse_frame",
        target_frames=200
    )
    
    if extracted_count > 0:
        print(f"\n✅ 抽帧完成!")
        print(f"📊 实际抽取: {extracted_count} 帧")
        print(f"📁 保存位置: {output_folder}")
        
        # Verify output
        if os.path.exists(output_folder):
            frame_files = [f for f in os.listdir(output_folder) if f.endswith('.jpg')]
            print(f"🔍 验证: 发现 {len(frame_files)} 个图片文件")
    else:
        print("❌ 抽帧失败!")

if __name__ == '__main__':
    main()
