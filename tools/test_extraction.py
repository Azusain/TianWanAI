# -*- coding:utf-8 -*-
import cv2
import os

def test_frame_extraction():
    """Simple test to extract a few frames from mouse3_1"""
    video_path = r"C:\Users\azusaing\Desktop\田湾\mouse3_1.mp4"
    output_dir = r"C:\Users\azusaing\Desktop\田湾\extracted_frames\test_frames"
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created: {output_dir}")
    
    # Test video reading
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        return False
    
    print(f"Video opened successfully: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}")
    
    # Extract first 5 frames
    for i in range(5):
        ret, frame = cap.read()
        if not ret:
            print(f"Cannot read frame {i}")
            break
            
        output_path = os.path.join(output_dir, f"test_frame_{i:03d}.jpg")
        # Use cv2.imencode to handle paths with Chinese characters
        success, encoded_img = cv2.imencode('.jpg', frame)
        if success:
            with open(output_path, 'wb') as f:
                f.write(encoded_img.tobytes())
            success = True
        else:
            success = False
        
        if success:
            # Check if file was actually created
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                print(f"✓ Saved frame {i}: {output_path} ({file_size} bytes)")
            else:
                print(f"✗ File not created: {output_path}")
        else:
            print(f"✗ cv2.imwrite failed for frame {i}")
    
    cap.release()
    return True

if __name__ == '__main__':
    test_frame_extraction()
