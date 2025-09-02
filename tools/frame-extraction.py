# -*- coding:utf8 -*-
import cv2
import os
import shutil

def get_frame_from_video(video_name, interval, prefix="res", start_idx=0, max_frames=None):
    # Use English folder name to avoid path issues with Chinese characters
    save_path = 'extracted_frames/'
    is_exists = os.path.exists(save_path)
    if not is_exists:
        os.makedirs(save_path)
        print('path of %s is build' % save_path)
    else:
        shutil.rmtree(save_path)
        os.makedirs(save_path)
        print('path of %s already exist and rebuild' % save_path)

    video_capture = cv2.VideoCapture(video_name)
    # global index
    i = 0
    # saved image index
    j = start_idx - 1
 
    while True:
        success, frame = video_capture.read()
        if not success:
          print('video is all read')
          break
        i += 1
        if i % interval == 0:
            j += 1
            save_name = f"{prefix}_{str(j)}.jpg"
            cv2.imwrite(save_path + save_name, frame)
            if j % 100 == 0:
                print('image of %s is saved' % save_name)
            # Stop if we've reached the maximum number of frames
            if max_frames is not None and j >= max_frames:
                print(f'Extracted {j} frames, reached maximum limit')
                break

if __name__ == '__main__':
    get_frame_from_video(
        video_name='9月1日.mp4',  # 输入视频路径, 会自动创建同名输出文件夹
        interval = 4,             # 采样间隔
        prefix="frame",          # 输出名称前缀
        start_idx=1,             # 输出编号起始位置
        max_frames=5000          # 最大提取帧数
    )
