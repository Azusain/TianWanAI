import os
import shutil

base_dir = os.path.dirname(os.path.abspath(__file__))

images_dir = os.path.join(base_dir, 'images')
labels_dir = os.path.join(base_dir, 'labels')

os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

for folder in os.listdir(base_dir):
  folder_path = os.path.join(base_dir, folder)
  if os.path.isdir(folder_path) and folder.startswith('inf_'):
    for file in os.listdir(folder_path):
      file_path = os.path.join(folder_path, file)
      if os.path.isfile(file_path):
        ext = os.path.splitext(file)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
          shutil.copy(file_path, os.path.join(images_dir, file))
        elif ext == '.txt':
          shutil.copy(file_path, os.path.join(labels_dir, file))