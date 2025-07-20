import torch
from ultralytics import YOLO

def main():
  if not torch.cuda.is_available():
    print('gpu not available')
    exit(-1)
  model = YOLO("./models/yolo11s.pt")
  model.train(
    data='train.yaml',
    imgsz=640,
    epochs=50,
    device=[0],
    batch=16
  )
      
if __name__ == "__main__":
  main()
    
      