from ultralytics import YOLO
path = "C:\\Users\\azusaing\\Desktop\\Code\\tianwan\\__SmokeFire\\weights\\smoke.pt"
[print(f"{k}: {v}") for k, v in YOLO(path).names.items()]