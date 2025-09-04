from ultralytics import YOLO
path = "C:\\Users\\azusaing\\Desktop\\Code\\tianwan\\models\\tshirt_cls\\weights\\tshirt_cls_v1.pt"
[print(f"{k}: {v}") for k, v in YOLO(path).names.items()]