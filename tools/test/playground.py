import cv2
from loguru import logger
from ultralytics import YOLO

# 加载 YOLO Pose 模型
pose_model = YOLO(r"C:\Users\azusaing\Desktop\Code\tianwan\models\yolo11m-pose.pt")

# img_path = r"C:\Users\azusaing\Desktop\data\debug9.15(final)\inf_6\20250915_171518_tshirt_detection.jpg"
img_list = [
  r"C:\Users\azusaing\Desktop\data\debug9.15(final)\inf_6\20250915_171518_tshirt_detection.jpg",
  r"C:\Users\azusaing\Desktop\data\debug9.15(final)\inf_4\20250915_164147_cigar_detection.jpg"
]

for img_path in img_list:
  img = cv2.imread(img_path)
  if img is None:
      logger.error(f"Failed to read image: {img_path}")
      exit(-1)


  results = pose_model.predict(
    img,
    conf=0
  )

  result_img = results[0].plot()  

  cv2.imshow("YOLO Pose Result", result_img)
  cv2.waitKey(0)
  
cv2.destroyAllWindows()
