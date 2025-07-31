import torch
import cv2
import numpy as np
from PIL import Image
from transformers import YolosImageProcessor, YolosForObjectDetection

THRESHHOLD = 0.5

def inference(image, model, feature_extractor):
  encoding = feature_extractor(images=image, return_tensors="pt")
  with torch.no_grad():
      outputs = model(**encoding)
  scores = outputs.logits.softmax(-1)[0, :, :-1]  
  labels = scores.argmax(-1)
  scores = scores.max(-1).values
  boxes = outputs.pred_boxes[0]
  threshold = THRESHHOLD
  selected = torch.where(scores > threshold)
  selected_scores = scores[selected]
  selected_labels = labels[selected]
  selected_boxes = boxes[selected]

  w, h = image.size
  pixel_boxes = []
  for box in selected_boxes:
      cx, cy, bw, bh = box
      x1 = (cx - bw / 2) * w
      x2 = (cx + bw / 2) * w
      y1 = (cy - bh / 2) * h
      y2 = (cy + bh / 2) * h
      pixel_boxes.append([x1.item(), y1.item(), x2.item(), y2.item()])
  return selected_scores, selected_labels, pixel_boxes
  

if __name__ == "__main__":
  # loading models from local.
  model_dir = "./models/fashionpedia"
  feature_extractor = YolosImageProcessor.from_pretrained(
      model_dir, local_files_only=True,
      size={"shortest_edge": 800, "longest_edge": 1333}  
  )
  model = YolosForObjectDetection.from_pretrained(model_dir, local_files_only=True)

  # inference.
  image = Image.open("image.png").convert("RGB")
  selected_scores, selected_labels, pixel_boxes = inference(image, model, feature_extractor)

  # draw results.
  image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
  for score, label_idx, bbox in zip(selected_scores, selected_labels, pixel_boxes):
      label_name = model.config.id2label[label_idx.item()]
      if label_name.__contains__("t-shirt"):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image_cv, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        label = f"{label_name} {score:.2f}"
        cv2.putText(image_cv, "t-shirt", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        break
        
  cv2.imshow("Detection Results", image_cv)
  cv2.waitKey(0)
  cv2.destroyAllWindows()