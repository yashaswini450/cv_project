from ultralytics import YOLO
import cv2
from pathlib import Path
img_path = list(Path("data/kaggle/Traffic Violations Analysis Dataset/Test data/no_helmet").glob("*.jpg"))[0]

model = YOLO("models/helmet_detector_v2.pt")
results = model.predict(img_path)
for r in results:
    for box in r.boxes:
        cls_id = int(box.cls[0].item())
        print(f"Class: {model.names[cls_id]}, Conf: {box.conf[0].item()}, Box: {box.xyxy[0].tolist()}")
