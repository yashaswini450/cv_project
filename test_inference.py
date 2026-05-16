from ultralytics import YOLO
import cv2

model = YOLO("models/helmet_detector.pt")
results = model("data/kaggle/Traffic Violations Analysis Dataset/Test data/helmet/Test helmmet (1).jpg")
for r in results:
    for box in r.boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        print(f"Class: {model.names[cls_id]}, Conf: {conf:.2f}")
