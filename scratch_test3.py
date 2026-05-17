from ultralytics import YOLO
model = YOLO("models/helmet_detector_v2.pt")
print(model.names)
