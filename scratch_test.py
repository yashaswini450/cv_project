import cv2
import json
from src.utils import load_image
from solution import TrafficViolationDetector
from pathlib import Path

detector = TrafficViolationDetector(model_dir="models", device="cpu")

# Find an image
no_helmet_img = list(Path("data/kaggle/Traffic Violations Analysis Dataset/Test data/no_helmet").glob("*.jpg"))[0]
overload_img = list(Path("data/kaggle/Traffic Violations Analysis Dataset/Test data/overloading").glob("*.jpg"))[0]

print("NO HELMET:")
pred = detector.predict(str(no_helmet_img))
print(json.dumps(pred, indent=2))

print("OVERLOADING:")
pred = detector.predict(str(overload_img))
print(json.dumps(pred, indent=2))
