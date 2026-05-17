import cv2
from src.utils import load_image
from src.detector import TwoWheelerDetector, RiderHelmetDetector
from pathlib import Path

img_path = list(Path("data/kaggle/Traffic Violations Analysis Dataset/Test data/no_helmet").glob("*.jpg"))[0]
img = load_image(str(img_path))

two_wheeler_detector = TwoWheelerDetector(model_path="models/helmet_detector_v2.pt", device="cpu")
rider_helmet_detector = RiderHelmetDetector(model_path="models/helmet_detector_v2.pt", device="cpu")

print("VEHICLES:")
vehicles = two_wheeler_detector.detect(img)
print(vehicles)

print("HEADS:")
heads = rider_helmet_detector.detect_all_heads(img)
print(heads)
