import sys
import os
import cv2
import json
from pathlib import Path

# Add workspace to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from solution import TrafficViolationDetector
from src.utils import load_image, expand_box, box_overlap_ratio

detector = TrafficViolationDetector(model_dir="./models", device="mps")

img_dir = Path("data/kaggle/Traffic Violations Analysis Dataset/Test data")

# Let's inspect a few failure cases
images = [
    img_dir / "no_helmet/test-nohelmet (12).jpg",
    img_dir / "no_helmet/test-nohelmet (13).jpg",
    img_dir / "no_helmet/test-nohelmet (19).jpg",
]

for img_path in images:
    if not img_path.exists():
        print(f"File not found: {img_path}")
        continue
    print(f"\n=======================================================")
    print(f"IMAGE: {img_path.name}")
    print(f"=======================================================")
    
    img = load_image(str(img_path))
    h, w = img.shape[:2]
    
    # Run two-wheeler detector
    vehicles = detector._twowheeler_detector.detect(img)
    print(f"Raw vehicles detected: {len(vehicles)}")
    for i, v in enumerate(vehicles):
        print(f"  V{i}: box={v['box']}, score={v['score']:.3f}")
        
    # Run head detection
    all_heads = detector._rider_helmet_detector.detect_all_heads(img)
    print(f"All heads detected: {len(all_heads)}")
    for i, hd in enumerate(all_heads):
        print(f"  Head {i}: box={hd['box']}, has_helmet={hd['has_helmet']}, score={hd['score']:.3f}")
        
    # Run full prediction
    res = detector.predict(str(img_path))
    print("Final predict output:", json.dumps(res, indent=2))
