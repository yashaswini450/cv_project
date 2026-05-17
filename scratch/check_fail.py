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
    img_dir / "overloading/test-overloading (104).jpg",
    img_dir / "overloading/test-overloading (100).jpg",
    img_dir / "overloading/test-overloading (121).jpg",
    img_dir / "no_helmet/test-nohelmet (12).jpg",
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
        
    # Run Step 1b nested suppression
    vehicles = sorted(vehicles, key=lambda v: v["score"], reverse=True)
    kept_vehicles = []
    for v in vehicles:
        is_nested = False
        for k in kept_vehicles:
            if box_overlap_ratio(v["box"], k["box"]) >= 0.70:
                is_nested = True
                break
        if not is_nested:
            kept_vehicles.append(v)
    print(f"Vehicles after nested suppression: {len(kept_vehicles)}")
    for i, v in enumerate(kept_vehicles):
        print(f"  V{i}: box={v['box']}, score={v['score']:.3f}")
        
    # Run head detection
    all_heads = detector._rider_helmet_detector.detect_all_heads(img)
    print(f"All heads detected: {len(all_heads)}")
    for i, hd in enumerate(all_heads):
        print(f"  Head {i}: box={hd['box']}, has_helmet={hd['has_helmet']}, score={hd['score']:.3f}")
        
    # Run person detection
    person_results = detector._person_model.predict(
        source=img, conf=0.30, iou=0.45, classes=[0], verbose=False, imgsz=1024
    )
    all_persons = []
    if person_results:
        for res in person_results:
            if res.boxes is None:
                continue
            for box_data in res.boxes:
                score = float(box_data.conf[0].item())
                bx = tuple(int(v) for v in box_data.xyxy[0].tolist())
                all_persons.append({"box": bx, "score": score})
    print(f"COCO persons detected: {len(all_persons)}")
    for i, p in enumerate(all_persons):
        print(f"  Person {i}: box={p['box']}, score={p['score']:.3f}")
        
    # Run full prediction
    res = detector.predict(str(img_path))
    print("Final predict output:", json.dumps(res, indent=2))
