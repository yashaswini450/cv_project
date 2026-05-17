"""
Quick debug script: for the first 10 overloading images that FAIL,
print out what the detector actually sees (vehicles, heads, riders).
"""
import torch
from pathlib import Path
from solution import TrafficViolationDetector
from src.utils import load_image

test_root = Path("data/kaggle/Traffic Violations Analysis Dataset/Test data/overloading")
device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Initializing on {device}...")
detector = TrafficViolationDetector(model_dir="models", device=device)

images = sorted([f for f in test_root.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])

fail_count = 0
for img_path in images:
    pred = detector.predict(str(img_path))
    violations = pred.get("violations", [])
    is_correct = any(v.get("num_riders", 0) > 2 for v in violations)
    
    if not is_correct and fail_count < 15:
        fail_count += 1
        # Now do a detailed trace
        img = load_image(str(img_path))
        vehicles = detector._twowheeler_detector.detect(img)
        all_heads = detector._rider_helmet_detector.detect_all_heads(img)
        
        print(f"\n{'='*60}")
        print(f"FAIL #{fail_count}: {img_path.name}")
        print(f"  Vehicles detected: {len(vehicles)}")
        for vi, v in enumerate(vehicles):
            riders = detector._rider_helmet_detector.detect(img, v["box"])
            print(f"  Vehicle {vi}: box={v['box']}, score={v['score']:.3f}")
            print(f"    Riders associated: {len(riders)}")
            for ri, r in enumerate(riders):
                print(f"      Rider {ri}: helmet={r['has_helmet']}, score={r['score']:.3f}, box={r['box']}")
        print(f"  Total heads in frame: {len(all_heads)}")
        for hi, h in enumerate(all_heads):
            print(f"    Head {hi}: helmet={h['has_helmet']}, score={h['score']:.3f}, box={h['box']}")
        print(f"  Final prediction: {pred}")

    if fail_count >= 15:
        break

print(f"\nDone. Showed {fail_count} failures.")
