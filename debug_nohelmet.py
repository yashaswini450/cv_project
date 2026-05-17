"""
Quick debug: check why no_helmet images fail (19/100 missed).
"""
import torch
from pathlib import Path
from solution import TrafficViolationDetector
from src.utils import load_image

test_root = Path("data/kaggle/Traffic Violations Analysis Dataset/Test data/no_helmet")
device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Initializing on {device}...")
detector = TrafficViolationDetector(model_dir="models", device=device)

images = sorted([f for f in test_root.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])

fail_count = 0
for img_path in images:
    pred = detector.predict(str(img_path))
    violations = pred.get("violations", [])
    is_correct = any(v.get("helmet_violations", 0) > 0 for v in violations)
    
    if not is_correct and fail_count < 10:
        fail_count += 1
        img = load_image(str(img_path))
        vehicles = detector._twowheeler_detector.detect(img)
        all_heads = detector._rider_helmet_detector.detect_all_heads(img)
        
        print(f"\nFAIL #{fail_count}: {img_path.name}")
        print(f"  Vehicles: {len(vehicles)}")
        for vi, v in enumerate(vehicles):
            riders = detector._rider_helmet_detector.detect(img, v["box"])
            print(f"  V{vi}: box={v['box']}, score={v['score']:.3f}, riders={len(riders)}")
            for r in riders:
                print(f"    helmet={r['has_helmet']}, score={r['score']:.3f}")
        print(f"  All heads: {len(all_heads)}")
        for h in all_heads:
            print(f"    helmet={h['has_helmet']}, score={h['score']:.3f}")

print(f"\nShowed {fail_count} failures.")
