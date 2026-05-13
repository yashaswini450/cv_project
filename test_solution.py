"""
test_solution.py
================
Local testing script for the TrafficViolationDetector.

Usage examples:
    # Run on a single image
    python test_solution.py --image data/sample_images/test1.jpg

    # Run on all images in a directory
    python test_solution.py --dir data/sample_images/

    # Run with a ground-truth JSON to compute accuracy
    python test_solution.py --dir data/test_cases/ --gt data/test_cases/labels.json

    # Visualize detections (saves annotated images)
    python test_solution.py --image test.jpg --visualize --output-dir results/
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import cv2

from solution import TrafficViolationDetector
from src.utils import load_image, draw_detections


# ─── Helpers ──────────────────────────────────────────────────────────────────

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS)


def compute_accuracy(predictions: List[Dict], ground_truths: List[Dict]) -> Dict:
    """
    Simple accuracy metric:
    - Violation accuracy: fraction of images where (num_riders, helmet_violations)
      exactly match for every vehicle.
    - OCR accuracy: fraction of plates where text exactly matches ground truth.
    """
    correct_violations = 0
    correct_plates = 0
    total_vehicles = 0
    total_plates = 0

    for pred, gt in zip(predictions, ground_truths):
        pred_v = pred.get("violations", [])
        gt_v = gt.get("violations", [])

        n = max(len(pred_v), len(gt_v))
        for i in range(min(len(pred_v), len(gt_v))):
            total_vehicles += 1
            total_plates += 1

            p = pred_v[i]
            g = gt_v[i]

            if (p["num_riders"] == g["num_riders"] and
                    p["helmet_violations"] == g["helmet_violations"]):
                correct_violations += 1

            if p["license_plate"].upper() == g["license_plate"].upper():
                correct_plates += 1

    violation_acc = correct_violations / (total_vehicles + 1e-9)
    ocr_acc = correct_plates / (total_plates + 1e-9)

    return {
        "violation_accuracy": round(violation_acc, 4),
        "ocr_accuracy": round(ocr_acc, 4),
        "total_vehicles": total_vehicles,
        "total_plates": total_plates,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test TrafficViolationDetector locally")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=Path, help="Path to a single test image")
    group.add_argument("--dir",   type=Path, help="Directory of test images")
    parser.add_argument("--gt",         type=Path, default=None,
                        help="Ground-truth JSON file (list of violation dicts)")
    parser.add_argument("--model-dir",  type=Path, default=Path("./models"))
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--visualize",  action="store_true",
                        help="Save annotated output images")
    parser.add_argument("--output-dir", type=Path, default=Path("results"),
                        help="Directory to save visualizations")
    args = parser.parse_args()

    # ── Collect images ────────────────────────────────────────────────────────
    images = collect_images(args.image if args.image else args.dir)
    if not images:
        print("No images found.")
        return

    # ── Load ground truth (optional) ──────────────────────────────────────────
    ground_truths: Optional[List[Dict]] = None
    if args.gt and args.gt.exists():
        with args.gt.open() as f:
            ground_truths = json.load(f)
        print(f"Loaded {len(ground_truths)} ground-truth entries.")

    # ── Initialize detector ───────────────────────────────────────────────────
    print(f"\nLoading models from {args.model_dir} on device={args.device} …")
    t0 = time.time()
    detector = TrafficViolationDetector(
        model_dir=str(args.model_dir),
        device=args.device,
    )
    print(f"Models loaded in {time.time() - t0:.2f}s\n")

    if args.visualize:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Run predictions ───────────────────────────────────────────────────────
    predictions: List[Dict] = []
    total_time = 0.0

    for img_path in images:
        t_start = time.time()
        result = detector.predict(str(img_path))
        elapsed = time.time() - t_start
        total_time += elapsed

        predictions.append(result)

        n_violations = len(result.get("violations", []))
        print(f"[{img_path.name}] → {n_violations} violation(s) | {elapsed:.3f}s")
        print(f"  {json.dumps(result, indent=4)}")

        # Optional visualization
        if args.visualize:
            img = load_image(img_path)
            annotated = draw_detections(img)  # basic — extend if you pass detections
            out_path = args.output_dir / img_path.name
            cv2.imwrite(str(out_path), annotated)
            print(f"  Saved annotated image to {out_path}")

    print(f"\n{'─' * 60}")
    print(f"Processed {len(images)} image(s) in {total_time:.2f}s "
          f"(avg {total_time / len(images):.3f}s/image)")

    # ── Accuracy report ───────────────────────────────────────────────────────
    if ground_truths and len(ground_truths) == len(predictions):
        metrics = compute_accuracy(predictions, ground_truths)
        print("\n📊 Accuracy Report:")
        print(f"  Violation Accuracy : {metrics['violation_accuracy'] * 100:.2f}%")
        print(f"  OCR Accuracy       : {metrics['ocr_accuracy'] * 100:.2f}%")
        print(f"  Total vehicles     : {metrics['total_vehicles']}")
        print(f"  Total plates       : {metrics['total_plates']}")
    print("─" * 60)


if __name__ == "__main__":
    main()
