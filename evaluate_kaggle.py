"""
evaluate_kaggle.py
==================
Evaluate the TrafficViolationDetector on the Kaggle test dataset.

Test structure:
  data/kaggle/Traffic Violations Analysis Dataset/Test data/
    ├── helmet/       → No violation expected (violations list should be empty)
    ├── no_helmet/    → Helmet violation expected (any v["helmet_violations"] > 0)
    └── overloading/  → Triple riding expected (any v["num_riders"] > 2)

Metrics reported:
  - Per-category accuracy
  - Overall accuracy
  - False Positive / False Negative counts per category
  - List of misclassified images (up to 10 per category)
"""

import os
import json
import time
from pathlib import Path
from collections import defaultdict
import torch
from solution import TrafficViolationDetector


def evaluate():
    test_root = Path("data/kaggle/Traffic Violations Analysis Dataset/Test data")
    model_dir = "models"
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"Initializing detector on {device}...")
    detector = TrafficViolationDetector(model_dir=model_dir, device=device)

    folders = ["helmet", "no_helmet", "overloading"]
    results = {}
    errors = defaultdict(list)  # category → list of (filename, pred_summary)

    for folder in folders:
        folder_path = test_root / folder
        if not folder_path.exists():
            print(f"Skipping {folder}: path not found")
            continue

        print(f"\nEvaluating folder: {folder}...")
        images = sorted([
            f for f in folder_path.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ])

        correct    = 0
        false_pos  = 0   # predicted violation when there should be none
        false_neg  = 0   # missed violation when there should be one
        total      = len(images)

        t0 = time.time()

        # Fixed: use enumerate instead of O(n²) images.index()
        for idx, img_path in enumerate(images, 1):
            pred       = detector.predict(str(img_path))
            violations = pred.get("violations", [])

            is_correct  = False
            is_fp       = False
            is_fn       = False

            if folder == "helmet":
                # Expecting NO violations (all riders have helmets)
                is_correct = (len(violations) == 0)
                if not is_correct:
                    is_fp = True   # false positive violation
            elif folder == "no_helmet":
                # Expecting at least one helmet violation
                is_correct = any(v.get("helmet_violations", 0) > 0 for v in violations)
                if not is_correct:
                    is_fn = True   # missed the helmet violation
            elif folder == "overloading":
                # Expecting at least one vehicle with >2 riders
                is_correct = any(v.get("num_riders", 0) > 2 for v in violations)
                if not is_correct:
                    is_fn = True   # missed the overloading violation

            if is_correct:
                correct += 1
            else:
                # Summarise prediction for debugging
                pred_summary = (
                    f"violations={len(violations)}, "
                    + ", ".join(
                        f"riders={v.get('num_riders',0)} "
                        f"no_helmet={v.get('helmet_violations',0)}"
                        for v in violations
                    ) if violations else "no violations"
                )
                errors[folder].append((img_path.name, pred_summary))

            if is_fp:
                false_pos += 1
            if is_fn:
                false_neg += 1

            # Print progress every 10 images
            if idx % 10 == 0:
                elapsed = time.time() - t0
                fps = idx / elapsed
                print(f"  [{idx}/{total}]  {fps:.1f} img/s  "
                      f"running accuracy: {correct/idx*100:.1f}%")

        elapsed = time.time() - t0
        results[folder] = {
            "correct":   correct,
            "total":     total,
            "accuracy":  round(correct / total, 4) if total > 0 else 0,
            "false_pos": false_pos,
            "false_neg": false_neg,
            "time_s":    round(elapsed, 1),
        }

    # ── Print report ─────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("📊  KAGGLE VIOLATION ACCURACY REPORT")
    print("=" * 55)
    print(f"{'Category':<15} {'Acc':>7}  {'Correct':>8}  {'FP':>5}  {'FN':>5}  {'Time':>6}")
    print("-" * 55)

    overall_correct = 0
    overall_total   = 0

    for folder, stats in results.items():
        print(
            f"{folder.upper():<15} "
            f"{stats['accuracy']*100:>6.2f}%  "
            f"{stats['correct']:>4}/{stats['total']:<4}  "
            f"{stats['false_pos']:>4}FP  "
            f"{stats['false_neg']:>4}FN  "
            f"{stats['time_s']:>5.1f}s"
        )
        overall_correct += stats["correct"]
        overall_total   += stats["total"]

    if overall_total > 0:
        print("-" * 55)
        overall_acc = overall_correct / overall_total * 100
        print(f"{'OVERALL':<15} {overall_acc:>6.2f}%  "
              f"{overall_correct:>4}/{overall_total:<4}")

    print("=" * 55)

    # ── Print misclassified examples ─────────────────────────────────────────
    max_errors_shown = 10
    for folder, err_list in errors.items():
        if err_list:
            print(f"\n❌ Misclassified in '{folder}' "
                  f"({len(err_list)} / {results[folder]['total']}):")
            for fname, summary in err_list[:max_errors_shown]:
                print(f"   {fname}: {summary}")
            if len(err_list) > max_errors_shown:
                print(f"   ... and {len(err_list) - max_errors_shown} more")


if __name__ == "__main__":
    evaluate()
