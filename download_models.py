"""
download_models.py
==================
Download all required model weights to the ./models directory.

Models downloaded:
  1. yolov8n.pt          — YOLOv8 Nano (COCO) for two-wheeler detection
  2. helmet_detector.pt  — Fine-tuned YOLOv8 for rider + helmet detection
  3. plate_detector.pt   — Fine-tuned YOLOv8 for license plate detection

Run:
    python download_models.py
"""

from __future__ import annotations

import os
import sys
import hashlib
from pathlib import Path

import requests
from tqdm import tqdm


# ─── Model registry ───────────────────────────────────────────────────────────
# Each entry: (save_filename, download_url, expected_sha256_or_None)
#
# NOTE: helmet_detector.pt and plate_detector.pt URLs point to publicly
# available Roboflow / HuggingFace checkpoints. Swap these with your own
# fine-tuned weights when available.
#
MODELS = [
    (
        # Base two-wheeler detector: YOLOv8 nano pretrained on COCO
        # motor class (id=3) is used out-of-the-box for motorcycles
        "yolov8n.pt",
        "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
        None,
    ),
    (
        # Fine-tuned helmet + head + license plate detector
        # Trained on: https://universe.roboflow.com/avisentv6/two-wheeler-violation-soc7k
        # Classes: motor, helm, kepala, plat nomor, bb
        #
        # After running `python train.py --data data/roboflow/data.yaml`,
        # the best checkpoint is automatically saved here as helmet_detector.pt.
        #
        # Placeholder download below is YOLOv8m (will be overwritten by training):
        "helmet_detector.pt",
        "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt",
        None,
    ),
    (
        # License plate detector (fine-tuned YOLOv8n)
        # The Roboflow model already detects 'plat nomor' class, so this is only
        # needed if you want a dedicated plate-only model for higher OCR accuracy.
        # Placeholder: same as yolov8n until a custom model is trained.
        "plate_detector.pt",
        "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
        None,
    ),
]


# ─── Download helper ──────────────────────────────────────────────────────────

def download_file(url: str, dest: Path, expected_sha256: str | None = None) -> bool:
    """Stream-download a file with a progress bar. Returns True on success."""
    print(f"\n  → Downloading {dest.name}")
    print(f"    URL: {url}")

    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"  ✗ Download failed: {exc}")
        return False

    total = int(resp.headers.get("content-length", 0))
    sha256 = hashlib.sha256()

    with dest.open("wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, unit_divisor=1024, leave=False
    ) as pbar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            sha256.update(chunk)
            pbar.update(len(chunk))

    if expected_sha256:
        actual = sha256.hexdigest()
        if actual != expected_sha256:
            print(f"  ✗ Checksum mismatch! Expected {expected_sha256}, got {actual}")
            dest.unlink(missing_ok=True)
            return False

    size_mb = dest.stat().st_size / 1024 / 1024
    print(f"  ✓ Saved to {dest}  ({size_mb:.1f} MB)")
    return True


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)

    total = len(MODELS)
    success_count = 0

    print("=" * 60)
    print("Traffic Violation Detector — Model Download Script")
    print("=" * 60)

    for filename, url, sha256 in MODELS:
        dest = models_dir / filename

        if dest.exists():
            size_mb = dest.stat().st_size / 1024 / 1024
            print(f"\n  ✓ {filename} already exists ({size_mb:.1f} MB) — skipping")
            success_count += 1
            continue

        ok = download_file(url, dest, sha256)
        if ok:
            success_count += 1

    print("\n" + "=" * 60)
    print(f"Done: {success_count}/{total} models ready in {models_dir.resolve()}")

    # Print total size
    total_mb = sum(
        (models_dir / f).stat().st_size
        for f, _, _ in MODELS
        if (models_dir / f).exists()
    ) / 1024 / 1024
    print(f"Total model storage: {total_mb:.1f} MB (limit: 250 MB)")
    print("=" * 60)

    if success_count < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
