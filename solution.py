"""
solution.py
============
Traffic Rule Violation Detection — Main submission file.

Required interface (must not be changed):
    class TrafficViolationDetector:
        def __init__(self, model_dir="./models")
        def predict(self, image_path: str) -> dict

Usage example:
    detector = TrafficViolationDetector(model_dir="./models")
    result   = detector.predict("path/to/image.jpg")
    print(result)
    # {
    #   "violations": [
    #     {"num_riders": 3, "helmet_violations": 2, "license_plate": "MH12AB1234"},
    #     ...
    #   ]
    # }
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, Any

import cv2

# ── Internal modules ──────────────────────────────────────────────────────────
from src.detector import TwoWheelerDetector, RiderHelmetDetector
from src.ocr import LicensePlateOCR
from src.violation_logic import ViolationEngine
from src.utils import load_image


class TrafficViolationDetector:
    """
    End-to-end traffic violation detector.

    All heavy models are loaded once in __init__. The predict() method
    is stateless and can be called repeatedly without reloading.

    Parameters
    ----------
    model_dir : str | Path
        Directory containing model weight files.
        Expected layout:
            model_dir/
            ├── helmet_detector.pt   ← Custom model: motor + helm + kepala + plat nomor
            └── plate_detector.pt    ← (optional) dedicated plate detector

    device : str
        PyTorch device string: 'cpu', 'cuda', 'mps'.
    """

    def __init__(self, model_dir: str = "./models", device: str = "cpu"):
        model_dir = Path(model_dir)

        # ── Resolve model paths ───────────────────────────────────────────────
        helmet_weights = self._resolve(model_dir, "helmet_detector.pt")
        plate_weights  = self._resolve(model_dir, "plate_detector.pt")

        # COCO fallback (for two-wheeler detection when no custom model)
        coco_weights   = self._resolve(model_dir, "yolov8n.pt") or "yolov8n.pt"

        # ── Two-Wheeler Detector ──────────────────────────────────────────────
        # Use the custom model if available — it has a 'motor' class that is
        # fine-tuned for this dataset and gives more reliable detections.
        # Fall back to COCO yolov8n only if no custom model exists.
        self._twowheeler_detector = TwoWheelerDetector(
            model_path=helmet_weights or coco_weights,
            conf_threshold=0.30,
            iou_threshold=0.45,
            include_bicycle=False,
            device=device,
        )

        # ── Rider + Helmet Detector ───────────────────────────────────────────
        # Runs helm/kepala detection on the FULL image (once per predict call),
        # then associates heads with vehicles by spatial overlap.
        self._rider_helmet_detector = RiderHelmetDetector(
            model_path=helmet_weights,
            person_model_path=coco_weights,
            conf_threshold=0.25,    # Lower threshold to catch marginal head detections
            iou_threshold=0.45,
            vehicle_overlap_thresh=0.10,  # Relaxed — heads can be partially above the bike
            device=device,
        )

        # ── OCR ───────────────────────────────────────────────────────────────
        # Share the helmet model's plate class IDs for zero-extra-cost plate detection
        _helmet_yolo = (
            self._rider_helmet_detector._helmet_model
            if self._rider_helmet_detector._use_dedicated else None
        )
        _plate_ids = (
            self._rider_helmet_detector._plate_ids
            if self._rider_helmet_detector._use_dedicated else []
        )

        self._ocr = LicensePlateOCR(
            plate_model_path=plate_weights,
            ocr_backend="easyocr",
            languages=["en"],
            conf_threshold=0.25,
            device=device,
            model_dir=model_dir,
            helmet_model=_helmet_yolo,
            helmet_plate_ids=_plate_ids,
        )

        self._violation_engine = ViolationEngine(
            triple_riding_threshold=2,
            min_rider_score=0.25,
            helmet_conf_threshold=0.30,
            run_ocr_on_all=False,
        )

    # ------------------------------------------------------------------ public

    def predict(self, image_path: str) -> Dict[str, Any]:
        """
        Run the full violation detection pipeline on a single image.

        Returns
        -------
        dict:
            {
              "violations": [
                {
                  "num_riders"       : int,
                  "helmet_violations": int,
                  "license_plate"    : str
                },
                ...
              ]
            }
            Returns {"violations": []} if no violations are detected.
        """
        # ── Load image ────────────────────────────────────────────────────────
        try:
            img = load_image(image_path)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[TrafficViolationDetector] ERROR loading image: {exc}")
            return {"violations": []}

        # ── Step 1: Two-wheeler detection ─────────────────────────────────────
        vehicles = self._twowheeler_detector.detect(img)

        if not vehicles:
            return {"violations": []}

        # ── Step 2: Run full-frame head detection ONCE (for all vehicles) ─────
        # This is the key efficiency + accuracy improvement: the model runs
        # once on the whole image, not once per vehicle crop.
        self._rider_helmet_detector.detect_all_heads(img)

        # ── Step 3: Associate heads with vehicles ─────────────────────────────
        enriched_vehicles = []
        for vehicle in vehicles:
            riders = self._rider_helmet_detector.detect(img, vehicle["box"])
            enriched_vehicles.append({**vehicle, "riders": riders})

        # ── Step 4: Violation classification + OCR ────────────────────────────
        violations = self._violation_engine.analyze(
            vehicles=enriched_vehicles,
            img=img,
            ocr_reader=self._ocr,
        )

        # ── Step 5: Format output ─────────────────────────────────────────────
        return ViolationEngine.format_output(violations)

    # --------------------------------------------------------------- internals

    @staticmethod
    def _resolve(model_dir: Path, filename: str) -> str | None:
        """Return full path to a model file if it exists, else None."""
        p = model_dir / filename
        return str(p) if p.exists() else None


# ─── CLI entry-point (for quick local testing) ────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Traffic Violation Detector")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--model-dir", default="./models", help="Model directory")
    parser.add_argument("--device", default="cpu", help="Torch device (cpu/cuda/mps)")
    args = parser.parse_args()

    detector = TrafficViolationDetector(model_dir=args.model_dir, device=args.device)
    result = detector.predict(args.image)
    print(json.dumps(result, indent=2))
