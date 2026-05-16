"""
src/violation_logic.py
Violation classification engine.

Given per-vehicle detections, this module applies the traffic violation rules:
  • Triple Riding  : num_riders > 2
  • Helmet Violation: any rider without a helmet
  • Combined       : both of the above

For every violating vehicle, it also triggers the OCR pipeline to extract
the license plate number.

Improvements over v1:
  - min_rider_score: ignore low-confidence rider detections (reduces FP)
  - min_riders check: skip overloading check when zero riders detected
    (probably an empty / parked bike)
  - helmet_conf_threshold: only count no-helmet riders above a higher
    confidence threshold to cut false positives in helmet images
"""

from __future__ import annotations

from typing import List, Dict, Any, Tuple, Optional

import numpy as np


class ViolationEngine:
    """
    Applies traffic violation rules to detection outputs.

    Parameters
    ----------
    triple_riding_threshold : int
        Number of riders above which triple riding is flagged (default: 2,
        meaning >2 → triple riding).
    min_rider_score : float
        Minimum detection confidence for a rider to be counted.
        Filters out marginal detections that inflate rider counts.
    helmet_conf_threshold : float
        Minimum confidence required to count a 'no-helmet' detection as a
        helmet violation. Set higher than min_rider_score to reduce FP in
        images where all riders have helmets.
    run_ocr_on_all : bool
        If True, run OCR on every vehicle even if no violation is detected.
        Useful for audit logging.
    """

    def __init__(
        self,
        triple_riding_threshold: int = 2,
        min_rider_score: float = 0.35,
        helmet_conf_threshold: float = 0.40,
        run_ocr_on_all: bool = False,
    ):
        self.triple_threshold = triple_riding_threshold
        self.min_rider_score = min_rider_score
        self.helmet_conf_threshold = helmet_conf_threshold
        self.run_ocr_on_all = run_ocr_on_all

    # ------------------------------------------------------------------ public

    def analyze(
        self,
        vehicles: List[Dict[str, Any]],
        img: np.ndarray,
        ocr_reader,                        # LicensePlateOCR instance
    ) -> List[Dict[str, Any]]:
        """
        Analyze all detected vehicles and return violation records.

        Parameters
        ----------
        vehicles : list of vehicle dicts, each containing:
            {
              'box'    : (x1, y1, x2, y2),
              'riders' : [{'box': ..., 'has_helmet': bool, 'score': float}, ...],
              'score'  : float,
              'label'  : str
            }
        img : np.ndarray
            Full-frame BGR image (needed for OCR).
        ocr_reader : LicensePlateOCR
            Initialized OCR reader.

        Returns
        -------
        List of violation dicts (only for vehicles with violations, unless
        run_ocr_on_all is True):
            {
              'num_riders'       : int,
              'helmet_violations': int,
              'triple_riding'    : bool,
              'license_plate'    : str,
              'vehicle_box'      : (x1, y1, x2, y2)
            }
        """
        output: List[Dict[str, Any]] = []

        for vehicle in vehicles:
            result = self._process_vehicle(vehicle, img, ocr_reader)
            if result is not None:
                output.append(result)

        return output

    # --------------------------------------------------------------- internals

    def _process_vehicle(
        self,
        vehicle: Dict[str, Any],
        img: np.ndarray,
        ocr_reader,
    ) -> Optional[Dict[str, Any]]:
        all_riders: List[Dict[str, Any]] = vehicle.get("riders", [])
        vehicle_box: Tuple[int, int, int, int] = vehicle["box"]

        # ── Filter low-confidence rider detections ────────────────────────────
        confident_riders = [
            r for r in all_riders if r.get("score", 0.0) >= self.min_rider_score
        ]

        num_riders = len(confident_riders)

        # ── Helmet violations: only count no-helmet detections above a higher
        #    confidence threshold to suppress false positives ─────────────────
        num_no_helmet = sum(
            1 for r in confident_riders
            if not r.get("has_helmet", True)
            and r.get("score", 0.0) >= self.helmet_conf_threshold
        )

        # ── Handle explicit triple riding class ───────────────────────────────
        is_explicit_triple = (vehicle.get("label", "").lower() in {"triple_riding", "tripleriding"})
        if is_explicit_triple:
            num_riders = max(3, num_riders)

        triple_riding    = num_riders > self.triple_threshold
        helmet_violation = num_no_helmet > 0

        # ── Early exit: no violation and no overriding flag ───────────────────
        has_violation = triple_riding or helmet_violation
        if not has_violation and not self.run_ocr_on_all:
            return None
        if num_riders == 0 and not is_explicit_triple and not self.run_ocr_on_all:
            return None

        # ── OCR ───────────────────────────────────────────────────────────────
        plate_result = ocr_reader.read_plate(img, search_box=vehicle_box)
        plate_text = plate_result.get("text", "")

        return {
            "num_riders": num_riders,
            "helmet_violations": num_no_helmet,
            "triple_riding": triple_riding,
            "license_plate": plate_text,
            "vehicle_box": vehicle_box,
        }

    # ─── Static helpers ──────────────────────────────────────────────────────

    @staticmethod
    def format_output(violations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Format the violation list into the expected submission JSON.

        Output format (per the project spec):
        {
          "violations": [
            {
              "num_riders": <int>,
              "helmet_violations": <int>,
              "license_plate": "<str>"
            },
            ...
          ]
        }
        """
        return {
            "violations": [
                {
                    "num_riders": v["num_riders"],
                    "helmet_violations": v["helmet_violations"],
                    "license_plate": v["license_plate"],
                }
                for v in violations
            ]
        }
