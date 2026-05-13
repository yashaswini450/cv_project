"""
src/violation_logic.py
Violation classification engine.

Given per-vehicle detections, this module applies the traffic violation rules:
  • Triple Riding  : num_riders > 2
  • Helmet Violation: any rider without a helmet
  • Combined       : both of the above

For every violating vehicle, it also triggers the OCR pipeline to extract
the license plate number.
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
        Number of riders above which triple riding is flagged (default: 2).
    run_ocr_on_all : bool
        If True, run OCR on every vehicle even if no violation is detected.
        Useful for audit logging.
    """

    def __init__(
        self,
        triple_riding_threshold: int = 2,
        run_ocr_on_all: bool = False,
    ):
        self.triple_threshold = triple_riding_threshold
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
        riders: List[Dict[str, Any]] = vehicle.get("riders", [])
        vehicle_box: Tuple[int, int, int, int] = vehicle["box"]

        num_riders = len(riders)
        num_no_helmet = sum(1 for r in riders if not r.get("has_helmet", True))

        triple_riding = num_riders > self.triple_threshold
        helmet_violation = num_no_helmet > 0
        has_violation = triple_riding or helmet_violation

        if not has_violation and not self.run_ocr_on_all:
            return None

        # OCR
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
