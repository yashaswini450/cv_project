"""
src/detector.py
Two-wheeler detection + rider/helmet detection using YOLOv8.

TwoWheelerDetector:
    Detects motorcycles and scooters using YOLOv8 pretrained on COCO.
    COCO class 3 = 'motorcycle'.

RiderHelmetDetector:
    Detects riders (persons) within a two-wheeler crop, and classifies
    whether each rider is wearing a helmet using a fine-tuned YOLOv8 model.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np

from .utils import box_overlap_ratio, expand_box


# ─── COCO class indices ───────────────────────────────────────────────────────
_COCO_TWO_WHEELER_CLASSES = {3: "motorcycle"}   # bicycle(1) excluded by default
_COCO_PERSON_CLASS = 0


class TwoWheelerDetector:
    """
    Detect two-wheelers (motorcycles / scooters) in an RGB image using YOLOv8.

    Parameters
    ----------
    model_path : str | Path | None
        Path to a custom YOLOv8 weights file (.pt).
        If None, uses the standard COCO-pretrained 'yolov8n.pt' (auto-downloaded).
    conf_threshold : float
        Minimum confidence to keep a detection.
    iou_threshold : float
        NMS IoU threshold.
    include_bicycle : bool
        If True, also include COCO class 1 (bicycle) detections.
    device : str
        Torch device string, e.g. 'cpu', 'cuda', 'mps'.
    """

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        include_bicycle: bool = False,
        device: str = "cpu",
    ):
        from ultralytics import YOLO

        weights = str(model_path) if model_path else "yolov8n.pt"
        self.model = YOLO(weights)
        self.model.to(device)
        self.conf = conf_threshold
        self.iou = iou_threshold
        self.device = device

        self._target_classes = list(_COCO_TWO_WHEELER_CLASSES.keys())
        if include_bicycle:
            self._target_classes.append(1)

    # ------------------------------------------------------------------ public

    def detect(self, img: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run inference on a BGR image and return two-wheeler detections.

        Returns
        -------
        List of dicts:
            {
              'box'   : (x1, y1, x2, y2)  – pixel coordinates,
              'score' : float,
              'label' : str
            }
        """
        results = self.model.predict(
            source=img,
            conf=self.conf,
            iou=self.iou,
            classes=self._target_classes,
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []
        if not results:
            return detections

        for result in results:
            if result.boxes is None:
                continue
            for box_data in result.boxes:
                cls_id = int(box_data.cls[0].item())
                score = float(box_data.conf[0].item())
                x1, y1, x2, y2 = [int(v) for v in box_data.xyxy[0].tolist()]
                detections.append(
                    {
                        "box": (x1, y1, x2, y2),
                        "score": score,
                        "label": _COCO_TWO_WHEELER_CLASSES.get(cls_id, "two_wheeler"),
                    }
                )

        return detections


# ─── Rider + Helmet ──────────────────────────────────────────────────────────

class RiderHelmetDetector:
    """
    Detects riders on a two-wheeler and classifies helmet presence.

    Strategy
    --------
    1. Run a YOLO model that has classes: 'rider'/'person', 'helmet', 'no_helmet'.
       (The model is expected to have been trained on a helmet detection dataset.)
    2. If a dedicated helmet model is not available, fall back to:
       a. Detect persons using the standard COCO YOLOv8 model.
       b. Classify the upper-third of each person crop as helmet / no_helmet
          using a simple brightness + shape heuristic (placeholder until
          a real classifier is loaded).

    Parameters
    ----------
    model_path : str | Path | None
        Path to YOLOv8 weights trained for rider + helmet detection.
        Supported class layout:
            0 = 'rider' or 'person'
            1 = 'helmet'
            2 = 'no_helmet'
        If None, falls back to the COCO person detector + heuristic.
    person_model_path : str | Path | None
        Path to a COCO-compatible YOLOv8 model for person detection
        (used in fallback mode).
    conf_threshold : float
    iou_threshold : float
    vehicle_overlap_thresh : float
        Minimum fraction of a rider box that must overlap the vehicle box
        to be considered a rider on that vehicle.
    device : str
    """

    # Roboflow dataset classes: helm=helmet, kepala=bare head, motor=motorcycle
    _HELMET_CLASS_NAMES = {"helmet", "with_helmet", "with helmet", "helm"}
    _NO_HELMET_CLASS_NAMES = {"no_helmet", "without_helmet", "without helmet", "no helmet", "kepala"}
    _RIDER_CLASS_NAMES = {"rider", "person", "motorcyclist", "bb"}

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        person_model_path: Optional[str | Path] = None,
        conf_threshold: float = 0.30,
        iou_threshold: float = 0.45,
        vehicle_overlap_thresh: float = 0.25,
        device: str = "cpu",
    ):
        from ultralytics import YOLO

        self.conf = conf_threshold
        self.iou = iou_threshold
        self.overlap_thresh = vehicle_overlap_thresh
        self.device = device
        self._use_dedicated = False

        if model_path and Path(model_path).exists():
            self._helmet_model = YOLO(str(model_path))
            self._helmet_model.to(device)
            self._use_dedicated = True
            # Build class-name → id mapping
            names: Dict[int, str] = self._helmet_model.names
            self._helmet_ids = [
                i for i, n in names.items() if n.lower() in self._HELMET_CLASS_NAMES
            ]
            self._no_helmet_ids = [
                i for i, n in names.items() if n.lower() in self._NO_HELMET_CLASS_NAMES
            ]
            self._rider_ids = [
                i for i, n in names.items() if n.lower() in self._RIDER_CLASS_NAMES
            ]
        else:
            # Fallback: COCO person model
            weights = str(person_model_path) if person_model_path else "yolov8n.pt"
            self._person_model = YOLO(weights)
            self._person_model.to(device)

    # ------------------------------------------------------------------ public

    def detect(
        self,
        img: np.ndarray,
        vehicle_box: Tuple[int, int, int, int],
    ) -> List[Dict[str, Any]]:
        """
        Detect riders on a specific two-wheeler and classify helmet status.

        Parameters
        ----------
        img : np.ndarray
            Full-frame BGR image.
        vehicle_box : (x1, y1, x2, y2)
            Bounding box of the two-wheeler in the full frame.

        Returns
        -------
        List of dicts per rider:
            {
              'box'        : (x1, y1, x2, y2),
              'has_helmet' : bool,
              'score'      : float
            }
        """
        if self._use_dedicated:
            return self._detect_with_dedicated_model(img, vehicle_box)
        else:
            return self._detect_with_fallback(img, vehicle_box)

    # --------------------------------------------------------------- internals

    def _detect_with_dedicated_model(
        self,
        img: np.ndarray,
        vehicle_box: Tuple[int, int, int, int],
    ) -> List[Dict[str, Any]]:
        """Use the dedicated rider+helmet YOLOv8 model."""
        h, w = img.shape[:2]
        # Expand vehicle box slightly for context
        search_box = expand_box(vehicle_box, 1.1, 1.15, w, h)

        results = self._helmet_model.predict(
            source=img,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )

        helmet_boxes: List[Tuple[int, int, int, int]] = []
        no_helmet_boxes: List[Tuple[int, int, int, int]] = []
        rider_boxes: List[Tuple[int, int, int, int]] = []

        if results:
            for result in results:
                if result.boxes is None:
                    continue
                for box_data in result.boxes:
                    cls_id = int(box_data.cls[0].item())
                    score = float(box_data.conf[0].item())
                    box = tuple(int(v) for v in box_data.xyxy[0].tolist())
                    if cls_id in self._rider_ids:
                        rider_boxes.append((box, score))
                    elif cls_id in self._helmet_ids:
                        helmet_boxes.append(box)
                    elif cls_id in self._no_helmet_ids:
                        no_helmet_boxes.append(box)

        # Filter riders that overlap with this vehicle
        riders = []
        for (box, score) in rider_boxes:
            overlap = box_overlap_ratio(box, search_box)
            if overlap >= self.overlap_thresh:
                # Determine helmet status by checking if a helmet box overlaps head region
                head_box = self._head_region(box)
                has_helmet = any(
                    box_overlap_ratio(hb, head_box) > 0.3 for hb in helmet_boxes
                )
                riders.append({"box": box, "has_helmet": has_helmet, "score": score})

        return riders

    def _detect_with_fallback(
        self,
        img: np.ndarray,
        vehicle_box: Tuple[int, int, int, int],
    ) -> List[Dict[str, Any]]:
        """
        Fallback: detect persons with COCO model, then use a heuristic to
        guess helmet status (darker/rounder head region = helmet).
        NOTE: This is an approximation; replace with a real classifier.
        """
        h, w = img.shape[:2]
        search_box = expand_box(vehicle_box, 1.1, 1.2, w, h)

        results = self._person_model.predict(
            source=img,
            conf=self.conf,
            iou=self.iou,
            classes=[_COCO_PERSON_CLASS],
            verbose=False,
        )

        riders = []
        if not results:
            return riders

        for result in results:
            if result.boxes is None:
                continue
            for box_data in result.boxes:
                score = float(box_data.conf[0].item())
                box = tuple(int(v) for v in box_data.xyxy[0].tolist())
                overlap = box_overlap_ratio(box, search_box)
                if overlap >= self.overlap_thresh:
                    has_helmet = self._heuristic_helmet_check(img, box)
                    riders.append({"box": box, "has_helmet": has_helmet, "score": score})

        return riders

    @staticmethod
    def _head_region(person_box: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Return the upper-quarter of a person bounding box (approximate head area)."""
        x1, y1, x2, y2 = person_box
        h = y2 - y1
        return (x1, y1, x2, y1 + h // 4)

    @staticmethod
    def _heuristic_helmet_check(img: np.ndarray,
                                 person_box: Tuple[int, int, int, int]) -> bool:
        """
        Very rough heuristic: if the head crop is not predominantly skin-colored,
        classify as 'helmet'. This is a placeholder — a proper classifier should
        replace this for production use.
        """
        x1, y1, x2, y2 = person_box
        h = y2 - y1
        # Take top 20% of the person box as head region
        head_crop = img[y1:y1 + h // 5, x1:x2]
        if head_crop.size == 0:
            return False  # conservative guess

        # Convert to HSV and check for skin-tone dominance
        hsv = cv2.cvtColor(head_crop, cv2.COLOR_BGR2HSV)
        # Skin tone: H=0-25, S=40-170, V=80-255
        lower_skin = np.array([0, 40, 80], dtype=np.uint8)
        upper_skin = np.array([25, 170, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_ratio = mask.sum() / (mask.size + 1e-6) / 255.0

        # If skin covers < 30% of head area → likely has helmet
        return skin_ratio < 0.30
