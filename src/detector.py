"""
src/detector.py
Two-wheeler detection + rider/helmet detection using YOLOv8.

TwoWheelerDetector:
    Detects motorcycles and scooters. If using the custom model (which has
    a 'motor' class), uses that for best dataset-specific accuracy. Falls back
    to the COCO pretrained yolov8n.pt (class 3 = motorcycle).

RiderHelmetDetector:
    Detects helmet/no-helmet heads over the FULL image, then associates each
    head detection with the nearest vehicle bounding box using spatial overlap.
    This is more reliable than cropping to each vehicle, because head detections
    in a crop are frequently lost due to context changes.

    Strategy:
    1. Run the custom YOLO model on the FULL frame to find helm/kepala detections.
    2. For each vehicle box, collect all head detections that have sufficient
       overlap with the expanded vehicle search region.
    3. Apply NMS within each vehicle's rider list.
    4. Filter out noise (tiny boxes, low confidence).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np

from .utils import box_overlap_ratio, expand_box, nms_boxes, box_area


# ─── COCO class indices ───────────────────────────────────────────────────────
_COCO_TWO_WHEELER_CLASSES = {3: "motorcycle"}   # bicycle(1) excluded by default
_COCO_PERSON_CLASS = 0

# Minimum bounding box area as a fraction of total image area.
_MIN_VEHICLE_AREA_RATIO = 0.003   # 0.3% of image
_MIN_RIDER_AREA_RATIO   = 0.0002  # 0.02% of image (heads are small)


class TwoWheelerDetector:
    """
    Detect two-wheelers (motorcycles / scooters) in an RGB image using YOLOv8.

    Parameters
    ----------
    model_path : str | Path | None
        Path to a YOLOv8 weights file (.pt).
        If None, the standard COCO 'yolov8n.pt' is auto-downloaded.
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

        # Default: COCO motorcycle class (3)
        self._target_classes = list(_COCO_TWO_WHEELER_CLASSES.keys())
        if include_bicycle:
            self._target_classes.append(1)

        # If custom model explicitly has a motor/motorcycle class, prefer it
        if model_path:
            names: Dict[int, str] = self.model.names
            custom_motor_ids = [
                i for i, n in names.items()
                if n.lower() in {"motor", "motorcycle", "bike", "scooter", "triple_riding", "tripleriding"}
            ]
            if custom_motor_ids:
                self._target_classes = custom_motor_ids
                print(f"  [TwoWheelerDetector] Using custom motor class IDs: {custom_motor_ids}")
            else:
                print(f"  [TwoWheelerDetector] No motor class in custom model; "
                      f"keeping COCO class IDs {self._target_classes}")

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
        h, w = img.shape[:2]
        img_area = h * w
        min_area = img_area * _MIN_VEHICLE_AREA_RATIO

        results = self.model.predict(
            source=img,
            conf=self.conf,
            iou=self.iou,
            classes=self._target_classes,
            augment=True,        # ── TTA enabled for robust detection ──
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

                # Skip tiny detections (noise / partial detections)
                if box_area((x1, y1, x2, y2)) < min_area:
                    continue

                detections.append(
                    {
                        "box": (x1, y1, x2, y2),
                        "score": score,
                        "label": self.model.names.get(cls_id, "two_wheeler"),
                    }
                )
        return detections


# ─── Rider + Helmet ──────────────────────────────────────────────────────────

class RiderHelmetDetector:
    """
    Detects riders on a two-wheeler and classifies helmet presence.

    Key insight: head detections (helm / kepala) must be run on the FULL image,
    then associated with each vehicle by spatial overlap. Cropping to the
    vehicle region before running the model causes heads to be missed because
    the model's attention changes when context is removed.

    Parameters
    ----------
    model_path : str | Path | None
        Path to YOLOv8 weights trained for rider + helmet detection.
    person_model_path : str | Path | None
        Path to a COCO-compatible YOLOv8 model for fallback person detection.
    conf_threshold : float
    iou_threshold : float
    vehicle_overlap_thresh : float
        Minimum fraction of a rider's head box that must overlap the expanded
        vehicle search box to be associated with that vehicle.
    device : str
    """

    _HELMET_CLASS_NAMES    = {"helmet", "with_helmet", "with helmet", "helm"}
    _NO_HELMET_CLASS_NAMES = {"no_helmet", "without_helmet", "without helmet",
                               "no helmet", "kepala"}
    _RIDER_CLASS_NAMES     = {"rider", "person", "motorcyclist", "bb"}

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        person_model_path: Optional[str | Path] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        vehicle_overlap_thresh: float = 0.10,
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
            self._motor_ids = [
                i for i, n in names.items() if n.lower() in {"motor", "motorcycle"}
            ]
            self._plate_ids = [
                i for i, n in names.items()
                if n.lower() in {"plat nomor", "plate", "license_plate", "numberplate"}
            ]
            print(f"  [RiderHelmetDetector] helm={self._helmet_ids}, "
                  f"no_helm={self._no_helmet_ids}, riders={self._rider_ids}, "
                  f"plate={self._plate_ids}")
        else:
            weights = str(person_model_path) if person_model_path else "yolov8n.pt"
            self._person_model = YOLO(weights)
            self._person_model.to(device)
            self._motor_ids = []
            self._plate_ids = []

        # Stores last full-frame head detections — populated by detect_all_heads()
        # and consumed by detect() → _detect_with_dedicated_model()
        self._cached_head_detections: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ public

    def detect_all_heads(self, img: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run the helmet model on the FULL image and return all head detections.
        Always runs fresh — call this ONCE per image from solution.py before
        the per-vehicle detect() loop.
        """
        self._cached_head_detections = []

        if not self._use_dedicated:
            return self._cached_head_detections

        h, w = img.shape[:2]
        img_area = h * w
        min_area = img_area * _MIN_RIDER_AREA_RATIO

        # Run on full image to get all head detections.
        # Explicitly pass classes=None to clear any persistent class filters
        # left over if the YOLO predictor instance is shared under the hood.
        results = self._helmet_model.predict(
            source=img,
            conf=self.conf,
            iou=self.iou,
            classes=None,
            augment=True,        # ── TTA enabled for robust head detection ──
            verbose=False,
        )

        if results:
            for result in results:
                if result.boxes is None:
                    continue
                for box_data in result.boxes:
                    cls_id = int(box_data.cls[0].item())
                    score = float(box_data.conf[0].item())
                    x1, y1, x2, y2 = [int(v) for v in box_data.xyxy[0].tolist()]

                    if box_area((x1, y1, x2, y2)) < min_area:
                        continue

                    if cls_id in self._helmet_ids:
                        self._cached_head_detections.append({
                            "box": (x1, y1, x2, y2),
                            "has_helmet": True,
                            "score": score,
                        })
                    elif cls_id in self._no_helmet_ids:
                        self._cached_head_detections.append({
                            "box": (x1, y1, x2, y2),
                            "has_helmet": False,
                            "score": score,
                        })
                    elif cls_id in self._rider_ids:
                        self._cached_head_detections.append({
                            "box": (x1, y1, x2, y2),
                            "has_helmet": False,
                            "score": score,
                        })
                    # Skip motor/plate classes

        # Resolve helm vs kepala conflicts (same head detected as both classes)
        self._cached_head_detections = self._resolve_helm_kepala_conflicts(
            self._cached_head_detections, iou_thresh=0.40, kepala_bias=1.30
        )

        return self._cached_head_detections

    @staticmethod
    def _resolve_helm_kepala_conflicts(
        detections: List[Dict[str, Any]],
        iou_thresh: float = 0.40,
        kepala_bias: float = 1.30,
    ) -> List[Dict[str, Any]]:
        """
        Resolve conflicts where both 'helm' and 'kepala' are detected for the
        same head region (overlapping boxes).

        The model is often uncertain and predicts both classes for the same
        physical head. We resolve conservatively:
          - If helm score is NOT at least `kepala_bias` times higher than the
            kepala score, we declare no-helmet (conservative / penalizes violators).
          - Otherwise, helmet wins.

        Returns a deduplicated list of head detections.
        """
        from .utils import iou as compute_iou

        helmet_dets    = [d for d in detections if d["has_helmet"]]
        no_helmet_dets = [d for d in detections if not d["has_helmet"]]
        resolved: List[Dict[str, Any]] = []
        used_helmet_idx: set = set()

        for nh in no_helmet_dets:
            matched = False
            for i, h in enumerate(helmet_dets):
                if i in used_helmet_idx:
                    continue
                overlap = compute_iou(nh["box"], h["box"])
                if overlap >= iou_thresh:
                    # Same physical head — resolve conflict
                    if h["score"] >= nh["score"] * kepala_bias:
                        # Helmet is substantially more confident → keep helmet
                        resolved.append(h)
                    else:
                        # Ambiguous or kepala is close → conservative: no-helmet
                        resolved.append(nh)
                    used_helmet_idx.add(i)
                    matched = True
                    break
            if not matched:
                resolved.append(nh)

        # Add unmatched helmet detections
        for i, h in enumerate(helmet_dets):
            if i not in used_helmet_idx:
                resolved.append(h)

        return resolved

    def detect(
        self,
        img: np.ndarray,
        vehicle_box: Tuple[int, int, int, int],
    ) -> List[Dict[str, Any]]:
        """
        Return riders associated with a specific vehicle.

        Uses full-frame head detections (cached from detect_all_heads) and
        filters by spatial overlap with the expanded vehicle bounding box.
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
        """Use full-frame detections (already computed by detect_all_heads) and filter by vehicle overlap."""
        h, w = img.shape[:2]

        # Expand vehicle box for spatial association
        # Heads are typically above/around the bike frame
        search_box = expand_box(vehicle_box, 1.4, 2.0, w, h)

        # Use the already-computed full-frame detections (set by detect_all_heads)
        # Do NOT call detect_all_heads() here — that would re-run the model and
        # reset the list, which is wasteful and can cause state issues.
        all_heads = self._cached_head_detections

        # Associate heads with this vehicle
        riders = [
            head for head in all_heads
            if box_overlap_ratio(head["box"], search_box) >= self.overlap_thresh
        ]

        # NMS within this vehicle's riders to remove duplicates
        riders = nms_boxes(riders, iou_threshold=self.iou)

        return riders

    def _detect_with_fallback(
        self,
        img: np.ndarray,
        vehicle_box: Tuple[int, int, int, int],
    ) -> List[Dict[str, Any]]:
        """Fallback: COCO person detector + heuristic."""
        h, w = img.shape[:2]
        search_box = expand_box(vehicle_box, 1.1, 1.2, w, h)

        results = self._person_model.predict(
            source=img,
            conf=self.conf,
            iou=self.iou,
            classes=[_COCO_PERSON_CLASS],
            augment=True,
            verbose=False,
        )

        raw_riders = []
        if not results:
            return raw_riders

        for result in results:
            if result.boxes is None:
                continue
            for box_data in result.boxes:
                score = float(box_data.conf[0].item())
                box = tuple(int(v) for v in box_data.xyxy[0].tolist())
                overlap = box_overlap_ratio(box, search_box)
                if overlap >= self.overlap_thresh:
                    has_helmet = self._heuristic_helmet_check(img, box)
                    raw_riders.append({"box": box, "has_helmet": has_helmet, "score": score})

        return nms_boxes(raw_riders, iou_threshold=self.iou)

    @staticmethod
    def _heuristic_helmet_check(img: np.ndarray,
                                 person_box: Tuple[int, int, int, int]) -> bool:
        x1, y1, x2, y2 = person_box
        h = y2 - y1
        head_crop = img[y1:y1 + h // 5, x1:x2]
        if head_crop.size == 0:
            return False
        hsv = cv2.cvtColor(head_crop, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 40, 80], dtype=np.uint8)
        upper_skin = np.array([25, 170, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_ratio = mask.sum() / (mask.size + 1e-6) / 255.0
        return skin_ratio < 0.30
