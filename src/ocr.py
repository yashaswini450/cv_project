"""
src/ocr.py
License plate detection and OCR pipeline.

Pipeline
--------
1. If the custom helmet model has a 'plat nomor' class, run it on the full
   vehicle crop to locate the plate (most accurate).
2. Otherwise fall back to: YOLO plate detector → contour heuristic.
3. If no plate is found in the bottom-half ROI, retry with the full vehicle crop.
4. Crop and preprocess the plate image (CLAHE + sharpen + binarize).
5. Run EasyOCR (default) or PaddleOCR to extract the plate text.
6. Post-process the raw text with format-aware OCR error corrections.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import cv2
import numpy as np

from .utils import enhance_for_ocr, normalize_plate_text, crop_box, expand_box


class LicensePlateOCR:
    """
    Detects and reads license plates from a BGR image.

    Parameters
    ----------
    plate_model_path : str | Path | None
        Path to a YOLOv8 model trained for license plate detection.
        If None, a contour-based heuristic detector is used as fallback.
    ocr_backend : str
        'easyocr' (default) or 'paddle'.
    languages : list[str]
        Language codes for EasyOCR / PaddleOCR (default: English).
    conf_threshold : float
        Minimum confidence for plate detections.
    device : str
        'cpu', 'cuda', or 'mps'.
    helmet_model : optional YOLO model
        If supplied and it has a 'plat nomor' class, that model is used for
        plate detection within the vehicle crop (no separate plate model needed).
    helmet_plate_ids : list[int]
        Class IDs for the plate class in the helmet model.
    """

    def __init__(
        self,
        plate_model_path: Optional[str | Path] = None,
        ocr_backend: str = "easyocr",
        languages: Optional[List[str]] = None,
        conf_threshold: float = 0.30,
        device: str = "cpu",
        model_dir: Optional[str | Path] = None,      # Added to locate offline OCR weights
        helmet_model=None,           # optional: shared YOLO model that has plate class
        helmet_plate_ids: Optional[List[int]] = None,
    ):
        self.conf = conf_threshold
        self.device = device
        self._use_yolo_plate = False
        self._ocr_backend = ocr_backend.lower()
        self._helmet_model = helmet_model
        self._helmet_plate_ids = helmet_plate_ids or []

        # ── 1. Plate detector ─────────────────────────────────────────────────
        if plate_model_path and Path(plate_model_path).exists():
            from ultralytics import YOLO
            self._plate_model = YOLO(str(plate_model_path))
            self._plate_model.to(device)
            self._use_yolo_plate = True

        # ── 2. OCR reader ─────────────────────────────────────────────────────
        langs = languages or ["en"]
        if self._ocr_backend == "easyocr":
            import easyocr
            use_gpu = device.startswith("cuda")
            
            # ── Strictly Offline Execution (Requirement) ──
            ocr_model_dir = str(Path(model_dir) / "easyocr") if model_dir else None
            self._reader = easyocr.Reader(
                langs, 
                gpu=use_gpu, 
                verbose=False,
                model_storage_directory=ocr_model_dir,
                download_enabled=False
            )
        elif self._ocr_backend == "paddle":
            from paddleocr import PaddleOCR
            self._reader = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        else:
            raise ValueError(f"Unsupported OCR backend: {ocr_backend!r}")

    # ------------------------------------------------------------------ public

    def read_plate(
        self,
        img: np.ndarray,
        search_box: Optional[Tuple[int, int, int, int]] = None,
    ) -> Dict[str, Any]:
        """
        Detect and read the license plate within the given search region.

        Parameters
        ----------
        img : np.ndarray
            Full-frame BGR image.
        search_box : (x1, y1, x2, y2) | None
            Region to restrict plate search (e.g., the two-wheeler bbox).
            If None, the whole image is searched.

        Returns
        -------
        dict with keys:
            'text'  : str   – cleaned plate text (empty string if not found)
            'box'   : tuple | None – plate bbox in full image coordinates
            'score' : float – confidence score
        """
        h, w = img.shape[:2]

        if search_box:
            x1, y1, x2, y2 = search_box
            # Plate search region: bottom 60% of vehicle box
            # (wider than before — was bottom 50%, which cut off some plates)
            mid_y = y1 + int((y2 - y1) * 0.40)
            plate_search_region = (x1, mid_y, x2, y2)
            roi = crop_box(img, plate_search_region, pad=0.1)
            offset = (plate_search_region[0], plate_search_region[1])
        else:
            roi = img
            offset = (0, 0)

        # ── Strategy 1: Use helmet model's plate class if available ──────────
        if self._helmet_model is not None and self._helmet_plate_ids:
            plate_box_local, det_score = self._yolo_detect_with_model(
                self._helmet_model, roi, class_ids=self._helmet_plate_ids
            )
            if plate_box_local is not None:
                return self._finalize_plate(
                    img, roi, plate_box_local, det_score, offset, search_box
                )

        # ── Strategy 2: Dedicated plate YOLO model ───────────────────────────
        if self._use_yolo_plate:
            plate_box_local, det_score = self._yolo_detect_plate(roi)
            if plate_box_local is not None:
                return self._finalize_plate(
                    img, roi, plate_box_local, det_score, offset, search_box
                )

        # ── Strategy 3: Heuristic contour detection in bottom ROI ────────────
        plate_box_local, det_score = self._heuristic_detect_plate(roi)

        # ── Strategy 4: If heuristic failed, retry on full vehicle crop ──────
        if plate_box_local is None and search_box:
            full_roi = crop_box(img, search_box, pad=0.05)
            plate_box_local, det_score = self._heuristic_detect_plate(full_roi)
            if plate_box_local is not None:
                # Recalculate offset to full vehicle box
                offset = (search_box[0], search_box[1])
                roi = full_roi

        if plate_box_local is None:
            # Last resort: use entire ROI as plate region
            plate_box_local = (0, 0, roi.shape[1], roi.shape[0])
            det_score = 0.0

        return self._finalize_plate(
            img, roi, plate_box_local, det_score, offset, search_box
        )

    # --------------------------------------------------------------- internals

    def _finalize_plate(
        self,
        img: np.ndarray,
        roi: np.ndarray,
        plate_box_local: Tuple[int, int, int, int],
        det_score: float,
        offset: Tuple[int, int],
        search_box: Optional[Tuple[int, int, int, int]],
    ) -> Dict[str, Any]:
        """Crop, OCR, and package the plate result."""
        plate_crop = crop_box(roi, plate_box_local, pad=0.05)

        text, ocr_score = self._run_ocr(plate_crop)
        text = normalize_plate_text(text)

        # Convert local plate box → full image coordinates
        if search_box:
            px1, py1, px2, py2 = plate_box_local
            full_box = (
                offset[0] + px1,
                offset[1] + py1,
                offset[0] + px2,
                offset[1] + py2,
            )
        else:
            full_box = None

        return {
            "text": text,
            "box": full_box,
            "score": det_score * ocr_score if det_score > 0 else ocr_score,
        }

    def _detect_plate(
        self, roi: np.ndarray
    ) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
        """Return (plate_box, score) within the ROI using YOLO or heuristic."""
        if self._use_yolo_plate:
            return self._yolo_detect_plate(roi)
        return self._heuristic_detect_plate(roi)

    def _yolo_detect_plate(
        self, roi: np.ndarray
    ) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
        return self._yolo_detect_with_model(self._plate_model, roi)

    @staticmethod
    def _yolo_detect_with_model(
        model,
        roi: np.ndarray,
        class_ids: Optional[List[int]] = None,
        conf: float = 0.25,
    ) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
        """Run any YOLO model on roi and return the highest-confidence detection."""
        kwargs = dict(source=roi, conf=conf, verbose=False)
        if class_ids:
            kwargs["classes"] = class_ids
        results = model.predict(**kwargs)
        best_score = 0.0
        best_box = None
        if results:
            for result in results:
                if result.boxes is None:
                    continue
                for box_data in result.boxes:
                    score = float(box_data.conf[0].item())
                    if score > best_score:
                        best_score = score
                        best_box = tuple(int(v) for v in box_data.xyxy[0].tolist())
        return best_box, best_score

    @staticmethod
    def _heuristic_detect_plate(
        roi: np.ndarray,
    ) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
        """
        Contour-based license plate detector.
        Works reasonably well on clean, well-lit Indian plates.
        Returns (box, confidence=0.5) or (None, 0.0).
        """
        if roi.size == 0:
            return None, 0.0

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = roi.shape[:2]

        candidates = []
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / (bh + 1e-6)
            area_ratio = (bw * bh) / (w * h)
            # Indian plate aspect ratio ≈ 2:1 to 5:1, area 1-20% of ROI
            if 1.5 < aspect < 6.0 and 0.005 < area_ratio < 0.25:
                candidates.append((x, y, x + bw, y + bh, bw * bh))

        if not candidates:
            return None, 0.0

        # Pick largest qualifying contour
        best = max(candidates, key=lambda c: c[4])
        return (best[0], best[1], best[2], best[3]), 0.5

    def _run_ocr(self, plate_img: np.ndarray) -> Tuple[str, float]:
        """Run OCR on a plate crop. Returns (raw_text, confidence)."""
        if plate_img.size == 0:
            return "", 0.0

        # Resize to standard width for better OCR accuracy
        target_w = 300
        scale = target_w / (plate_img.shape[1] + 1e-6)
        new_h = max(1, int(plate_img.shape[0] * scale))
        plate_resized = cv2.resize(
            plate_img,
            (target_w, new_h),
            interpolation=cv2.INTER_CUBIC,
        )
        enhanced = enhance_for_ocr(plate_resized)

        if self._ocr_backend == "easyocr":
            return self._easyocr_read(enhanced)
        elif self._ocr_backend == "paddle":
            return self._paddle_read(enhanced)
        return "", 0.0

    def _easyocr_read(self, img: np.ndarray) -> Tuple[str, float]:
        """Run EasyOCR on a grayscale/binary image with Indian plate format filtering."""
        try:
            results = self._reader.readtext(img, detail=1, paragraph=False)
            if not results:
                return "", 0.0

            texts = [r[1] for r in results]
            scores = [r[2] for r in results]

            # Clean individual fragments
            clean_texts = [re.sub(r"[^A-Z0-9]", "", t.upper()) for t in texts]

            # Search sliding windows of consecutive text blocks for an Indian plate match
            # (XX00XX0000, XX000000, etc.)
            for window in range(1, len(clean_texts) + 1):
                for i in range(len(clean_texts) - window + 1):
                    sub = "".join(clean_texts[i : i + window])
                    # Indian plate check: >= 8 chars, starts with 2 letters and 2 digits, ends with digits
                    if len(sub) >= 8 and sub[:2].isalpha() and sub[2:4].isdigit() and sub[-2:].isdigit():
                        sub_scores = scores[i : i + window]
                        return sub, float(np.mean(sub_scores))

            # Fallback: concatenate all
            combined = " ".join(texts)
            avg_score = float(np.mean(scores))
            return combined, avg_score
        except Exception:
            return "", 0.0

    def _paddle_read(self, img: np.ndarray) -> Tuple[str, float]:
        """Run PaddleOCR on a grayscale/binary image."""
        try:
            results = self._reader.ocr(img, cls=True)
            if not results or not results[0]:
                return "", 0.0
            texts, scores = [], []
            for line in results[0]:
                texts.append(line[1][0])
                scores.append(line[1][1])
            return " ".join(texts), float(np.mean(scores))
        except Exception:
            return "", 0.0
