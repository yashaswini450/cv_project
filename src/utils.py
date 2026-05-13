"""
src/utils.py
Shared utility functions for image loading, drawing, and preprocessing.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, List, Tuple, Optional


# ─── Image I/O ───────────────────────────────────────────────────────────────

def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    Load an image from disk and return it as a BGR numpy array.
    Raises FileNotFoundError if the path does not exist.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"OpenCV could not decode image: {path}")
    return img


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert BGR (OpenCV default) to RGB."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(img: np.ndarray) -> np.ndarray:
    """Convert RGB to BGR."""
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# ─── Region Cropping ─────────────────────────────────────────────────────────

def crop_box(img: np.ndarray, box: Tuple[int, int, int, int],
             pad: float = 0.0) -> np.ndarray:
    """
    Crop a region from the image using a bounding box (x1, y1, x2, y2).
    Optionally apply fractional padding around the box.
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box

    if pad > 0:
        pw = int((x2 - x1) * pad)
        ph = int((y2 - y1) * pad)
        x1 = max(0, x1 - pw)
        y1 = max(0, y1 - ph)
        x2 = min(w, x2 + pw)
        y2 = min(h, y2 + ph)

    return img[y1:y2, x1:x2]


def expand_box(box: Tuple[int, int, int, int],
               factor_x: float, factor_y: float,
               img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """Expand a bounding box by a multiplier in each dimension, clamped to image bounds."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = (x2 - x1) * factor_x / 2, (y2 - y1) * factor_y / 2
    return (
        max(0, int(cx - bw)),
        max(0, int(cy - bh)),
        min(img_w, int(cx + bw)),
        min(img_h, int(cy + bh)),
    )


# ─── Preprocessing ───────────────────────────────────────────────────────────

def enhance_for_ocr(img: np.ndarray) -> np.ndarray:
    """
    Pre-process a license plate crop for better OCR accuracy.
    Steps: CLAHE -> sharpen -> binarize (Otsu).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    # 1. CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 2. Sharpen
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    gray = cv2.filter2D(gray, -1, kernel)

    # 3. Otsu binarization
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def normalize_plate_text(text: str) -> str:
    """
    Post-process raw OCR output for license plates:
    - Uppercase
    - Remove spaces and special characters (keep alphanumeric only)
    - Common substitutions: 0↔O in letter positions, 1↔I in letter positions
    """
    text = text.upper().strip()
    # Keep only alphanumeric
    text = "".join(c for c in text if c.isalnum())
    return text


# ─── Geometry helpers ────────────────────────────────────────────────────────

def iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """Compute Intersection-over-Union of two axis-aligned bounding boxes."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union_area = area_a + area_b - inter_area

    return inter_area / (union_area + 1e-6)


def box_overlap_ratio(inner: Tuple[int, int, int, int],
                      outer: Tuple[int, int, int, int]) -> float:
    """
    Fraction of `inner` box's area that overlaps with `outer`.
    Used to associate riders/riders' heads with a vehicle bounding box.
    """
    xi1, yi1, xi2, yi2 = inner
    xo1, yo1, xo2, yo2 = outer

    inter_x1 = max(xi1, xo1)
    inter_y1 = max(yi1, yo1)
    inter_x2 = min(xi2, xo2)
    inter_y2 = min(yi2, yo2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    inner_area = (xi2 - xi1) * (yi2 - yi1) + 1e-6
    return inter_area / inner_area


# ─── Visualization ───────────────────────────────────────────────────────────

_COLORS = {
    "vehicle": (0, 255, 0),       # Green
    "rider": (255, 165, 0),       # Orange
    "helmet": (0, 200, 255),      # Cyan
    "no_helmet": (0, 0, 255),     # Red
    "plate": (255, 255, 0),       # Yellow
    "violation": (0, 0, 200),     # Dark red
}


def draw_detections(img: np.ndarray,
                    vehicles: Optional[List] = None,
                    riders: Optional[List] = None,
                    plates: Optional[List] = None,
                    violations: Optional[List[dict]] = None) -> np.ndarray:
    """
    Draw all detection results on a copy of the image.
    Each list item is a dict with at least keys: 'box', 'label', 'score'.
    violations list items also carry 'num_riders', 'helmet_violations', 'license_plate'.
    Returns the annotated BGR image.
    """
    out = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    lw = max(1, int(min(img.shape[:2]) / 300))

    def _draw(box, label, color):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, lw * 2)
        (tw, th), _ = cv2.getTextSize(label, font, 0.5, lw)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4), font, 0.5, (255, 255, 255), lw)

    for item in (vehicles or []):
        _draw(item["box"], f"vehicle {item.get('score', 0):.2f}", _COLORS["vehicle"])

    for item in (riders or []):
        color = _COLORS["helmet"] if item.get("has_helmet") else _COLORS["no_helmet"]
        label = "helmet" if item.get("has_helmet") else "no_helmet"
        _draw(item["box"], label, color)

    for item in (plates or []):
        _draw(item["box"], item.get("text", "?"), _COLORS["plate"])

    return out
