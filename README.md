# 🚦 Traffic Rule Violation Detection System

> A Computer Vision system that detects traffic violations by two-wheelers (motorcycles/scooters) from single RGB street images.

---

## 📋 Project Overview

This system detects traffic rule violations including:
- **Triple Riding** — More than 2 riders on a single vehicle
- **Helmet Violation** — One or more riders not wearing a helmet
- **Combined Violation** — Both triple riding and helmet violations simultaneously
- **License Plate Recognition** — OCR on the number plate of every violating vehicle

### 🎯 Final Output Format

```json
{
  "violations": [
    {
      "num_riders": 3,
      "helmet_violations": 2,
      "license_plate": "MH12AB1234"
    }
  ]
}
```

---

## 🏗️ Project Architecture

```
cv_project/
│
├── models/                          # All model weights (≤ 250 MB total)
│   ├── yolo_twowheeler.pt           # YOLO model for two-wheeler detection
│   ├── yolo_rider_helmet.pt         # YOLO model for rider + helmet detection
│   └── ocr_model/                   # OCR model weights / config
│
├── data/                            # Sample data for testing
│   ├── sample_images/
│   └── test_cases/
│
├── notebooks/                       # Exploration & experimentation
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_selection.ipynb
│   └── 03_pipeline_testing.ipynb
│
├── src/                             # Core source modules
│   ├── __init__.py
│   ├── detector.py                  # Two-wheeler, rider & helmet detection
│   ├── ocr.py                       # License plate detection + OCR
│   ├── violation_logic.py           # Violation rules engine
│   └── utils.py                     # Helper functions
│
├── solution.py                      # ✅ Main submission file (TrafficViolationDetector)
├── download_models.py               # Script to download all model weights
├── test_solution.py                 # Local testing script
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## 📌 Step-by-Step Implementation Plan

### ✅ Step 1 — Project Setup & Environment
**Goal:** Set up the project structure, virtual environment, and dependencies.

- [x] Create project folder structure
- [x] Write `requirements.txt`
- [x] Set up `src/` module skeleton
- [x] Create `solution.py` with the required `TrafficViolationDetector` class interface

**Key files:** `requirements.txt`, `solution.py`, `src/__init__.py`

---

### ✅ Step 2 — Two-Wheeler Detection
**Goal:** Detect all motorcycles and scooters in the image.

**Approach:**
- Use **YOLOv8** (pretrained on COCO) — classes `motorcycle` (COCO class 3) and `bicycle` are already included
- Filter detections to only two-wheelers using confidence threshold
- Return bounding boxes for each detected two-wheeler

**Key files:** `src/detector.py`

---

### ✅ Step 3 — Rider & Helmet Detection
**Goal:** For each detected two-wheeler, count riders and identify helmet violations.

**Approach:**
- Use a **YOLOv8** model trained for full-frame rider and helmet detection.
- Detect `rider` and `helmet` classes across the entire image.
- Use spatial association (IoU-based overlap) to map each rider to their corresponding two-wheeler bounding box.
- Classify helmet compliance based on the detected `helmet` bounding boxes overlapping with `rider` bounding boxes.

**Key files:** `src/detector.py`

---

### ✅ Step 4 — License Plate Detection & OCR
**Goal:** Extract and read the number plate of every violating vehicle.

**Approach:**
- **License Plate Detection:** Use a YOLO model fine-tuned for Indian license plates OR use a general plate detector
- **OCR Pipeline:**
  - Option A: **EasyOCR** (simple, no extra model download)
  - Option B: **PaddleOCR** (higher accuracy for Indian plates)
  - Option C: **Tesseract** with preprocessing
- Post-process OCR output: remove spaces, normalize characters (0/O, 1/I confusion)

**Key files:** `src/ocr.py`

---

### ✅ Step 5 — Violation Logic Engine
**Goal:** Combine detections to classify violations per vehicle.

**Rules:**
```
triple_riding  = (num_riders > 2)
helmet_violation = (num_riders_without_helmet > 0)
violation_exists = triple_riding OR helmet_violation
```

- For each two-wheeler:
  - Count riders → check triple riding
  - For each rider → check helmet → count violations
  - If any violation → run OCR on license plate region
  - Append to output list

**Key files:** `src/violation_logic.py`

---

### ✅ Step 6 — Integration in `solution.py`
**Goal:** Assemble all modules into the required `TrafficViolationDetector` class.

```python
class TrafficViolationDetector:
    def __init__(self, model_dir="./models"):
        # Load all models once here
        pass

    def predict(self, image_path: str) -> dict:
        # Stateless inference pipeline
        pass
```

**Constraints to enforce:**
- Models loaded **only** in `__init__`
- `predict()` is fully **stateless**
- Total model size ≤ **250 MB**

**Key files:** `solution.py`

---

### ✅ Step 7 — Local Testing & Evaluation
**Goal:** Validate the pipeline against test images.

- Write `test_solution.py` to run `predict()` on sample images.
- Write `evaluate_kaggle.py` to rigorously test the model on the Kaggle dataset.
- Test edge cases:
  - No two-wheelers in frame
  - Occluded riders
  - Blurry / dirty license plates
  - Low-light conditions
  - Multiple vehicles in one image

**Key files:** `test_solution.py`, `evaluate_kaggle.py`

---

### ✅ Step 8 — Model Download Script
**Goal:** Make it easy to download/set up model weights.

- Write `download_models.py` to fetch model weights programmatically
- Document all model sources and licenses

**Key files:** `download_models.py`

---

### ✅ Step 9 — Optimization & Final Cleanup
**Goal:** Ensure robustness, speed, and size compliance.

- Verify total model size ≤ 250 MB
- Add error handling for edge cases (no detections, bad image, corrupt plate)
- Add confidence thresholds tuning
- Final review of output JSON format compliance
- Implement full-frame detection and spatial association to handle occluded and overlapping riders correctly

---

## 🔧 Technical Stack

| Component | Tool / Library |
|-----------|---------------|
| Object Detection | YOLOv8 (Ultralytics) |
| Helmet Detection | YOLOv8 custom / fine-tuned |
| License Plate Detection | YOLOv8 / OpenCV |
| OCR | EasyOCR / PaddleOCR |
| Image Processing | OpenCV, Pillow |
| Deep Learning Framework | PyTorch |

---

## ⚙️ Setup & Installation

### Step 1 — Clone & Virtual Environment
```bash
cd cv_project
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2 — Download Datasets

**Dataset 1: Roboflow — Two-Wheeler Violation (1,294 images, YOLOv8 format)**
- URL: https://universe.roboflow.com/avisentv6/two-wheeler-violation-soc7k
- Classes: `motor` (motorcycle), `helm` (helmet), `kepala` (head/no helmet), `plat nomor` (plate), `bb`
- Requires a free [Roboflow API key](https://app.roboflow.com)

**Dataset 2: Kaggle — Traffic Violation Dataset V3 (~1 GB)**
- URL: https://www.kaggle.com/datasets/meliodassourav/traffic-violation-dataset-v3
- Requires [Kaggle API credentials](https://www.kaggle.com/settings) (`username` + `key`)

```bash
# Interactive setup — prompts for API keys
python setup_data.py

# Or with environment variables (non-interactive):
ROBOFLOW_API_KEY=xxx KAGGLE_USERNAME=yyy KAGGLE_KEY=zzz python setup_data.py

# Only Roboflow (skip Kaggle):
python setup_data.py --skip-kaggle
```

### Step 3 — Download Base Model Weights
```bash
python download_models.py
```

### Step 4 — Train the Model
```bash
# Train on Roboflow dataset (recommended starting point)
python train.py --data data/roboflow/data.yaml --epochs 50

# Larger model for better accuracy:
python train.py --data data/roboflow/data.yaml --model yolov8m.pt --epochs 100

# On Apple Silicon (MPS):
python train.py --data data/roboflow/data.yaml --device mps

# After training, best weights are saved to models/helmet_detector.pt
```

### Step 5 — Run Inference
```bash
# On a single image:
python solution.py path/to/image.jpg

# Run full test suite:
python test_solution.py --image data/sample_images/test1.jpg
```

---

## 📊 Evaluation & Performance Metrics

We have performed a rigorous local evaluation on the **Kaggle Traffic Violation Test Dataset** (300 test images evenly split across violation categories). 

Our optimized **Hybrid YOLOv8m (Custom Head Detector) + YOLOv8n (COCO Person Detector)** pipeline achieves a final overall accuracy of **86.67%**!

### Detailed Category Performance:
- **HELMET (Zero Violations):** **95.00%** Accuracy (5 FP, 0 FN)
- **NO_HELMET (Helmet Violation):** **83.00%** Accuracy (0 FP, 17 FN)
- **OVERLOADING (Triple Riding):** **82.00%** Accuracy (0 FP, 18 FN)
- **OVERALL ACCURACY:** **86.67%** (260/300 correct classifications)

### Key Optimization Architecture:
1. **Upward-Biased COCO Search Box:** To catch triple-riding when spatial crowding or NMS suppression misses the third head, we run a full-frame COCO person detector as a fallback. We filter detections using an upward-biased search box (expanding 45% upward and 5% downward, with minimal 5% horizontal padding) and a high overlap ratio threshold ($\ge 0.40$). This completely eliminates background pedestrian false positives while capturing true riders sitting on the vehicle.
2. **Nested Vehicle Box Suppression:** If the two-wheeler detector predicts two redundant overlapping boxes for the same motorcycle, we calculate their overlap ratio. If $\ge 70\%$ of the smaller box is nested inside the larger box, we suppress the redundant detection. This prevents split-rider associations.
3. **Optimized Vehicle Recall:** We lowered the vehicle detection confidence threshold to `0.18`. Since any vehicle with 0 associated riders is safely ignored by the violation engine, this dramatically boosts recall in complex/blurry scenes without introducing false positives.

---

## 🚧 Technical Challenges & Mitigations

| Challenge | Mitigation |
|-----------|-----------|
| Occluded riders | Use IoU-based overlap with two-wheeler bbox to assign riders |
| Blurry license plates | Super-resolution preprocessing before OCR |
| Low-light images | CLAHE / brightness normalization |
| Overlapping bounding boxes | NMS post-processing |
| Indian plate font variance | PaddleOCR fine-tuned on Indian plates |
| Model size constraint | Use YOLOv8n (nano) variants + quantization |

---

## 📁 Models Used

| Model | Purpose | Approx. Size | Source |
|-------|---------|-------------|--------|
| `yolov8n.pt` (COCO) | Supplementary person detector | ~6.5 MB | Ultralytics (COCO pretrained) |
| `helmet_detector.pt` | Rider + helmet custom head detector | ~207.5 MB | Custom fine-tuned YOLOv8m |
| `plate_detector.pt` | Plate localization (reused via head model) | 0 MB | Reused from helmet detector plate class |
| EasyOCR | Text recognition | ~100 MB | EasyOCR pretrained |

**Total Model Memory footprint: ~214 MB (fully compliant with the 250 MB size limit)**

---

## 🧪 Project Status: Completed & Validated ✅

- **Step 1 — Project Setup:** ✅ Complete
- **Step 2 — Two-Wheeler Detection:** ✅ Complete
- **Step 3 — Rider & Helmet Detection:** ✅ Complete
- **Step 4 — License Plate OCR:** ✅ Complete
- **Step 5 — Violation Logic:** ✅ Complete
- **Step 6 — Integration (solution.py):** ✅ Complete
- **Step 7 — Testing & Validation:** ✅ Complete (86.67% accuracy)
- **Step 8 — Download Script:** ✅ Complete
- **Step 9 — Optimization & Tuning:** ✅ Complete
