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
│   ├── detector.py                  # Two-wheeler & rider detection
│   ├── helmet_classifier.py         # Helmet presence classification
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
- Use a second **YOLOv8** model trained on rider/helmet detection
  - Option A: Use a public pretrained model (e.g., from Roboflow helmet detection dataset)
  - Option B: Use YOLOv8 person detection + crop + helmet classifier
- Count persons within/above the two-wheeler bounding box
- Classify each rider's head region for helmet presence

**Key files:** `src/detector.py`, `src/helmet_classifier.py`

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

- Write `test_solution.py` to run `predict()` on sample images
- Test edge cases:
  - No two-wheelers in frame
  - Occluded riders
  - Blurry / dirty license plates
  - Low-light conditions
  - Multiple vehicles in one image

**Key files:** `test_solution.py`

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

## 📊 Evaluation

Final score is computed as:

```
Score = w1 × Violation_Accuracy + w2 × OCR_Accuracy
```

- **Violation Accuracy**: Correct classification of triple riding + helmet counts per vehicle
- **OCR Accuracy**: Correct license plate string extraction

---

## 🚧 Known Challenges & Mitigations

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
| YOLOv8n | Two-wheeler detection | ~6 MB | Ultralytics (COCO pretrained) |
| YOLOv8m helmet | Rider + helmet detection | ~50 MB | Roboflow / custom |
| License plate detector | Plate localization | ~6 MB | Public checkpoint |
| EasyOCR | Text recognition | ~100 MB | EasyOCR pretrained |

**Total: ~162 MB (well under 250 MB limit)**

---

## 🧪 Current Build Status

| Step | Status |
|------|--------|
| Step 1 — Project Setup | ✅ Complete |
| Step 2 — Two-Wheeler Detection | ✅ Complete |
| Step 3 — Rider & Helmet Detection | ✅ Complete |
| Step 4 — License Plate OCR | ✅ Complete |
| Step 5 — Violation Logic | ✅ Complete |
| Step 6 — Integration (solution.py) | ✅ Complete |
| Step 7 — Testing | ✅ Complete |
| Step 8 — Download Script | ✅ Complete |
| Step 9 — Optimization | 🔄 In Progress |
