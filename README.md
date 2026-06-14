```markdown
# Traffic Rule Violation Detection System 🚦

A lightweight computer vision pipeline that detects two-wheeler traffic violations from single RGB street images. The system flags instances of triple riding and missing helmets, and automatically runs OCR to extract the license plate of any violating vehicle.

## 📋 What it Detects
- **Triple Riding:** More than 2 riders on a single motorcycle/scooter.
- **Helmet Violations:** One or more riders on the vehicle aren't wearing a helmet.
- **Combined Violations:** Both of the above occurring simultaneously.
- **License Plate Extraction:** Performs OCR on the number plate of *only* the violating vehicles.

### Output Format
The system outputs a clean JSON response for downstream processing:
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

## 🧠 How It Works (The Pipeline)

Instead of relying on a single massive model, this project uses a modular, multi-stage pipeline to keep memory usage low (well under the 250 MB limit) while maximizing accuracy.

1. **Two-Wheeler Detection:** A base YOLOv8 model (pretrained on COCO) scans the image specifically for motorcycles and bicycles.
2. **Rider & Helmet Association:** A custom-trained YOLOv8 head detector finds riders and helmets across the whole frame. We use spatial association (IoU-based overlap) to map these riders to their respective vehicles.
3. **Violation Logic Engine:** For each bike, the system counts the assigned riders and checks for helmet compliance. If a rule is broken, the vehicle is flagged.
4. **License Plate OCR:** If a vehicle is flagged, the system isolates the license plate bounding box and passes it to EasyOCR to extract the text, applying basic post-processing to fix common character confusions (like `0` vs `O`).

---

## 🏗️ Project Structure

```text
cv_project/
├── models/                  # Model weights (kept under 250 MB total)
├── data/                    # Sample test images
├── notebooks/               # R&D and experimentation notebooks
├── src/                     # Core pipeline modules
│   ├── detector.py          # YOLO inference (bikes, riders, helmets)
│   ├── ocr.py               # License plate cropping and EasyOCR
│   ├── violation_logic.py   # Rules engine for flagging
│   └── utils.py             # Bounding box & IoU helpers
├── solution.py              # ✅ Main entry point (TrafficViolationDetector)
├── download_models.py       # Weight fetching script
├── test_solution.py         # Local evaluation script
└── requirements.txt         

```

---

## ⚙️ Quick Start

### 1. Install Dependencies

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

```

### 2. Download Data & Models

We use datasets from Roboflow and Kaggle for training and evaluation. You can set this up automatically:

```bash
# Prompts for API keys if you want to download the datasets
python setup_data.py --skip-kaggle  # Use --skip-kaggle if you only want Roboflow

# Fetch the required YOLO and EasyOCR weights
python download_models.py

```

### 3. Run Inference

```bash
# Run on a single image
python solution.py path/to/image.jpg

# Or run the local test suite
python test_solution.py --image data/sample_images/test1.jpg

```

---

## 📊 Performance & Benchmarks

We evaluated the pipeline locally against 300 test images from the Kaggle Traffic Violation Dataset (evenly split across violation categories).

**Overall System Accuracy: 85.0%** (255/300 correct classifications)

**Breakdown:**

* **Helmet Compliance (Normal):** 88.0% Accuracy
* **Helmet Violation:** 78.0% Accuracy
* **Triple Riding:** 89.0% Accuracy

### Under the Hood: Key Optimizations

Getting the system to work reliably in crowded street scenes required a few specific tweaks:

* **Upward-Biased Search Boxes:** To catch triple-riding when spatial crowding causes the main detector to miss the third head, we use a COCO person detector as a fallback. We filter detections using an upward-biased search box (expanding 45% upward) and a high overlap ratio (>= 0.40). This captures true riders sitting on the bike while ignoring background pedestrians.
* **Nested Box Suppression:** If the detector predicts two redundant, overlapping boxes for the same bike (>= 70% overlap), the smaller one is suppressed to prevent "split-rider" assignment bugs.
* **Optimized Recall:** We dropped the vehicle detection confidence threshold to `0.18`. Since the logic engine naturally ignores vehicles with zero associated riders, this safely boosts recall in blurry scenes without triggering false positives.

---

## 📦 Model Footprint

To ensure this project remains lightweight and deployable, the total model footprint is strictly managed.

| Model | Purpose | Size |
| --- | --- | --- |
| `yolov8s.pt` | Supplementary person detector | ~22.6 MB |
| `helmet_detector.pt` | Custom rider/helmet head detector | ~22.5 MB |
| EasyOCR | Text recognition (CRAFT + English) | ~98.3 MB |

**Total Memory Footprint: ~143.4 MB** (Fully compliant with the < 250 MB constraint).

---

## 🚧 Edge Cases Handled

* **Occlusions:** Bounding box IoU logic ensures riders slightly hidden behind others are still mapped to the correct bike.
* **Low Light/Blur:** CLAHE and basic image normalization are applied before inference to help with noisy nighttime images.
* **Overlapping Vehicles:** NMS (Non-Maximum Suppression) is heavily utilized to prevent a single rider from being assigned to two bikes parked next to each other.

```

```
