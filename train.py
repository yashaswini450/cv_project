"""
train.py
========
Train a YOLOv8 model on the two-wheeler violation dataset.

The Roboflow dataset has these classes (mapped to our task):
  Class 0 : bb           → bounding box helper (ignored in inference)
  Class 1 : helm         → helmet ✅
  Class 2 : kepala       → head (rider head without helmet) ✅
  Class 3 : motor        → motorcycle/two-wheeler ✅
  Class 4 : plat nomor   → license plate ✅

Training strategy:
  1. Start from YOLOv8n pretrained on COCO (transfer learning)
  2. Fine-tune on Roboflow dataset for all 5 classes
  3. Save best checkpoint to models/helmet_detector.pt

Usage:
    # Basic training (YOLOv8 nano, 50 epochs)
    python train.py --data data/roboflow/data.yaml

    # Larger model, more epochs
    python train.py --data data/roboflow/data.yaml --model yolov8m --epochs 100

    # Resume interrupted training
    python train.py --resume runs/train/exp/weights/last.pt
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


# ─── Defaults ─────────────────────────────────────────────────────────────────

DEFAULTS = {
    "model":      "yolov8s.pt",   # Small model: better accuracy than nano, still <25 MB
    "epochs":     100,
    "imgsz":      736,            # Slightly larger for better small-object detection
    "batch":      16,
    "lr0":        0.01,
    "patience":   20,             # More patience for cosine schedule
    "workers":    4,
    "device":     "cpu",          # override to 'cuda' or 'mps' if available
    "project":    "runs/train",
    "name":       "twowheeler_violation",
    "save_dir":   "models",
}


# ─── Training ─────────────────────────────────────────────────────────────────

def train(args):
    from ultralytics import YOLO

    print("=" * 60)
    print("🚦 Traffic Violation — YOLOv8 Training")
    print("=" * 60)

    # Auto-detect MPS on Apple Silicon, CUDA on NVIDIA
    device = args.device
    if device == "auto":
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"  Device : {device}")
    print(f"  Model  : {args.model}")
    print(f"  Data   : {args.data}")
    print(f"  Epochs : {args.epochs}")
    print(f"  Batch  : {args.batch}")
    print("=" * 60)

    # Load model
    model = YOLO(args.model)

    # Train
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        patience=args.patience,
        workers=args.workers,
        device=device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        # Learning rate schedule
        cos_lr=True,              # Cosine annealing for smoother convergence
        lrf=0.01,                 # Final LR = lr0 * lrf
        # Augmentation
        mosaic=1.0,
        flipud=0.0,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        copy_paste=0.1,           # Copy-paste augmentation for small objects
        mixup=0.1,                # Mixup for better generalization
        close_mosaic=15,          # Disable mosaic in last 15 epochs for fine-tuning
        # Regularization
        weight_decay=0.0005,
        warmup_epochs=3,
        dropout=0.0,
        # Logging
        verbose=True,
        plots=True,
    )

    # Copy best weights to models/
    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    if best_weights.exists():
        dest = save_dir / "helmet_detector.pt"
        shutil.copy(best_weights, dest)
        size_mb = dest.stat().st_size / 1024 / 1024
        print(f"\n✅ Best weights saved to {dest} ({size_mb:.1f} MB)")
    else:
        print(f"\n⚠ Could not find best weights at {best_weights}")

    return results


def resume(weights_path: str):
    """Resume interrupted training from a checkpoint."""
    from ultralytics import YOLO
    print(f"Resuming training from {weights_path} …")
    model = YOLO(weights_path)
    model.train(resume=True)


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate(args):
    """Run validation on the test split and print metrics."""
    from ultralytics import YOLO

    model_path = args.weights or str(Path(args.save_dir) / "helmet_detector.pt")
    print(f"\n📊 Evaluating {model_path} on test set …")
    model = YOLO(model_path)
    metrics = model.val(data=args.data, split="test", verbose=True)

    print("\n── Per-class Results ──────────────────────────────────")
    if hasattr(metrics, "box"):
        print(f"  mAP50     : {metrics.box.map50:.4f}")
        print(f"  mAP50-95  : {metrics.box.map:.4f}")
        print(f"  Precision : {metrics.box.mp:.4f}")
        print(f"  Recall    : {metrics.box.mr:.4f}")
    print("───────────────────────────────────────────────────────")
    return metrics


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train / Evaluate YOLOv8 for traffic violations")
    subparsers = parser.add_subparsers(dest="command")

    # ── train subcommand ──
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--data",    required=True, help="Path to data.yaml")
    train_parser.add_argument("--model",   default=DEFAULTS["model"])
    train_parser.add_argument("--epochs",  type=int, default=DEFAULTS["epochs"])
    train_parser.add_argument("--imgsz",   type=int, default=DEFAULTS["imgsz"])
    train_parser.add_argument("--batch",   type=int, default=DEFAULTS["batch"])
    train_parser.add_argument("--lr0",     type=float, default=DEFAULTS["lr0"])
    train_parser.add_argument("--patience",type=int, default=DEFAULTS["patience"])
    train_parser.add_argument("--workers", type=int, default=DEFAULTS["workers"])
    train_parser.add_argument("--device",  default="auto")
    train_parser.add_argument("--project", default=DEFAULTS["project"])
    train_parser.add_argument("--name",    default=DEFAULTS["name"])
    train_parser.add_argument("--save-dir",default=DEFAULTS["save_dir"])

    # ── resume subcommand ──
    resume_parser = subparsers.add_parser("resume", help="Resume training from checkpoint")
    resume_parser.add_argument("weights", help="Path to last.pt checkpoint")

    # ── eval subcommand ──
    eval_parser = subparsers.add_parser("eval", help="Evaluate the trained model")
    eval_parser.add_argument("--data",    required=True, help="Path to data.yaml")
    eval_parser.add_argument("--weights", default=None,  help="Path to model weights")
    eval_parser.add_argument("--save-dir",default=DEFAULTS["save_dir"])

    # ── Fallback: allow `python train.py --data ...` directly ────────────────
    parser.add_argument("--data",    default=None)
    parser.add_argument("--model",   default=DEFAULTS["model"])
    parser.add_argument("--epochs",  type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--imgsz",   type=int, default=DEFAULTS["imgsz"])
    parser.add_argument("--batch",   type=int, default=DEFAULTS["batch"])
    parser.add_argument("--lr0",     type=float, default=DEFAULTS["lr0"])
    parser.add_argument("--patience",type=int, default=DEFAULTS["patience"])
    parser.add_argument("--workers", type=int, default=DEFAULTS["workers"])
    parser.add_argument("--device",  default="auto")
    parser.add_argument("--project", default=DEFAULTS["project"])
    parser.add_argument("--name",    default=DEFAULTS["name"])
    parser.add_argument("--save-dir",default=DEFAULTS["save_dir"])
    parser.add_argument("--resume",  default=None, help="Resume from checkpoint")
    parser.add_argument("--eval",    action="store_true", help="Evaluate after training")
    parser.add_argument("--weights", default=None)

    args = parser.parse_args()

    if args.command == "resume" or args.resume:
        w = args.weights if args.command == "resume" else args.resume
        resume(w)
    elif args.command == "eval":
        evaluate(args)
    elif args.data:
        results = train(args)
        if args.eval:
            evaluate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
