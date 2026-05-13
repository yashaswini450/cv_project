"""
setup_data.py
=============
One-shot script to download and prepare all datasets.

Datasets:
  1. Roboflow — Two-Wheeler Violation (YOLOv8 format)
     https://universe.roboflow.com/avisentv6/two-wheeler-violation-soc7k
  2. Kaggle   — Traffic Violation Dataset V3
     https://www.kaggle.com/datasets/meliodassourav/traffic-violation-dataset-v3

Usage:
    # Interactive (prompts for API keys):
    python setup_data.py

    # Non-interactive (pass keys via env vars):
    ROBOFLOW_API_KEY=xxx KAGGLE_USERNAME=yyy KAGGLE_KEY=zzz python setup_data.py

    # Skip Kaggle (only get Roboflow):
    python setup_data.py --skip-kaggle

    # Skip Roboflow (only get Kaggle):
    python setup_data.py --skip-roboflow
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


# ─── Config ──────────────────────────────────────────────────────────────────

DATA_DIR = Path("./data")
ROBOFLOW_WORKSPACE = "avisentv6"
ROBOFLOW_PROJECT = "two-wheeler-violation-soc7k"
ROBOFLOW_VERSION = 2
ROBOFLOW_FORMAT = "yolov8"

KAGGLE_DATASET = "meliodassourav/traffic-violation-dataset-v3"
KAGGLE_OUT_DIR = DATA_DIR / "kaggle"


# ─── Utilities ───────────────────────────────────────────────────────────────

def _ask(prompt: str, env_var: str, secret: bool = False) -> str:
    """Get a value from env var or interactive prompt."""
    value = os.environ.get(env_var, "").strip()
    if value:
        print(f"  ✓ {env_var} loaded from environment.")
        return value
    if secret:
        import getpass
        value = getpass.getpass(f"  Enter {prompt}: ").strip()
    else:
        value = input(f"  Enter {prompt}: ").strip()
    return value


def _pip_install(package: str):
    """Install a package silently if not already available."""
    try:
        __import__(package.split("[")[0].replace("-", "_"))
    except ImportError:
        print(f"  Installing {package} …")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", package]
        )


# ─── Roboflow ────────────────────────────────────────────────────────────────

def download_roboflow(api_key: str) -> Path:
    """
    Download the two-wheeler violation dataset from Roboflow in YOLOv8 format.
    Returns the path to the downloaded dataset directory.
    """
    _pip_install("roboflow")
    from roboflow import Roboflow  # type: ignore

    print(f"\n  Connecting to Roboflow workspace: {ROBOFLOW_WORKSPACE}")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    version = project.version(ROBOFLOW_VERSION)

    out_dir = DATA_DIR / "roboflow"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Downloading version {ROBOFLOW_VERSION} as '{ROBOFLOW_FORMAT}' …")
    # Roboflow downloads into the CWD by default; we'll move it after
    dataset = version.download(ROBOFLOW_FORMAT, location=str(out_dir))

    data_yaml = out_dir / "data.yaml"
    print(f"  ✓ Roboflow dataset ready at {out_dir}")
    print(f"    data.yaml: {data_yaml}")

    # Patch data.yaml paths to be absolute so YOLO training works from any CWD
    _patch_data_yaml(data_yaml, out_dir)

    return out_dir


def _patch_data_yaml(yaml_path: Path, dataset_root: Path):
    """Rewrite relative paths in data.yaml to absolute paths."""
    if not yaml_path.exists():
        return
    try:
        import yaml  # type: ignore
    except ImportError:
        _pip_install("pyyaml")
        import yaml  # type: ignore

    with yaml_path.open() as f:
        cfg = yaml.safe_load(f)

    abs_root = dataset_root.resolve()
    for key in ("train", "val", "valid", "test"):
        if key in cfg:
            p = Path(cfg[key])
            if not p.is_absolute():
                cfg[key] = str(abs_root / p)

    with yaml_path.open("w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    print(f"  ✓ Patched paths in {yaml_path.name} to absolute paths.")


# ─── Kaggle ──────────────────────────────────────────────────────────────────

def download_kaggle(username: str, key: str) -> Path:
    """
    Download the traffic violation V3 dataset from Kaggle.
    Returns the path to the extracted dataset directory.
    """
    # Write kaggle.json credentials
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    creds_file = kaggle_dir / "kaggle.json"

    creds = {"username": username, "key": key}
    with creds_file.open("w") as f:
        json.dump(creds, f)
    creds_file.chmod(0o600)
    print("  ✓ Kaggle credentials saved to ~/.kaggle/kaggle.json")

    _pip_install("kaggle")
    import kaggle  # type: ignore  # noqa — triggers credential load

    KAGGLE_OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"  Downloading Kaggle dataset: {KAGGLE_DATASET} …")
    print("  (This is ~1 GB, may take a few minutes)")

    subprocess.check_call([
        sys.executable, "-m", "kaggle", "datasets", "download",
        "-d", KAGGLE_DATASET,
        "-p", str(KAGGLE_OUT_DIR),
        "--unzip",
    ])

    print(f"  ✓ Kaggle dataset extracted to {KAGGLE_OUT_DIR}")
    return KAGGLE_OUT_DIR


# ─── Dataset Summary ─────────────────────────────────────────────────────────

def print_summary(roboflow_dir: Path | None, kaggle_dir: Path | None):
    print("\n" + "=" * 60)
    print("✅  Dataset Setup Complete")
    print("=" * 60)

    if roboflow_dir and roboflow_dir.exists():
        # Count images
        train_imgs = list((roboflow_dir / "train" / "images").glob("*")) if (roboflow_dir / "train" / "images").exists() else []
        val_imgs   = list((roboflow_dir / "valid" / "images").glob("*")) if (roboflow_dir / "valid" / "images").exists() else []
        test_imgs  = list((roboflow_dir / "test"  / "images").glob("*")) if (roboflow_dir / "test"  / "images").exists() else []
        print(f"\n📂 Roboflow Dataset: {roboflow_dir}")
        print(f"   Classes  : motor, helm, kepala, plat nomor, bb")
        print(f"   Train    : {len(train_imgs)} images")
        print(f"   Valid    : {len(val_imgs)} images")
        print(f"   Test     : {len(test_imgs)} images")
        print(f"   data.yaml: {roboflow_dir / 'data.yaml'}")

    if kaggle_dir and kaggle_dir.exists():
        contents = list(kaggle_dir.iterdir())
        print(f"\n📂 Kaggle Dataset: {kaggle_dir}")
        print(f"   Contents : {[c.name for c in contents]}")

    print("\n🚀 Next step: train the model")
    print("   python train.py --data data/roboflow/data.yaml")
    print("=" * 60)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download datasets for traffic violation detection")
    parser.add_argument("--skip-roboflow", action="store_true", help="Skip Roboflow download")
    parser.add_argument("--skip-kaggle",   action="store_true", help="Skip Kaggle download")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("🚦 Traffic Violation Detector — Dataset Setup")
    print("=" * 60)

    roboflow_dir = None
    kaggle_dir = None

    # ── Roboflow ──────────────────────────────────────────────────────────────
    if not args.skip_roboflow:
        print("\n📦 Dataset 1: Roboflow — Two-Wheeler Violation")
        print("   Requires a free Roboflow API key.")
        print("   Get yours at: https://app.roboflow.com → Settings → Roboflow API")
        api_key = _ask("Roboflow API Key", "ROBOFLOW_API_KEY", secret=True)
        if api_key:
            try:
                roboflow_dir = download_roboflow(api_key)
            except Exception as e:
                print(f"  ✗ Roboflow download failed: {e}")
        else:
            print("  ⚠ Skipping Roboflow (no API key provided).")

    # ── Kaggle ────────────────────────────────────────────────────────────────
    if not args.skip_kaggle:
        print("\n📦 Dataset 2: Kaggle — Traffic Violation Dataset V3 (~1 GB)")
        print("   Requires Kaggle API credentials.")
        print("   Get yours at: https://www.kaggle.com → Account → API → Create New Token")
        username = _ask("Kaggle Username", "KAGGLE_USERNAME", secret=False)
        key      = _ask("Kaggle API Key",  "KAGGLE_KEY",     secret=True)
        if username and key:
            try:
                kaggle_dir = download_kaggle(username, key)
            except Exception as e:
                print(f"  ✗ Kaggle download failed: {e}")
        else:
            print("  ⚠ Skipping Kaggle (credentials not provided).")

    print_summary(roboflow_dir, kaggle_dir)


if __name__ == "__main__":
    main()
