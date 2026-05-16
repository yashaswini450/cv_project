import os
import shutil
import glob
import json
import subprocess
import sys
from pathlib import Path

# Try to import yaml
try:
    import yaml
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pyyaml"])
    import yaml


DATA_DIR = Path("./data")
UNIFIED_DIR = DATA_DIR / "unified_dataset"

# Canonical unified classes
CANONICAL_CLASSES = ["helm", "kepala", "motor", "plat nomor", "rider"]
CANONICAL_MAP = {c: i for i, c in enumerate(CANONICAL_CLASSES)}

# Mappings from potential dataset class names to canonical names
CLASS_ALIASES = {
    "helmet": "helm",
    "with_helmet": "helm",
    "with helmet": "helm",
    "helm": "helm",
    
    "no_helmet": "kepala",
    "without_helmet": "kepala",
    "without helmet": "kepala",
    "no helmet": "kepala",
    "kepala": "kepala",
    
    "motorcycle": "motor",
    "motor": "motor",
    "bike": "motor",
    "scooter": "motor",
    "two wheeler": "motor",
    "twowheeler": "motor",
    
    "license_plate": "plat nomor",
    "plate": "plat nomor",
    "numberplate": "plat nomor",
    "plat nomor": "plat nomor",
    
    "rider": "rider",
    "person": "rider",
    "motorcyclist": "rider",
    "bb": "rider"
}

def ask(prompt: str, env_var: str, secret: bool = False) -> str:
    value = os.environ.get(env_var, "").strip()
    if value:
        return value
    if secret:
        import getpass
        value = getpass.getpass(f"Enter {prompt}: ").strip()
    else:
        value = input(f"Enter {prompt}: ").strip()
    return value


def download_roboflow_dataset(api_key: str, workspace: str, project_name: str, dest_name: str) -> Path:
    out_dir = DATA_DIR / dest_name
    if out_dir.exists():
        print(f"  [Roboflow] Dataset {project_name} already exists at {out_dir}. Skipping download.")
        return out_dir

    try:
        from roboflow import Roboflow
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "roboflow"])
        from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    
    # Try versions sequentially
    version_to_dl = None
    for v in [1, 2, 3, 4, 5]:
        try:
            version_to_dl = project.version(v)
            break
        except Exception:
            continue
            
    if not version_to_dl:
        print(f"  [Roboflow] Could not find a valid version for {project_name}")
        return None

    print(f"  [Roboflow] Downloading {project_name} version {version_to_dl.version}...")
    dataset = version_to_dl.download("yolov8", location=str(out_dir))
    return out_dir


def download_kaggle_dataset(username: str, key: str, dataset_path: str, dest_name: str) -> Path:
    out_dir = DATA_DIR / dest_name
    if out_dir.exists():
        print(f"  [Kaggle] Dataset {dataset_path} already exists at {out_dir}. Skipping download.")
        return out_dir

    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    creds_file = kaggle_dir / "kaggle.json"
    with creds_file.open("w") as f:
        json.dump({"username": username, "key": key}, f)
    creds_file.chmod(0o600)

    try:
        import kaggle
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kaggle"])
        import kaggle

    print(f"  [Kaggle] Downloading {dataset_path}...")
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([
        sys.executable, "-m", "kaggle", "datasets", "download",
        "-d", dataset_path,
        "-p", str(out_dir),
        "--unzip",
    ])
    return out_dir


def process_and_merge_dataset(src_dir: Path, split: str):
    """
    Reads images and labels from src_dir/split, maps the labels,
    and copies them to UNIFIED_DIR/split.
    """
    if not src_dir.exists():
        return

    # Parse data.yaml to get class names
    yaml_path = src_dir / "data.yaml"
    if not yaml_path.exists():
        # Kaggle datasets might not have data.yaml at root. 
        # Fallback to searching or hardcoding if necessary.
        print(f"  Warning: No data.yaml found in {src_dir}. Looking for classes.txt...")
        # For Kaggle, sometimes there is a classes.txt or it's buried in subdirs.
        # We will attempt a generic search.
        yaml_files = list(src_dir.rglob("data.yaml"))
        if yaml_files:
            yaml_path = yaml_files[0]
        else:
            print(f"  Could not find classes for {src_dir}. Skipping.")
            return

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    if isinstance(data.get("names"), dict):
        orig_classes = {int(k): v for k, v in data["names"].items()}
    elif isinstance(data.get("names"), list):
        orig_classes = {i: v for i, v in enumerate(data["names"])}
    else:
        print(f"  Could not parse class names in {yaml_path}. Skipping.")
        return

    # Build local ID -> Canonical ID map
    local_to_canonical = {}
    for local_id, local_name in orig_classes.items():
        canonical_name = CLASS_ALIASES.get(local_name.lower())
        if canonical_name:
            local_to_canonical[local_id] = CANONICAL_MAP[canonical_name]

    print(f"  [{src_dir.name}] Class mapping: {local_to_canonical}")

    # Process split
    # Some datasets use "valid" instead of "val"
    src_split = split
    if split == "val" and not (src_dir / "val").exists() and (src_dir / "valid").exists():
        src_split = "valid"

    images_dir = src_dir / src_split / "images"
    labels_dir = src_dir / src_split / "labels"

    # Handle nested directories (Kaggle sometimes puts them inside another folder)
    if not images_dir.exists():
        found = list(src_dir.rglob(f"{src_split}/images"))
        if found:
            images_dir = found[0]
            labels_dir = found[0].parent / "labels"
        else:
            return

    dest_images = UNIFIED_DIR / split / "images"
    dest_labels = UNIFIED_DIR / split / "labels"
    dest_images.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)

    img_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpeg"))
    
    copied = 0
    for img_path in img_files:
        lbl_path = labels_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        # Read and map labels
        new_lines = []
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                local_cls = int(parts[0])
                if local_cls in local_to_canonical:
                    canon_cls = local_to_canonical[local_cls]
                    new_lines.append(f"{canon_cls} " + " ".join(parts[1:]))

        if new_lines: # Only copy if there's at least one relevant bounding box
            # To avoid filename collisions across datasets, prefix the filename
            prefix = src_dir.name + "_"
            new_img_name = prefix + img_path.name
            new_lbl_name = prefix + lbl_path.name

            shutil.copy2(img_path, dest_images / new_img_name)
            with open(dest_labels / new_lbl_name, 'w') as f:
                f.write("\n".join(new_lines))
            copied += 1

    print(f"  [{src_dir.name} - {split}] Copied {copied} images with mapped labels.")


def main():
    print("🚦 Unified Dataset Merger")
    
    # 1. Get Credentials
    rf_api_key = ask("Roboflow API Key", "ROBOFLOW_API_KEY", secret=True)
    k_user = ask("Kaggle Username", "KAGGLE_USERNAME")
    k_key = ask("Kaggle API Key", "KAGGLE_KEY", secret=True)

    # 2. Download Datasets
    print("\n--- Downloading Datasets ---")
    
    # Existing Roboflow dataset
    ds_base = DATA_DIR / "roboflow"
    
    # Dataset 3: Triple Riding
    ds3 = None
    if rf_api_key:
        ds3 = download_roboflow_dataset(rf_api_key, "project-ksfzr", "triple-riding", "roboflow_triple_riding")
    
    # Dataset 4: License Plate
    ds4 = None
    if rf_api_key:
        # Download to a temporary/staging location to check size first
        temp_ds4 = download_roboflow_dataset(rf_api_key, "objectdetection-slsst", "license-plate-detection-naxbh", "roboflow_license_plate")
        if temp_ds4:
            # Count images
            num_imgs = len(list(temp_ds4.rglob("*.jpg"))) + len(list(temp_ds4.rglob("*.png")))
            if num_imgs <= 2500:
                print(f"  [Dataset 4] Contains {num_imgs} images. Including it.")
                ds4 = temp_ds4
            else:
                print(f"  [Dataset 4] Contains {num_imgs} images (exceeds 2500 limit). Removing to save training time.")
                shutil.rmtree(temp_ds4)
    
    # Dataset 5: Kaggle Indian Traffic
    ds5 = None
    if k_user and k_key:
        ds5 = download_kaggle_dataset(k_user, k_key, "sakshamjn/vehicle-detection-8-classes-object-detection", "kaggle_indian_traffic")

    # 3. Process and Merge
    print("\n--- Merging Datasets ---")
    if UNIFIED_DIR.exists():
        shutil.rmtree(UNIFIED_DIR)
    UNIFIED_DIR.mkdir(parents=True)

    datasets_to_merge = [ds for ds in [ds_base, ds3, ds4, ds5] if ds is not None and ds.exists()]
    
    for split in ["train", "val", "test"]:
        print(f"\nProcessing split: {split}")
        for ds in datasets_to_merge:
            process_and_merge_dataset(ds, split)

    # 4. Write Unified data.yaml
    yaml_content = {
        "train": str((UNIFIED_DIR / "train" / "images").resolve()),
        "val": str((UNIFIED_DIR / "val" / "images").resolve()),
        "test": str((UNIFIED_DIR / "test" / "images").resolve()),
        "nc": len(CANONICAL_CLASSES),
        "names": CANONICAL_CLASSES
    }
    
    with open(UNIFIED_DIR / "unified_data.yaml", "w") as f:
        yaml.dump(yaml_content, f, sort_keys=False)
        
    print(f"\n✅ Merging complete. Unified data.yaml saved at {UNIFIED_DIR / 'unified_data.yaml'}")
    print("Update your train.py to point to data/unified_dataset/unified_data.yaml")

if __name__ == "__main__":
    main()
