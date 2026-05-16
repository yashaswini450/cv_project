import shutil
import yaml
from pathlib import Path

# Paths
unified_dir = Path("data/unified_dataset")
kaggle_yaml = unified_dir / "kaggle_data.yaml"

print("Preparing dataset for Kaggle upload...")

# Create a Kaggle-compatible yaml with relative paths
yaml_content = {
    "path": "/kaggle/input/unified-traffic-dataset/unified_dataset",
    "train": "train/images",
    "val": "val/images",
    "test": "test/images",
    "nc": 6,
    "names": ["helm", "kepala", "motor", "plat nomor", "rider", "triple_riding"]
}

with open(kaggle_yaml, "w") as f:
    yaml.dump(yaml_content, f, sort_keys=False)

print(f"✅ Created {kaggle_yaml}")

# Zip the directory
print("Zipping the dataset (this might take a minute)...")
shutil.make_archive("kaggle_upload", 'zip', "data", "unified_dataset")

print("✅ Zip complete! You can now upload 'kaggle_upload.zip' to Kaggle.")
