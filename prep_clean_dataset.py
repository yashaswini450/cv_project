import os
import shutil
import yaml
import random
from pathlib import Path

def create_clean_split():
    print("🚦 Creating strictly split clean dataset (80/10/10)...")
    
    src_dir = Path("data/roboflow")
    dst_dir = Path("data/clean_dataset_80_10_10")
    
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    
    # Create structure
    for split in ["train", "val", "test"]:
        (dst_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (dst_dir / split / "labels").mkdir(parents=True, exist_ok=True)
        
    # Gather all images from all splits in the clean roboflow dataset
    all_images = []
    for split in ["train", "valid", "test"]:
        img_dir = src_dir / split / "images"
        if img_dir.exists():
            all_images.extend(list(img_dir.glob("*.jpg")))
            
    # Shuffle with a fixed seed
    random.seed(42)
    random.shuffle(all_images)
    
    total = len(all_images)
    train_end = int(total * 0.80)
    val_end = train_end + int(total * 0.10)
    
    train_imgs = all_images[:train_end]
    val_imgs = all_images[train_end:val_end]
    test_imgs = all_images[val_end:]
    
    def copy_files(img_list, split_name):
        copied = 0
        for img_path in img_list:
            lbl_path = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
            if lbl_path.exists():
                shutil.copy2(img_path, dst_dir / split_name / "images" / img_path.name)
                shutil.copy2(lbl_path, dst_dir / split_name / "labels" / lbl_path.name)
                copied += 1
        return copied

    t_count = copy_files(train_imgs, "train")
    v_count = copy_files(val_imgs, "val")
    te_count = copy_files(test_imgs, "test")
    
    print(f"✅ Split Complete:")
    print(f"   Train: {t_count} images")
    print(f"   Val:   {v_count} images")
    print(f"   Test:  {te_count} images")
    
    # Generate standard local YAML
    yaml_content = {
        "train": str((dst_dir / "train" / "images").resolve()),
        "val": str((dst_dir / "val" / "images").resolve()),
        "test": str((dst_dir / "test" / "images").resolve()),
        "nc": 5,
        "names": ["bb", "helm", "kepala", "motor", "plat nomor"]
    }
    with open(dst_dir / "clean_data.yaml", "w") as f:
        yaml.dump(yaml_content, f, sort_keys=False)

    # Generate Kaggle YAML
    kaggle_yaml_content = {
        "path": "/kaggle/input/clean-traffic-dataset/clean_dataset_80_10_10",
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 5,
        "names": ["bb", "helm", "kepala", "motor", "plat nomor"]
    }
    with open(dst_dir / "kaggle_clean_data.yaml", "w") as f:
        yaml.dump(kaggle_yaml_content, f, sort_keys=False)
        
    print(f"✅ Created clean_data.yaml and kaggle_clean_data.yaml")

    # Zip for Kaggle
    print("📦 Zipping the clean dataset for Kaggle upload (this might take a minute)...")
    shutil.make_archive("clean_kaggle_upload", 'zip', "data", "clean_dataset_80_10_10")
    print("✅ Zip complete! You can now upload 'clean_kaggle_upload.zip' to Kaggle.")

if __name__ == "__main__":
    create_clean_split()
