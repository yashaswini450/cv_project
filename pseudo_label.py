import os
import shutil
import glob
from pathlib import Path
from ultralytics import YOLO

def pseudo_label():
    src_dir = Path("data/Triple riding.v2-original.yolov11/train")
    dst_dir = Path("data/triple_riding_pseudo/train")
    
    src_img_dir = src_dir / "images"
    dst_img_dir = dst_dir / "images"
    dst_lbl_dir = dst_dir / "labels"
    
    # Create directories
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    model = YOLO("models/helmet_detector.pt")
    
    images = list(src_img_dir.glob("*.jpg"))
    print(f"Found {len(images)} images to process.")
    
    for i, img_path in enumerate(images):
        if i % 50 == 0:
            print(f"Processing {i}/{len(images)}...")
            
        # Copy image
        dst_img_path = dst_img_dir / img_path.name
        if not dst_img_path.exists():
            shutil.copy2(img_path, dst_img_path)
            
        # Run inference
        results = model.predict(source=str(img_path), conf=0.20, verbose=False)
        
        # Write labels
        lbl_name = img_path.stem + ".txt"
        lbl_path = dst_lbl_dir / lbl_name
        
        with open(lbl_path, "w") as f:
            for result in results:
                if result.boxes is None:
                    continue
                for box_data in result.boxes:
                    cls_id = int(box_data.cls[0].item())
                    # Format: class x_center y_center width height
                    # YOLO xywhn provides normalized values [0, 1]
                    x, y, w, h = box_data.xywhn[0].tolist()
                    f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    # Generate data.yaml
    yaml_content = """
train: train/images
val: train/images

nc: 4
names: ['helm', 'kepala', 'motor', 'plat nomor']
"""
    with open("data/triple_riding_pseudo/data.yaml", "w") as f:
        f.write(yaml_content.strip())
        
    print("Pseudo-labeling complete! Data saved to data/triple_riding_pseudo/")

if __name__ == "__main__":
    pseudo_label()
