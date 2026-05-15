from roboflow import Roboflow
import os

rf = Roboflow(api_key="8bL7FieD0q")
project = rf.workspace("avisentv6").project("two-wheeler-violation-soc7k")
version = project.version(2)

print("Starting download...")
dataset = version.download("yolov8", location="./data/roboflow")
print(f"Download finished. Files in data/roboflow: {os.listdir('./data/roboflow') if os.path.exists('./data/roboflow') else 'Directory not found'}")
