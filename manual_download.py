import requests
import zipfile
import os
from pathlib import Path

# Credentials from the user's previous command
API_KEY = "WEyp5eVsCPxteeipLrE0"
WORKSPACE = "avisentv6"
PROJECT = "two-wheeler-violation-soc7k"
VERSION = 2

def download_dataset():
    # 1. Get the download URL from Roboflow API
    # The 'yolov8' format is what we need
    url = f"https://api.roboflow.com/avisentv6/two-wheeler-violation-soc7k/2/yolov8?api_key={API_KEY}"
    
    print(f"Fetching download URL from Roboflow...")
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error fetching URL: {response.status_code}")
        print(response.text)
        return

    data = response.json()
    if 'export' not in data or 'link' not in data['export']:
        print("Export not ready or link missing. Data returned:")
        print(data)
        return

    download_link = data['export']['link']
    print(f"Downloading ZIP from: {download_link}")

    # 2. Download the actual ZIP file
    zip_path = Path("roboflow_manual.zip")
    with requests.get(download_link, stream=True) as r:
        r.raise_for_status()
        with open(zip_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    
    print(f"Download complete: {zip_path.stat().st_size} bytes")

    # 3. Extract to data/roboflow
    extract_path = Path("data/roboflow")
    extract_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting to {extract_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    print("Extraction complete!")
    
    # Cleanup
    zip_path.unlink()
    print("Cleanup done.")

if __name__ == "__main__":
    download_dataset()
