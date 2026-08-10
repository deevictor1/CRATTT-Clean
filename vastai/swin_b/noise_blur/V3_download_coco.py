# ============================================================
# BLOCK V3: Download COCO val2017
# Run on: Vast.ai
# Runs once per instance — skips if already present
# Takes approximately 10-15 minutes on first run
# ============================================================

import os, subprocess

COCO_BASE = "/workspace/coco"
IMAGE_DIR = f"{COCO_BASE}/val2017"
ANN_PATH  = f"{COCO_BASE}/annotations/instances_val2017.json"

os.makedirs(COCO_BASE, exist_ok=True)

if os.path.exists(IMAGE_DIR):
    n = len([f for f in os.listdir(IMAGE_DIR)
             if f.endswith('.jpg')])
    if n > 4900:
        print(f"✅ COCO val2017 already present: {n} images")
        print(f"✅ Annotations: {ANN_PATH}")
    else:
        print(f"⚠️  Only {n} images found — re-downloading")
else:
    print("COCO val2017 not found — downloading now...")
    print()

    print("Step 1/4: Downloading val2017 images (~1GB)...")
    subprocess.run([
        "wget", "-q", "--show-progress",
        "http://images.cocodataset.org/zips/val2017.zip",
        "-O", f"{COCO_BASE}/val2017.zip"
    ], check=True)

    print("Step 2/4: Downloading annotations (~241MB)...")
    subprocess.run([
        "wget", "-q", "--show-progress",
        "http://images.cocodataset.org/annotations/"
        "annotations_trainval2017.zip",
        "-O", f"{COCO_BASE}/annotations.zip"
    ], check=True)

    print("Step 3/4: Extracting images...")
    subprocess.run([
        "unzip", "-q", f"{COCO_BASE}/val2017.zip",
        "-d", COCO_BASE
    ], check=True)

    print("Step 4/4: Extracting annotations...")
    subprocess.run([
        "unzip", "-q", f"{COCO_BASE}/annotations.zip",
        "-d", COCO_BASE
    ], check=True)

    os.remove(f"{COCO_BASE}/val2017.zip")
    os.remove(f"{COCO_BASE}/annotations.zip")

    n = len([f for f in os.listdir(IMAGE_DIR)
             if f.endswith('.jpg')])
    print(f"\n✅ Downloaded: {n} images")
    print(f"✅ Images      : {IMAGE_DIR}")
    print(f"✅ Annotations : {ANN_PATH}")
