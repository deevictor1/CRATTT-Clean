# ============================================================
# BLOCK V7: Load Images
# Run on: Vast.ai
# Loads all N_IMAGES COCO val2017 images into RAM
# 5000 images ~ 4-6GB RAM — safe on 32GB+ instances
# ============================================================

import os
import numpy as np
from PIL import Image
from tqdm import tqdm

print(f"Loading {N_IMAGES} COCO val2017 images...")

all_files = sorted([
    os.path.join(IMAGE_DIR, f)
    for f in os.listdir(IMAGE_DIR)
    if f.endswith('.jpg')
])

print(f"Total available : {len(all_files)} images")

image_files = all_files[:N_IMAGES]
print(f"Loading         : {len(image_files)} images")

img_id_map = {}
for path in image_files:
    fname  = os.path.basename(path)
    img_id = int(fname.replace('.jpg', ''))
    img_id_map[fname] = img_id

loaded_images = {}
failed        = []

for path in tqdm(image_files, desc="Loading images"):
    try:
        img = np.array(Image.open(path).convert("RGB"))
        loaded_images[path] = img
    except Exception as e:
        failed.append(path)

print(f"\n✅ Loaded  : {len(loaded_images)} images")
if failed:
    print(f"⚠️  Failed  : {len(failed)} images")

sample    = next(iter(loaded_images.values()))
total_gb  = (sample.nbytes * len(loaded_images)) / 1e9
print(f"   RAM used: ~{total_gb:.1f} GB")
print(f"\n✅ Image ID map: {len(img_id_map)} entries")
