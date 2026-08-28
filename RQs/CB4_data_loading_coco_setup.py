# ============================================================
# BLOCK 4: Data Loading & COCO Setup
# Loads 20 COCO validation images into memory once.
# Verifies ground truth annotations are accessible.
# ============================================================

import os
import glob
import numpy as np
from PIL import Image
from tqdm.notebook import tqdm

# --- 4.1 Load Image File Paths ---
image_files = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))

assert len(image_files) > 0, \
    f"No images found at {IMAGE_DIR} — check dataset path"

# Take the first NUM_IMAGES for evaluation
image_files = image_files[:EVAL_PARAMS["num_images"]]
print(f"✅ Found {len(image_files)} images for evaluation")

# --- 4.2 Build Image ID Map ---
# COCO image IDs are encoded in the filename e.g. 000000000139.jpg → 139
img_id_map = {
    os.path.basename(f): int(os.path.basename(f).split('.')[0])
    for f in image_files
}
coco_img_ids = list(img_id_map.values())
print(f"✅ Image ID map built: {len(img_id_map)} entries")
print(f"   First 3 IDs: {coco_img_ids[:3]}")

# --- 4.3 Pre-load Images Into Memory ---
# Avoids repeated disk reads during the benchmark loops
print(f"\nPre-loading {len(image_files)} images into memory...")
loaded_images = {}

for img_path in tqdm(image_files, desc="Loading"):
    img_array = np.array(Image.open(img_path).convert("RGB"))
    loaded_images[img_path] = img_array

# Memory estimate
sample_shape = next(iter(loaded_images.values())).shape
total_mb = sum(
    img.nbytes for img in loaded_images.values()
) / 1e6

print(f"✅ All images loaded")
print(f"   Sample shape: {sample_shape}")
print(f"   Total memory: {total_mb:.1f} MB")

# --- 4.4 Verify COCO Annotations ---
# Check that ground truth boxes exist for our image IDs
print(f"\nVerifying COCO annotations...")
missing_annotations = []

for img_id in coco_img_ids:
    ann_ids = coco_gt.getAnnIds(imgIds=img_id)
    if len(ann_ids) == 0:
        missing_annotations.append(img_id)

if missing_annotations:
    print(f"⚠️  {len(missing_annotations)} images have no annotations: "
          f"{missing_annotations}")
else:
    print(f"✅ All {len(coco_img_ids)} images have ground truth annotations")

# --- 4.5 Annotation Statistics ---
# Useful context for interpreting mAP results
total_gt_boxes = 0
category_counts = {}

for img_id in coco_img_ids:
    ann_ids = coco_gt.getAnnIds(imgIds=img_id)
    anns = coco_gt.loadAnns(ann_ids)
    total_gt_boxes += len(anns)
    
    for ann in anns:
        cat_name = coco_gt.loadCats(ann['category_id'])[0]['name']
        category_counts[cat_name] = category_counts.get(cat_name, 0) + 1

# Top 10 most frequent categories in our evaluation set
top_cats = sorted(category_counts.items(), 
                  key=lambda x: x[1], reverse=True)[:10]

print(f"\n--- Ground Truth Statistics ---")
print(f"Total GT boxes across {len(image_files)} images: {total_gt_boxes}")
print(f"Mean GT boxes per image: {total_gt_boxes/len(image_files):.1f}")
print(f"\nTop 10 categories in evaluation set:")
for cat, count in top_cats:
    print(f"   {cat:<20} {count:>4} instances")

# --- 4.6 Save Dataset Manifest ---
import json

manifest = {
    "num_images": len(image_files),
    "image_ids": coco_img_ids,
    "total_gt_boxes": total_gt_boxes,
    "mean_gt_per_image": round(total_gt_boxes / len(image_files), 2),
    "top_categories": dict(top_cats)
}

manifest_path = os.path.join(EVAL_PARAMS["save_dir"], "dataset_manifest.json")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\n✅ Dataset manifest saved: {manifest_path}")

print("\n" + "="*50)
print("BLOCK 4 COMPLETE — Data loaded and verified")
print("="*50)
