# ============================================================
# BLOCK 6: Full ImageNet-C Corruption Sweep
# 15 corruptions × 5 severities × 2 models × 20 images
# Estimated runtime: 90-120 minutes on T4
# Checkpoints saved after every corruption — resumable
# ============================================================

import torch
import numpy as np
import json
import os
import cv2
from PIL import Image
from imagecorruptions import corrupt
from tqdm.notebook import tqdm

# --- 6.1 Resume Logic ---
# If a checkpoint exists for a corruption, skip it and reload from disk
def load_checkpoint(corruption_name, ckpt_dir):
    path = os.path.join(ckpt_dir, f"{corruption_name}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def save_checkpoint(corruption_name, data, ckpt_dir):
    path = os.path.join(ckpt_dir, f"{corruption_name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

# --- 6.2 Main Sweep Loop ---
all_sweep_results = {}
total_corruptions = len(ALL_CORRUPTIONS)

print(f"Starting full corruption sweep")
print(f"Corruptions : {total_corruptions}")
print(f"Severities  : {SEVERITIES}")
print(f"Images      : {len(image_files)}")
print(f"Checkpoint  : {EVAL_PARAMS['ckpt_dir']}")
print("="*50)

for cat_name, corruptions in CORRUPTION_CATEGORIES.items():
    print(f"\n📂 Category: {cat_name}")

    for corruption in corruptions:

        # Check for existing checkpoint
        ckpt = load_checkpoint(corruption, EVAL_PARAMS["ckpt_dir"])
        if ckpt is not None:
            print(f"  ↩️  {corruption}: loaded from checkpoint")
            all_sweep_results[corruption] = ckpt
            continue

        print(f"  ▶  {corruption}")
        corruption_results = {
            "category":   cat_name,
            "corruption": corruption,
            "dino_mAP":   [],
            "yolo_mAP":   [],
            "severities": SEVERITIES
        }

        pbar = tqdm(SEVERITIES, desc=f"    {corruption}",
                    leave=False)

        for sev in pbar:
            dino_preds = []
            yolo_preds = []

            for img_path in image_files:
                img_id  = img_id_map[os.path.basename(img_path)]
                raw_img = loaded_images[img_path]

                # Apply ImageNet-C corruption
                c_img = corrupt(
                    raw_img,
                    corruption_name=corruption,
                    severity=sev
                )

                # GroundingDINO inference — patched extraction
                boxes, scores, labels = run_dino_with_categories(c_img)
                dino_preds.extend(dino_to_coco_format_patched(boxes, scores, labels, img_id))

                # YOLO-World inference — BGR conversion for correct colour channel order
                c_img_bgr = cv2.cvtColor(c_img, cv2.COLOR_RGB2BGR)
                y_res = yolo_model.predict(
                    c_img_bgr,
                    conf=CRATTT_PARAMS["yolo_conf"],
                    verbose=False
                )[0]

                yolo_preds.extend(yolo_to_coco_format(y_res, img_id))

            # Compute mAP for this severity
            dm, ds = compute_map(dino_preds, coco_gt, coco_img_ids)
            ym, ys = compute_map(yolo_preds, coco_gt, coco_img_ids)

            corruption_results["dino_mAP"].append(float(dm))
            corruption_results["yolo_mAP"].append(float(ym))

            pbar.set_postfix({
                "sev":  sev,
                "DINO": f"{dm:.3f}",
                "YOLO": f"{ym:.3f}"
            })

            # Warn if evaluation failed
            if ds != "ok":
                print(f"    ⚠️  DINO eval failed sev{sev}: {ds}")
            if ys != "ok":
                print(f"    ⚠️  YOLO eval failed sev{sev}: {ys}")

        # Save checkpoint immediately after each corruption
        save_checkpoint(
            corruption,
            corruption_results,
            EVAL_PARAMS["ckpt_dir"]
        )
        all_sweep_results[corruption] = corruption_results

        # Print corruption summary
        print(f"     DINO mAP: "
              f"{[f'{x:.3f}' for x in corruption_results['dino_mAP']]}")
        print(f"     YOLO mAP: "
              f"{[f'{x:.3f}' for x in corruption_results['yolo_mAP']]}")

# --- 6.3 Save Complete Results ---
full_results_path = os.path.join(
    EVAL_PARAMS["save_dir"], "full_sweep_results.json"
)
with open(full_results_path, "w") as f:
    json.dump(all_sweep_results, f, indent=2)

print(f"\n✅ Full sweep complete")
print(f"   Results saved: {full_results_path}")
print(f"   Corruptions completed: {len(all_sweep_results)}/15")
print("\n" + "="*50)
print("BLOCK 6 COMPLETE — Corruption sweep finished")
print("="*50)
