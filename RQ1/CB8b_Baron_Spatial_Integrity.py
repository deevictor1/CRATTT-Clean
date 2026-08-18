# ============================================================
# BLOCK 8B: BARON Spatial Integrity Under High-Severity Corruption
# Evaluates BARON's region-extraction spatial integrity directly
# (Objective 2), using the patched extraction (run_dino_with_categories),
# matching the corrected methodology used throughout Phase 4 onward,
# not Block 8's original unpatched post_process_grounded_object_detection.
# ============================================================

import torch
import numpy as np
import json
import os
from imagecorruptions import corrupt

N_BARON_TEST = 20          # pilot scale, consistent with this dissertation's
                            # established pilot-before-full-scale discipline
HIGH_SEVERITIES = [4, 5]   # "high-severity", matching the objective's own wording
BLUR_TYPES    = ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"]
WEATHER_TYPES = ["snow", "frost", "fog", "brightness"]

BARON_CKPT_PATH = os.path.join(EVAL_PARAMS["save_dir"], "baron_spatial_integrity.json")


def iou_xyxy(box_a, box_b):
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def get_baron_regions(image_np):
    """Runs the patched DINO extraction, then BARON, on a single image.
    Returns list of xyxy boxes (already in original image coordinates)."""
    boxes, scores, labels = run_dino_with_categories(image_np)
    if len(boxes) == 0:
        return []
    crops, valid_boxes, kept_idx = baron.extract(
        image_np, boxes, scores, threshold=CRATTT_PARAMS["dino_text_thr"]
    )
    return valid_boxes.tolist() if len(valid_boxes) > 0 else []


def match_and_score(clean_boxes, corrupt_boxes):
    """Greedy best-IoU matching, clean region -> best corrupted region.
    Returns (mean_iou_of_matched, survival_rate)."""
    if len(clean_boxes) == 0:
        return None, None
    matched_ious = []
    survived = 0
    for cb in clean_boxes:
        if len(corrupt_boxes) == 0:
            continue
        best_iou = max(iou_xyxy(cb, xb) for xb in corrupt_boxes)
        if best_iou > 0:  # any overlap counts as "survived"
            survived += 1
            matched_ious.append(best_iou)
    survival_rate = survived / len(clean_boxes)
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
    return mean_iou, survival_rate


# --- checkpointed run ---
if os.path.exists(BARON_CKPT_PATH):
    with open(BARON_CKPT_PATH) as f:
        results = json.load(f)
    print(f"✅ Loaded existing BARON spatial-integrity results, "
          f"{len(results)} images already processed")
else:
    results = []

test_files = image_files[:N_BARON_TEST]
already_done = {r["image"] for r in results}

for img_path in test_files:
    fname = os.path.basename(img_path)
    if fname in already_done:
        continue

    clean_img = loaded_images[img_path]
    clean_regions = get_baron_regions(clean_img)

    per_category = {}
    for category, corruption_list in [("Blur", BLUR_TYPES), ("Weather", WEATHER_TYPES)]:
        cat_ious, cat_survival = [], []
        for corruption in corruption_list:
            for severity in HIGH_SEVERITIES:
                c_img = corrupt(clean_img, corruption_name=corruption, severity=severity)
                corrupt_regions = get_baron_regions(c_img)
                mean_iou, survival = match_and_score(clean_regions, corrupt_regions)
                if mean_iou is not None:
                    cat_ious.append(mean_iou)
                    cat_survival.append(survival)
        per_category[category] = {
            "mean_iou": float(np.mean(cat_ious)) if cat_ious else None,
            "mean_survival_rate": float(np.mean(cat_survival)) if cat_survival else None,
        }

    results.append({
        "image": fname,
        "n_clean_regions": len(clean_regions),
        "Blur": per_category["Blur"],
        "Weather": per_category["Weather"],
    })

    with open(BARON_CKPT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[{len(results)}/{N_BARON_TEST}] {fname}: "
          f"Blur IoU={per_category['Blur']['mean_iou']}, "
          f"Weather IoU={per_category['Weather']['mean_iou']}")

# --- aggregate ---
blur_ious = [r["Blur"]["mean_iou"] for r in results if r["Blur"]["mean_iou"] is not None]
blur_surv = [r["Blur"]["mean_survival_rate"] for r in results if r["Blur"]["mean_survival_rate"] is not None]
weather_ious = [r["Weather"]["mean_iou"] for r in results if r["Weather"]["mean_iou"] is not None]
weather_surv = [r["Weather"]["mean_survival_rate"] for r in results if r["Weather"]["mean_survival_rate"] is not None]

print(f"\n{'='*55}")
print(f"BARON SPATIAL INTEGRITY — HIGH SEVERITY (4-5), N={len(results)}")
print(f"{'='*55}")
print(f"Blur    — mean IoU: {np.mean(blur_ious):.4f}   mean survival rate: {np.mean(blur_surv):.4f}")
print(f"Weather — mean IoU: {np.mean(weather_ious):.4f}   mean survival rate: {np.mean(weather_surv):.4f}")
print(f"{'='*55}")
