"""
RQ2 DIAGNOSTIC — Structured Proxy vs Entropy Baseline (Swin-B)
Adapted from cells 35, 36, 38 in crattt-clean-2.ipynb.

Runs standalone off just the consolidated recovery cell -- does
NOT need any of the Phase 5 signal-design scripts. Needs
clip_model/clip_processor (Block 9), which the standard
consolidated_recovery_cell.py already loads as part of its
minimal replay.

ADAPTATION: added a margin/cat_sim correlation-vs-correctness
print to the severe-corruption cell (motion_blur sev5), matching
what the gentle-corruption cell already printed, so both
conditions have the same diagnostic breakdown available.
"""

# ====================================================================
# PRE-FLIGHT CHECK
# ====================================================================
print("=" * 65)
print("PRE-FLIGHT CHECK -- RQ2 Diagnostic")
print("=" * 65)

required = ["category_order", "category_token_spans", "get_category_distribution",
            "clip_model", "clip_processor", "coco_gt", "COCO_MAP",
            "dino_model", "dino_processor", "DINO_TEXT_PROMPT", "device",
            "image_files", "loaded_images", "img_id_map"]

all_ok = True
for name in required:
    present = name in globals()
    print(f"{'✅' if present else '❌'} {name}")
    if not present:
        all_ok = False

print()
if all_ok:
    print("✅ All dependencies present -- proceed.")
else:
    raise RuntimeError("Missing dependencies above -- run the consolidated recovery cell first.")

# ====================================================================
# run_dino_with_categories_raw (verbatim, same validated logic
# as the patched b24_run_dino, also returning full distribution)
# ====================================================================
def run_dino_with_categories_raw(image_np, box_threshold=0.25):
    dino_model.eval()
    with torch.no_grad():
        inputs = dino_processor(images=image_np, text=DINO_TEXT_PROMPT, return_tensors="pt").to(device)
        outputs = dino_model(**inputs)

    cat_scores_raw, cat_dist = get_category_distribution(outputs, category_token_spans, category_order)
    max_raw_scores, max_cat_idx = cat_scores_raw.max(dim=-1)

    keep_mask    = max_raw_scores >= box_threshold
    keep_indices = keep_mask.nonzero(as_tuple=True)[0]

    if len(keep_indices) == 0:
        return [], [], [], [], None

    img_h, img_w = image_np.shape[:2]
    pred_cxcywh = outputs.pred_boxes[0]
    cx = pred_cxcywh[:, 0] * img_w
    cy = pred_cxcywh[:, 1] * img_h
    pw = pred_cxcywh[:, 2] * img_w
    ph = pred_cxcywh[:, 3] * img_h
    pred_xyxy = torch.stack([cx - pw/2, cy - ph/2, cx + pw/2, cy + ph/2], dim=-1)

    boxes     = pred_xyxy[keep_indices]
    scores    = max_raw_scores[keep_indices]
    labels    = [category_order[i] for i in max_cat_idx[keep_indices].tolist()]
    query_idx = keep_indices.tolist()
    full_dist = cat_dist[keep_indices]

    return boxes, scores, labels, query_idx, full_dist

print("✅ run_dino_with_categories_raw defined")

# ====================================================================
# RQ2 Diagnostic: motion_blur, severity 5 (verbatim + correlation print added)
# ====================================================================
# ============================================================
# RQ2 Diagnostic: CoT-PL-Inspired Proxy vs Entropy Baseline
# Runtime estimate: ~3-5 minutes (N=15, single corruption/severity)
# ============================================================
import torch
import numpy as np
import pandas as pd
from PIL import Image as PILImage
from torchvision.ops import box_iou
from imagecorruptions import corrupt as ic_corrupt
import random as _py_random
from tqdm.notebook import tqdm
from sklearn.metrics import roc_auc_score
import os

print("=" * 65)
print("RQ2 Diagnostic: Structured 3-Step Proxy vs Entropy Baseline")
print("=" * 65)
print()

required = ["category_order", "category_token_spans", "run_dino_with_categories_raw",
            "clip_model", "clip_processor", "coco_gt", "COCO_MAP"]
missing = [f for f in required if f not in globals()]
if missing:
    raise RuntimeError(f"Missing: {missing}\nRe-run Step 1 (category-distribution extraction) first.")

# ── Corruption seeding (self-contained) ──
CORRUPTION_SEED_OFFSET = {"gaussian_noise": 1000, "motion_blur": 2000, "snow": 3000, "contrast": 4000}
def seed_for_corruption(img_id, corruption, severity):
    offset = CORRUPTION_SEED_OFFSET.get(corruption, 9000)
    return (img_id * 100 + offset + severity) % (2**31)
def apply_corruption_deterministic(raw_img, img_id, corruption, severity):
    seed_val = seed_for_corruption(img_id, corruption, severity)
    np.random.seed(seed_val); _py_random.seed(seed_val)
    return ic_corrupt(raw_img, corruption_name=corruption, severity=severity)

# ── Ground-truth matching (self-contained) ──
def match_detections_to_gt(boxes, labels, img_id, iou_thresh=0.5):
    if len(boxes) == 0:
        return []
    results = [False] * len(boxes)
    claimed_gt = set()
    for cat in set(labels):
        cat_id = COCO_MAP.get(cat)
        if cat_id is None:
            continue
        anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=[img_id], catIds=[cat_id]))
        if not anns:
            continue
        gt_boxes = torch.tensor([
            [a["bbox"][0], a["bbox"][1], a["bbox"][0]+a["bbox"][2], a["bbox"][1]+a["bbox"][3]]
            for a in anns], dtype=torch.float32, device=boxes.device)
        for i in [i for i, l in enumerate(labels) if l == cat]:
            ious = box_iou(boxes[i].unsqueeze(0), gt_boxes)[0]
            best_iou, best_j = ious.max(0)
            key = (cat, best_j.item())
            if best_iou.item() >= iou_thresh and key not in claimed_gt:
                results[i] = True
                claimed_gt.add(key)
    return results

# ── Crop with padding ──
def crop_with_padding(image_np, box, pad_ratio=0.15):
    h, w = image_np.shape[:2]
    x1, y1, x2, y2 = box.tolist()
    bw, bh = x2 - x1, y2 - y1
    x1p, y1p = max(0, x1 - bw*pad_ratio), max(0, y1 - bh*pad_ratio)
    x2p, y2p = min(w, x2 + bw*pad_ratio), min(h, y2 + bh*pad_ratio)
    if x2p <= x1p or y2p <= y1p:
        return None
    crop = image_np[int(y1p):int(y2p), int(x1p):int(x2p)]
    return PILImage.fromarray(crop.astype(np.uint8)) if crop.size > 0 else None

# ── Step 2+3: Recognize (CLIP category sim) + Ground (vs background) ──
def clip_recognize_and_ground(crop_pil, category):
    inputs = clip_processor(
        text=[f"a photo of a {category}", "a photo of background"],
        images=crop_pil, return_tensors="pt", padding=True
    ).to(device)
    with torch.no_grad():
        outputs = clip_model(**inputs)
    logits = outputs.logits_per_image[0]
    return logits[0].item(), logits[1].item()  # cat_sim, bg_sim

def entropy_of_dist(dist_row, eps=1e-8):
    p = dist_row.clamp(min=eps)
    return -(p * torch.log(p)).sum().item()

# ── Main diagnostic loop ──
N_EVAL = 15
CORRUPTION = "motion_blur"
SEVERITY = 5

records = []
for img_path in tqdm(image_files[:N_EVAL], desc=f"{CORRUPTION} sev{SEVERITY}"):
    img_id = img_id_map[os.path.basename(img_path)]
    raw_img = loaded_images[img_path]
    c_img = apply_corruption_deterministic(raw_img, img_id, CORRUPTION, SEVERITY)

    boxes, scores, labels, qidx, dist = run_dino_with_categories_raw(c_img)
    if len(boxes) == 0:
        continue

    correctness = match_detections_to_gt(boxes, labels, img_id)

    for i in range(len(boxes)):
        crop = crop_with_padding(c_img, boxes[i])
        if crop is None:
            continue
        cat_sim, bg_sim = clip_recognize_and_ground(crop, labels[i])
        ent = entropy_of_dist(dist[i])

        records.append({
            "img_id": img_id, "label": labels[i], "correct": correctness[i],
            "cat_sim": cat_sim, "bg_sim": bg_sim, "margin": cat_sim - bg_sim,
            "entropy": ent,
        })

df_rq2 = pd.DataFrame(records)
print(f"\nTotal detections analyzed: {len(df_rq2)}")
print()

# ── Sanity check: eyeball a few examples ──
print("=" * 65)
print("SANITY CHECK — first 8 detections")
print("=" * 65)
print(df_rq2[["label", "correct", "cat_sim", "bg_sim", "margin", "entropy"]].head(8).round(3))
print()

# ── AUC comparison: does each signal discriminate correctness? ──
print("=" * 65)
print("DISCRIMINATION — AUC (continuous signal vs correctness)")
print("=" * 65)
auc_proxy = roc_auc_score(df_rq2["correct"], df_rq2["margin"])
auc_baseline = roc_auc_score(df_rq2["correct"], -df_rq2["entropy"])  # lower entropy = better
print(f"Structured proxy (margin)     AUC: {auc_proxy:.3f}")
print(f"Entropy baseline (-entropy)   AUC: {auc_baseline:.3f}")
print()

# ── Equal-rate binary comparison (median split, fair comparison) ──
print("=" * 65)
print("EQUAL-RATE COMPARISON — median split per method")
print("=" * 65)
df_rq2["proxy_verified"] = df_rq2["margin"] >= df_rq2["margin"].median()
df_rq2["baseline_verified"] = df_rq2["entropy"] <= df_rq2["entropy"].median()

for method, col in [("Structured proxy", "proxy_verified"), ("Entropy baseline", "baseline_verified")]:
    v_prec = df_rq2[df_rq2[col]]["correct"].mean()
    r_prec = df_rq2[~df_rq2[col]]["correct"].mean()
    print(f"{method:20s}  verified={v_prec:.3f}  rejected={r_prec:.3f}  gap={v_prec-r_prec:+.3f}")

print("=" * 65)
print("CORRELATION — signal components vs correctness (motion_blur sev5)")
print("=" * 65)
print("Correlation: margin vs correct  —", df_rq2["margin"].corr(df_rq2["correct"].astype(int)))
print("Correlation: cat_sim vs correct —", df_rq2["cat_sim"].corr(df_rq2["correct"].astype(int)))
print()

os.makedirs("/kaggle/working/tables", exist_ok=True)
df_rq2.to_csv("/kaggle/working/tables/table_rq2_diagnostic.csv", index=False)
print("\n✅ Saved: table_rq2_diagnostic.csv")

# ====================================================================
# RQ2 Diagnostic — Gentle Corruption Check: contrast, severity 1 (verbatim)
# ====================================================================
# ============================================================
# RQ2 Diagnostic — Gentle Corruption Check: Contrast, Severity 1
# Runtime estimate: <1 minute
# ============================================================
N_EVAL = 15
CORRUPTION = "contrast"
SEVERITY = 1

records_gentle = []
for img_path in tqdm(image_files[:N_EVAL], desc=f"{CORRUPTION} sev{SEVERITY}"):
    img_id = img_id_map[os.path.basename(img_path)]
    raw_img = loaded_images[img_path]
    c_img = apply_corruption_deterministic(raw_img, img_id, CORRUPTION, SEVERITY)

    boxes, scores, labels, qidx, dist = run_dino_with_categories_raw(c_img)
    if len(boxes) == 0:
        continue

    correctness = match_detections_to_gt(boxes, labels, img_id)

    for i in range(len(boxes)):
        crop = crop_with_padding(c_img, boxes[i])
        if crop is None:
            continue
        cat_sim, bg_sim = clip_recognize_and_ground(crop, labels[i])
        ent = entropy_of_dist(dist[i])

        records_gentle.append({
            "img_id": img_id, "label": labels[i], "correct": correctness[i],
            "cat_sim": cat_sim, "bg_sim": bg_sim, "margin": cat_sim - bg_sim,
            "entropy": ent,
        })

df_gentle = pd.DataFrame(records_gentle)
print(f"\nTotal detections analyzed: {len(df_gentle)}")
print()

print("=" * 65)
print("SANITY CHECK — first 8 detections (contrast, severity 1)")
print("=" * 65)
print(df_gentle[["label", "correct", "cat_sim", "bg_sim", "margin", "entropy"]].head(8).round(3))
print()

print("=" * 65)
print("DISCRIMINATION — AUC")
print("=" * 65)
auc_proxy_g = roc_auc_score(df_gentle["correct"], df_gentle["margin"])
auc_baseline_g = roc_auc_score(df_gentle["correct"], -df_gentle["entropy"])
print(f"Structured proxy (margin)     AUC: {auc_proxy_g:.3f}   (motion_blur sev5 was: 0.612)")
print(f"Entropy baseline (-entropy)   AUC: {auc_baseline_g:.3f}   (motion_blur sev5 was: 0.783)")
print()

print("Correlation: margin vs correct  —", df_gentle["margin"].corr(df_gentle["correct"].astype(int)))
print("Correlation: cat_sim vs correct —", df_gentle["cat_sim"].corr(df_gentle["correct"].astype(int)))
print()

print("=" * 65)
print("EQUAL-RATE COMPARISON — median split per method")
print("=" * 65)
df_gentle["proxy_verified"] = df_gentle["margin"] >= df_gentle["margin"].median()
df_gentle["baseline_verified"] = df_gentle["entropy"] <= df_gentle["entropy"].median()

for method, col in [("Structured proxy", "proxy_verified"), ("Entropy baseline", "baseline_verified")]:
    v_prec = df_gentle[df_gentle[col]]["correct"].mean()
    r_prec = df_gentle[~df_gentle[col]]["correct"].mean()
    print(f"{method:20s}  verified={v_prec:.3f}  rejected={r_prec:.3f}  gap={v_prec-r_prec:+.3f}")

os.makedirs("/kaggle/working/tables", exist_ok=True)
df_gentle.to_csv("/kaggle/working/tables/table_rq2_gentle.csv", index=False)
print("\n✅ Saved: table_rq2_gentle.csv")
