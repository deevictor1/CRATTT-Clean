# ============================================================
# BLOCK 31: Continuous Reliability Analysis: Does Rjoint Predict Correctness?
# Runtime estimate: ~20-21 minutes (checkpointed, resumable)
# ============================================================
import torch
import numpy as np
import pandas as pd
from torchvision.ops import box_iou
from tqdm.notebook import tqdm
import os, gc

print("=" * 65)
print("BLOCK 31: Continuous Reliability Analysis — Rjoint vs Correctness")
print("=" * 65)
print()

# FIX: same as the N=200 and stratified blocks — zero lora_B
# before this purely-diagnostic run, since dino_model otherwise
# still carries leftover weights from Block 24d's training.
def reset_lora_b():
    for name, param in dino_model.named_parameters():
        if "lora_B" in name:
            param.data.zero_()

reset_lora_b()
dino_model.eval()
print("✅ LoRA B zeroed, model in eval mode — clean state")
print(f"✅ Image pool: {len(image_files)} (N_EVAL=200, already sufficient)")
print()

def match_detections_to_gt(boxes, labels, img_id, iou_thresh=0.5):
    if len(boxes) == 0:
        return []
    results = [False] * len(boxes)
    claimed_gt = set()
    for cat in set(labels):
        cat_id = COCO_MAP.get(cat)
        if cat_id is None:
            continue
        ann_ids = coco_gt.getAnnIds(imgIds=[img_id], catIds=[cat_id])
        anns = coco_gt.loadAnns(ann_ids)
        if not anns:
            continue
        gt_boxes = torch.tensor([
            [a["bbox"][0], a["bbox"][1], a["bbox"][0]+a["bbox"][2], a["bbox"][1]+a["bbox"][3]]
            for a in anns
        ], dtype=torch.float32, device=boxes.device)
        det_idx_for_cat = [i for i, l in enumerate(labels) if l == cat]
        for i in det_idx_for_cat:
            ious = box_iou(boxes[i].unsqueeze(0), gt_boxes)[0]
            best_iou, best_j = ious.max(0)
            gt_key = (cat, best_j.item())
            if best_iou.item() >= iou_thresh and gt_key not in claimed_gt:
                results[i] = True
                claimed_gt.add(gt_key)
    return results

N_EVAL = 200
CORRUPTIONS_CONT = ["motion_blur", "contrast"]
SEVERITY = 5
CHECKPOINT_DIR = "/kaggle/working/results/continuous_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def ckpt_path(corr):
    return os.path.join(CHECKPOINT_DIR, f"{corr}.csv")

all_records = []

for corruption in CORRUPTIONS_CONT:
    cpath = ckpt_path(corruption)

    if os.path.exists(cpath):
        df_c = pd.read_csv(cpath)
        all_records.extend(df_c.to_dict("records"))
        print(f"⏭️  {corruption} — checkpoint found ({len(df_c)} rows), skipping.")
        continue

    cell_records = []
    for idx, img_path in enumerate(tqdm(image_files[:N_EVAL], desc=f"{corruption} sev{SEVERITY}")):
        img_id  = img_id_map[os.path.basename(img_path)]
        raw_img = loaded_images[img_path]
        c_img   = apply_corruption_deterministic(raw_img, img_id, corruption, SEVERITY)

        dets = b24_compute_snr(c_img)
        if dets:
            all_boxes  = torch.stack([d["box"] for d in dets])
            all_labels = [d["label"] for d in dets]
            all_rjoints = [
                (d["iou_consensus"] ** B24["alpha"]) *
                (min(d["s_snr"] / max(B24["snr_baseline"], 1e-6), 2.0) ** B24["beta"])
                for d in dets
            ]
            correctness = match_detections_to_gt(all_boxes, all_labels, img_id)

            for lbl, rj, correct in zip(all_labels, all_rjoints, correctness):
                cell_records.append({
                    "corruption": corruption, "img_id": img_id,
                    "label": lbl, "rjoint": rj, "correct": correct,
                })

        if (idx + 1) % 20 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    df_cell = pd.DataFrame(cell_records)
    df_cell.to_csv(cpath, index=False)
    all_records.extend(cell_records)
    print(f"✅ Checkpoint saved: {cpath} ({len(df_cell)} rows)")

df_cont = pd.DataFrame(all_records)
print(f"\nTotal detections: {len(df_cont)}")

os.makedirs("/kaggle/working/tables", exist_ok=True)
df_cont.to_csv("/kaggle/working/tables/table_continuous_reliability.csv", index=False)

from sklearn.metrics import roc_auc_score, average_precision_score

print()
print("=" * 65)
print("DISCRIMINATION METRICS — Rjoint as a Continuous Predictor")
print("=" * 65)

auc = roc_auc_score(df_cont["correct"], df_cont["rjoint"])
ap  = average_precision_score(df_cont["correct"], df_cont["rjoint"])
print(f"AUC-ROC: {auc:.3f}  (0.5=no discrimination, 1.0=perfect)")
print(f"Average Precision: {ap:.3f}  (baseline rate: {df_cont['correct'].mean():.3f})")
print()

for corruption in CORRUPTIONS_CONT:
    df_c = df_cont[df_cont["corruption"] == corruption]
    auc_c = roc_auc_score(df_c["correct"], df_c["rjoint"])
    print(f"{corruption}: AUC = {auc_c:.3f}  (N={len(df_c)})")

print()
print("=" * 65)
print("CALIBRATION CURVE — Correctness Rate by Rjoint Decile")
print("=" * 65)

df_cont["rjoint_decile"] = pd.qcut(df_cont["rjoint"], 10, labels=False, duplicates="drop")
calib = df_cont.groupby("rjoint_decile").agg(
    rjoint_mean=("rjoint", "mean"),
    correctness_rate=("correct", "mean"),
    n=("correct", "count"),
).round(3)
print(calib)

is_monotonic = calib["correctness_rate"].is_monotonic_increasing
print()
print(f"Strictly monotonic increasing: {is_monotonic}")
if not is_monotonic:
    n_violations = sum(calib["correctness_rate"].diff().dropna() < 0)
    print(f"Number of decile-to-decile decreases: {n_violations} (out of {len(calib)-1} transitions)")

print()
print("✅ Saved: table_continuous_reliability.csv")
print()
print("=" * 65)
print("BLOCK 31 COMPLETE")
print("=" * 65)
