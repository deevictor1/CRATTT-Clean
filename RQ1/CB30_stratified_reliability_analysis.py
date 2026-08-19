# ============================================================
# BLOCK 30: Stratified Reliability Analysis — Severity × Corruption
# Stage A: Fast diagnostic, N=30 per condition
# Runtime estimate: ~30-35 minutes (checkpointed, resumable)
# ============================================================
import torch
import numpy as np
import pandas as pd
from torchvision.ops import box_iou
from tqdm.notebook import tqdm
import os, gc

print("=" * 65)
print("BLOCK 30: Stratified Reliability Analysis — Stage A (N=30 per condition)")
print("=" * 65)
print()

# FIX: same as the N=200 reliability block — zero lora_B before this
# purely-diagnostic run. dino_model otherwise still carries leftover
# weights from Block 24d's last training step, not the clean
# calibrated state this stratified breakdown should characterize.
def reset_lora_b():
    for name, param in dino_model.named_parameters():
        if "lora_B" in name:
            param.data.zero_()

reset_lora_b()
dino_model.eval()
print("✅ LoRA B zeroed, model in eval mode — clean state for stratified analysis")
print(f"✅ Image pool: {len(image_files)} (N_EVAL=30 per condition, no expansion needed)")
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

N_EVAL = 30
SEVERITIES = [1, 2, 3, 4, 5]
CORRUPTIONS_STRAT = ["gaussian_noise", "motion_blur", "snow", "contrast"]
CHECKPOINT_DIR = "/kaggle/working/results/stratified_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def ckpt_path(sev, corr):
    return os.path.join(CHECKPOINT_DIR, f"sev{sev}_{corr}.csv")

all_records = []

for sev in SEVERITIES:
    for corruption in CORRUPTIONS_STRAT:
        cpath = ckpt_path(sev, corruption)

        if os.path.exists(cpath):
            df_c = pd.read_csv(cpath)
            all_records.extend(df_c.to_dict("records"))
            print(f"⏭️  sev{sev} | {corruption} — checkpoint found ({len(df_c)} rows), skipping.")
            continue

        cell_records = []
        for idx, img_path in enumerate(tqdm(image_files[:N_EVAL], desc=f"sev{sev} {corruption}", leave=False)):
            img_id  = img_id_map[os.path.basename(img_path)]
            raw_img = loaded_images[img_path]
            c_img   = apply_corruption_deterministic(raw_img, img_id, corruption, sev)

            dets = b24_compute_snr(c_img)
            if dets:
                all_boxes  = torch.stack([d["box"] for d in dets])
                all_labels = [d["label"] for d in dets]
                all_rjoints = [
                    (d["iou_consensus"] ** B24["alpha"]) *
                    (min(d["s_snr"] / max(B24["snr_baseline"], 1e-6), 2.0) ** B24["beta"])
                    for d in dets
                ]
                is_verified = [rj >= B24["tau_internal"] for rj in all_rjoints]
                correctness = match_detections_to_gt(all_boxes, all_labels, img_id)

                for lbl, verified, correct in zip(all_labels, is_verified, correctness):
                    cell_records.append({
                        "severity": sev, "corruption": corruption,
                        "img_id": img_id, "label": lbl,
                        "verified": verified, "correct": correct,
                    })

            if (idx + 1) % 15 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        df_cell = pd.DataFrame(cell_records)
        df_cell.to_csv(cpath, index=False)
        all_records.extend(cell_records)

        v_prec = df_cell[df_cell.verified]["correct"].mean() if (df_cell.verified.sum() > 0) else float("nan")
        r_prec = df_cell[~df_cell.verified]["correct"].mean() if ((~df_cell.verified).sum() > 0) else float("nan")
        print(f"  sev{sev} | {corruption:15s}  verified={v_prec:.3f}  rejected={r_prec:.3f}  gap={v_prec-r_prec:+.3f}")

df_strat = pd.DataFrame(all_records)
print(f"\nTotal detections: {len(df_strat)}")

os.makedirs("/kaggle/working/tables", exist_ok=True)
df_strat.to_csv("/kaggle/working/tables/table_stratified_reliability.csv", index=False)

print()
print("=" * 65)
print("SUMMARY TABLE — Precision by Severity × Corruption")
print("=" * 65)
pivot = df_strat.groupby(["severity", "corruption", "verified"])["correct"].mean().unstack("verified")
pivot.columns = ["rejected_precision", "verified_precision"]
pivot["gap"] = pivot["verified_precision"] - pivot["rejected_precision"]
print(pivot.round(3))

print()
print("=" * 65)
print("SAMPLE-SIZE DIAGNOSTIC")
print("=" * 65)
print("Total detections per cell:")
print(df_strat.groupby(["severity", "corruption"]).size().unstack())
print()
print("Verified-detection count per cell (this is what each gap is actually based on):")
vcounts = df_strat.groupby(["severity", "corruption", "verified"]).size().unstack(fill_value=0)
vcounts.columns = ["rejected_n", "verified_n"]
print(vcounts)

print()
print("✅ Saved: table_stratified_reliability.csv")

print()
print("=" * 65)
print("BLOCK 30 COMPLETE")
print("=" * 65)
