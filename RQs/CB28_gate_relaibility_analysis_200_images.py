# Expand image pool to N=200
TARGET_N = 200
print(f"Current image pool size: {len(image_files)}")

if len(image_files) < TARGET_N:
    IMAGE_DIR_CURRENT = os.path.dirname(image_files[0])
    all_candidates = sorted(
        f for f in os.listdir(IMAGE_DIR_CURRENT) if f.lower().endswith(".jpg")
    )
    existing_basenames = {os.path.basename(p) for p in image_files}
    added = 0
    for fname in all_candidates:
        if len(image_files) >= TARGET_N:
            break
        if fname in existing_basenames:
            continue
        full_path = os.path.join(IMAGE_DIR_CURRENT, fname)
        try:
            candidate_img_id = int(os.path.splitext(fname)[0])
        except ValueError:
            continue
        if len(coco_gt.getAnnIds(imgIds=[candidate_img_id])) == 0:
            continue
        img_arr = np.array(PILImage.open(full_path).convert("RGB"))
        loaded_images[full_path] = img_arr
        img_id_map[fname] = candidate_img_id
        image_files.append(full_path)
        added += 1
    print(f"✅ Added {added} images — pool now at {len(image_files)}")
else:
    print("✅ Already have enough images")

# ============================================================
# BLOCK 28: TTRV Gate Reliability Analysis — N=200, Checkpointed
# Runtime estimate: ~20-22 minutes total (resumable if interrupted)
# ============================================================
import torch
import numpy as np
import pandas as pd
from torchvision.ops import box_iou
from tqdm.notebook import tqdm
import os, gc

print("=" * 65)
print("BLOCK 28: TTRV Gate Reliability Analysis — N=200")
print("=" * 65)
print()

# FIX: zero lora_B before this purely-diagnostic run. b24_compute_snr
# never trains anything here, but dino_model's lora_B currently holds
# whatever Block 24d's training loop last left it at — an arbitrary,
# undocumented value, not the clean calibrated state this analysis
# should characterize.
def reset_lora_b():
    for name, param in dino_model.named_parameters():
        if "lora_B" in name:
            param.data.zero_()

reset_lora_b()
dino_model.eval()
print("✅ LoRA B zeroed, model in eval mode — clean state for reliability analysis")
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
RELIAB_CORRUPTIONS = ["motion_blur", "contrast"]
RELIAB_SEVERITY = 5
CHECKPOINT_DIR = "/kaggle/working/results/reliability_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def checkpoint_path(corruption):
    return os.path.join(CHECKPOINT_DIR, f"{corruption}.csv")

all_records = []

for corruption in RELIAB_CORRUPTIONS:
    ckpt_path = checkpoint_path(corruption)

    if os.path.exists(ckpt_path):
        df_ckpt = pd.read_csv(ckpt_path)
        all_records.extend(df_ckpt.to_dict("records"))
        print(f"⏭️  {corruption} — checkpoint found ({len(df_ckpt)} rows), skipping.")
        continue

    corruption_records = []
    for idx, img_path in enumerate(tqdm(image_files[:N_EVAL], desc=f"{corruption} sev{RELIAB_SEVERITY}")):
        img_id  = img_id_map[os.path.basename(img_path)]
        raw_img = loaded_images[img_path]
        c_img   = apply_corruption_deterministic(raw_img, img_id, corruption, RELIAB_SEVERITY)

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
                corruption_records.append({
                    "corruption": corruption, "img_id": img_id,
                    "label": lbl, "verified": verified, "correct": correct,
                })

        if (idx + 1) % 20 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    df_corr = pd.DataFrame(corruption_records)
    df_corr.to_csv(ckpt_path, index=False)
    all_records.extend(corruption_records)
    print(f"✅ Checkpoint saved: {ckpt_path} ({len(df_corr)} rows)")

df_reliab_200 = pd.DataFrame(all_records)
print(f"\nTotal detections analyzed: {len(df_reliab_200)}")
print()

print("=" * 65)
print("RESULTS — Precision by Gate Decision (N=200)")
print("=" * 65)
summary = df_reliab_200.groupby("verified")["correct"].agg(["mean", "count"])
summary.columns = ["precision", "n_detections"]
print(summary)

os.makedirs("/kaggle/working/tables", exist_ok=True)
df_reliab_200.to_csv("/kaggle/working/tables/table_ttrv_reliability_n200.csv", index=False)
print("\n✅ Saved: table_ttrv_reliability_n200.csv")

# ─────────────────────────────────────────────────────────────
# Significance Check at N=200
# Runtime: instant
# ─────────────────────────────────────────────────────────────
from scipy.stats import chi2_contingency, fisher_exact

print()
print("=" * 65)
print("Significance Test (N=200): Gate Decision vs Correctness")
print("=" * 65)
print()

contingency_200 = pd.crosstab(df_reliab_200["verified"], df_reliab_200["correct"])
print("Contingency table (rows=verified, cols=correct):")
print(contingency_200)
print()

chi2, p_chi2, dof, expected = chi2_contingency(contingency_200)
print(f"Chi-square test:  χ² = {chi2:.3f}, p = {p_chi2:.6f}")

odds_ratio, p_fisher = fisher_exact(contingency_200)
print(f"Fisher's exact test:  odds ratio = {odds_ratio:.3f}, p = {p_fisher:.6f}")
print()

print("Expected cell counts:")
print(pd.DataFrame(expected, index=contingency_200.index, columns=contingency_200.columns).round(1))
print()

print("=" * 65)
if p_fisher < 0.05:
    print(f"✅ RESULT: statistically significant (Fisher p={p_fisher:.6f} < 0.05)")
    print(f"   Verified precision: {df_reliab_200[df_reliab_200.verified]['correct'].mean():.3f}")
    print(f"   Rejected precision: {df_reliab_200[~df_reliab_200.verified]['correct'].mean():.3f}")
    print(f"   Odds ratio: {odds_ratio:.3f}")
else:
    print(f"➖ RESULT: still not significant (Fisher p={p_fisher:.6f})")
print("=" * 65)

import numpy as np
p1 = df_reliab_200[df_reliab_200.verified]['correct'].mean()
p2 = df_reliab_200[~df_reliab_200.verified]['correct'].mean()
cohens_h = 2*np.arcsin(np.sqrt(p1)) - 2*np.arcsin(np.sqrt(p2))
print(f"\nCohen's h (effect size for proportions): {cohens_h:.3f}")
print("(0.2=small, 0.5=medium, 0.8=large, by conventional benchmarks)")

print()
print("=" * 65)
print("BLOCK 28 COMPLETE")
print("=" * 65)
