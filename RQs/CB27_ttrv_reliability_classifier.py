# ============================================================
# BLOCK 27: TTRV Gate as Reliability Classifier: Precision Analysis
# Runtime estimate: ~2-3 minutes at N=20
# ============================================================
import torch
import numpy as np
import pandas as pd
from torchvision.ops import box_iou
from tqdm.notebook import tqdm
import os

print("=" * 65)
print("BLOCK 27: TTRV Gate Reliability Analysis — Verified vs Rejected Precision")
print("=" * 65)
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

# FIX: original HOLDOUT_START=60 sliced past the end of a
# 20-image pool, returning an empty image list. Reset to 0 so
# this actually evaluates all 20 available images.
HOLDOUT_START = 0
HOLDOUT_N = 20
RELIAB_CORRUPTIONS = ["motion_blur", "contrast"]
RELIAB_SEVERITY = 5

records = []
for corruption in RELIAB_CORRUPTIONS:
    for img_path in tqdm(image_files[HOLDOUT_START:HOLDOUT_START+HOLDOUT_N],
                          desc=f"{corruption} sev{RELIAB_SEVERITY}"):
        img_id  = img_id_map[os.path.basename(img_path)]
        raw_img = loaded_images[img_path]
        c_img   = apply_corruption_deterministic(raw_img, img_id, corruption, RELIAB_SEVERITY)

        dets = b24_compute_snr(c_img)
        if not dets:
            continue

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
            records.append({
                "corruption": corruption, "img_id": img_id,
                "label": lbl, "verified": verified, "correct": correct,
            })

df_reliab = pd.DataFrame(records)
print(f"\nTotal detections analyzed: {len(df_reliab)}")
print()

print("=" * 65)
print("RESULTS — Precision by Gate Decision")
print("=" * 65)
summary = df_reliab.groupby("verified")["correct"].agg(["mean", "count"])
summary.columns = ["precision", "n_detections"]
print(summary)
print()
print(f"Verified-set precision   : {df_reliab[df_reliab.verified]['correct'].mean():.3f}")
print(f"Rejected-set precision   : {df_reliab[~df_reliab.verified]['correct'].mean():.3f}")

os.makedirs("/kaggle/working/tables", exist_ok=True)
df_reliab.to_csv("/kaggle/working/tables/table_ttrv_reliability.csv", index=False)
print("\n✅ Saved: table_ttrv_reliability.csv")

# ─────────────────────────────────────────────────────────────
# Significance Check: TTRV Verification vs Detection Correctness
# Runtime: instant (no model inference, just stats on saved data)
# ─────────────────────────────────────────────────────────────
from scipy.stats import chi2_contingency, fisher_exact

print()
print("=" * 65)
print("Significance Test: Gate Decision vs Correctness")
print("=" * 65)
print()

contingency = pd.crosstab(df_reliab["verified"], df_reliab["correct"])
print("Contingency table (rows=verified, cols=correct):")
print(contingency)
print()

chi2, p_chi2, dof, expected = chi2_contingency(contingency)
print(f"Chi-square test:  χ² = {chi2:.3f}, p = {p_chi2:.4f}")

odds_ratio, p_fisher = fisher_exact(contingency)
print(f"Fisher's exact test:  odds ratio = {odds_ratio:.3f}, p = {p_fisher:.4f}")
print()

print("Expected cell counts (chi-square assumes all ≥5):")
print(pd.DataFrame(expected, index=contingency.index, columns=contingency.columns).round(1))
print()

if (expected < 5).any():
    print("⚠️  At least one expected cell count is below 5 — chi-square may be unreliable here.")
    print("   Fisher's exact test is the more trustworthy result at this sample size.")
else:
    print("✅ All expected cell counts ≥5 — chi-square assumptions reasonably satisfied.")

print()
print("=" * 65)
if p_fisher < 0.05:
    print(f"✅ RESULT: statistically significant (Fisher p={p_fisher:.4f} < 0.05)")
    print("   The gate's verification decision is significantly associated with correctness,")
    print("   even at this sample size.")
else:
    print(f"➖ RESULT: not statistically significant at this sample size (Fisher p={p_fisher:.4f})")
    print("   The precision gap is suggestive but not yet confirmed — more data needed.")
print("=" * 65)
print()
print("=" * 65)
print("BLOCK 27 COMPLETE")
print("=" * 65)
