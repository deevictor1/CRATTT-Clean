# ============================================================
# BLOCK 24b: INTERNAL TTRV — Fixed Two-Pass Calibration
# + PER-IMAGE EPISODIC RESET FIX
# Dada Victor Damilare | MRES7015 | University of Greater Manchester
# ============================================================
#
# WHY THIS BLOCK EXISTS
# ─────────────────────
# Block 24 had a calibration bug (raw vs normalised Rjoint scale
# mismatch) — fixed in 24b.2 below via two-pass calibration.
#
# SECOND BUG FOUND AND FIXED HERE (24b.5):
# LoRA's B matrices were only reset once per corruption TYPE,
# not once per IMAGE. This meant image i's TTT update could
# leak into image i+1's "Baseline" measurement within the same
# corruption loop, since the model was never returned to its
# frozen state between images. This made Baseline, CRATTT_Int,
# and CRATTT_TTT all path-dependent on the order and outcome of
# every prior image — violating standard episodic TTT, where
# each test instance must adapt from the same fixed starting
# point. Fixed by resetting LoRA + reinitialising the optimiser
# before every single image.
# ============================================================

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import os

from imagecorruptions import corrupt as ic_corrupt
from tqdm.notebook import tqdm
from IPython.display import display

print("=" * 65)
print("BLOCK 24b: INTERNAL TTRV — Fixed Calibration + Episodic Reset")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────
# 24b.1  VERIFY BLOCK 24 FUNCTIONS EXIST
# ─────────────────────────────────────────────────────────────
required = [
    "b24_augment", "b24_run_dino", "b24_compute_snr",
    "b24_apply_gate", "b24_confidence_loss", "B24",
]
missing = [f for f in required if f not in globals()]
if missing:
    raise RuntimeError(
        f"These functions from Block 24 are missing: {missing}\n"
        "Re-run Block 24 first (the full block, not just 24b)."
    )
print("✅ All Block 24 helper functions present")
print()

# ─────────────────────────────────────────────────────────────
# 24b.2  FIXED TWO-PASS CALIBRATION (unchanged from before)
# ─────────────────────────────────────────────────────────────
print("─" * 50)
print("24b.2  Fixed Two-Pass Calibration")
print("─" * 50)

min_cons = B24["consistency_thr"] / B24["n_views"]

raw_snr_vals  = []
raw_det_cache = []

for img_path in image_files[: B24["n_calib"]]:
    raw_img = loaded_images[img_path]
    dets    = b24_compute_snr(raw_img)
    raw_det_cache.append(dets)
    for d in dets:
        if d["iou_consensus"] >= min_cons:
            raw_snr_vals.append(d["s_snr"])

if len(raw_snr_vals) < 5:
    SNR_BASELINE_24b = 3.0
    print("⚠️  Fewer than 5 calibration points — using fallback SNR_BASELINE = 3.0")
else:
    SNR_BASELINE_24b = float(np.percentile(raw_snr_vals, 25))

print(f"  Pass 1 — SNR values collected : {len(raw_snr_vals)}")
print(f"  SNR mean  (clean)             : {np.mean(raw_snr_vals):.2f}")
print(f"  SNR median (clean)            : {np.median(raw_snr_vals):.2f}")
print(f"  SNR p25 → SNR_BASELINE        : {SNR_BASELINE_24b:.3f}")

norm_rjoint_vals = []
for dets in raw_det_cache:
    for d in dets:
        if d["iou_consensus"] >= min_cons:
            snr_norm = min(d["s_snr"] / max(SNR_BASELINE_24b, 1e-6), 2.0)
            rj = (d["iou_consensus"] ** B24["alpha"] *
                  snr_norm           ** B24["beta"])
            norm_rjoint_vals.append(rj)

if len(norm_rjoint_vals) < 5:
    TAU_INTERNAL_24b = 0.30
    print("⚠️  Fewer than 5 normalised Rjoint values — using fallback τ = 0.30")
else:
    TAU_INTERNAL_24b = float(np.percentile(norm_rjoint_vals, 25))

print()
print(f"  Pass 2 — Normalised Rjoint values : {len(norm_rjoint_vals)}")
print(f"  Rjoint mean                       : {np.mean(norm_rjoint_vals):.4f}")
print(f"  Rjoint median                     : {np.median(norm_rjoint_vals):.4f}")
print(f"  τ_internal (p25) → TAU_INTERNAL   : {TAU_INTERNAL_24b:.4f}")
print()

B24["snr_baseline"] = SNR_BASELINE_24b
B24["tau_internal"] = TAU_INTERNAL_24b

# ─────────────────────────────────────────────────────────────
# 24b.3  GATE DIAGNOSTIC (unchanged)
# ─────────────────────────────────────────────────────────────
print("─" * 50)
print("24b.3  Gate Diagnostic — Corrupted Sample Image")
print("─" * 50)

_test_raw   = loaded_images[image_files[0]]
_test_c_img = ic_corrupt(_test_raw, corruption_name="gaussian_noise", severity=5)
_test_dets  = b24_compute_snr(_test_c_img)
_vb, _vl, _vr = b24_apply_gate(_test_dets)

print(f"  DINO proposals (corrupted)   : {len(_test_dets)}")
print(f"  Gate passed (verified)       : {len(_vb)}")
if _test_dets:
    _snr = [d["s_snr"] for d in _test_dets]
    _snr_norm = [min(s / max(SNR_BASELINE_24b, 1e-6), 2.0) for s in _snr]
    _iou      = [d["iou_consensus"] for d in _test_dets]
    _rj       = [ic**B24["alpha"] * sn**B24["beta"]
                 for ic, sn in zip(_iou, _snr_norm)]
    print(f"  Rjoint range                 : [{min(_rj):.3f}, {max(_rj):.3f}]")
    print(f"  τ_internal                   : {TAU_INTERNAL_24b:.4f}")

if len(_vb) == 0:
    print()
    print("  ⚠️  Still zero verified detections after calibration fix.")
    print("  Applying emergency tau reduction: τ × 0.6")
    B24["tau_internal"] = round(TAU_INTERNAL_24b * 0.6, 4)
    TAU_INTERNAL_24b    = B24["tau_internal"]
    _vb, _vl, _vr       = b24_apply_gate(_test_dets)
    print(f"  New τ_internal: {TAU_INTERNAL_24b:.4f}")
    print(f"  Gate passed after reduction: {len(_vb)}")
else:
    print(f"  ✅ Gate is active — {len(_vb)} verified on sample corrupted image")
print()

# ─────────────────────────────────────────────────────────────
# 24b.4  INITIAL LORA RESET (kept for a clean starting state;
# the real per-episode reset now happens inside 24b.5)
# ─────────────────────────────────────────────────────────────
print("─" * 50)
print("24b.4  Initial LoRA Reset")
print("─" * 50)

def reset_lora_b():
    for name, param in dino_model.named_parameters():
        if "lora_B" in name:
            param.data.zero_()

reset_lora_b()
n_trainable = sum(p.numel() for p in dino_model.parameters()
                  if p.requires_grad)
print(f"  ✅ LoRA reset — trainable params: {n_trainable:,}")
print()

# ─────────────────────────────────────────────────────────────
# 24b.5  MAIN EVALUATION LOOP — FIXED: per-image episodic reset
# ─────────────────────────────────────────────────────────────
print("─" * 50)
print("24b.5  Main Evaluation (per-image episodic reset)")
print("─" * 50)
print()

all_rows_24b = []

for (cat, corruption, severity) in B24["corruptions"]:

    cat_rows   = []
    weight_log = []

    for img_path in tqdm(image_files[: B24["n_images"]],
                          desc=f"{corruption} sev{severity}"):

        # ── FIX: fresh LoRA state + fresh optimiser, every image ──
        reset_lora_b()
        opt_img = torch.optim.AdamW(
            [p for p in dino_model.parameters() if p.requires_grad],
            lr=B24["ttt_lr"],
            weight_decay=1e-4,
        )

        img_id  = img_id_map[os.path.basename(img_path)]
        raw_img = loaded_images[img_path]
        c_img   = ic_corrupt(raw_img,
                              corruption_name=corruption,
                              severity=severity)

        # ── A: Baseline (guaranteed frozen — model just reset) ──
        dino_model.eval()
        boxes_b, scores_b, labels_b = b24_run_dino(c_img)

        preds_baseline = [
            {
                "image_id":    img_id,
                "category_id": COCO_MAP[l],
                "bbox": [b[0].item(), b[1].item(),
                         (b[2]-b[0]).item(), (b[3]-b[1]).item()],
                "score": s.item(),
            }
            for b, s, l in zip(boxes_b, scores_b, labels_b)
            if l in COCO_MAP
        ]
        mAP_base, _ = compute_map(preds_baseline, coco_gt, [img_id])

        # ── B: Internal TTRV (no TTT) ─────────────────────────
        dets = b24_compute_snr(c_img)
        v_boxes, v_labels, v_rjoints = b24_apply_gate(dets)

        preds_crattt = [
            {
                "image_id":    img_id,
                "category_id": COCO_MAP[l],
                "bbox": [b[0].item(), b[1].item(),
                         (b[2]-b[0]).item(), (b[3]-b[1]).item()],
                "score": rj,
            }
            for b, l, rj in zip(v_boxes, v_labels, v_rjoints)
            if l in COCO_MAP
        ]
        mAP_crattt, _ = compute_map(preds_crattt, coco_gt, [img_id])

        # ── C: TTT — starts from the same frozen state every time ──
        mAP_ttt   = mAP_crattt
        ttt_fired = False

        if len(v_boxes) > 0:
            dino_model.train()

            for _ in range(B24["ttt_steps"]):
                opt_img.zero_grad()
                loss = b24_confidence_loss(c_img, v_boxes, device)
                if loss is not None and loss.requires_grad:
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        [p for p in dino_model.parameters()
                         if p.requires_grad],
                        max_norm=B24["max_norm"],
                    )
                    opt_img.step()
                    ttt_fired = True

            dino_model.eval()

            if ttt_fired:
                boxes_t, scores_t, labels_t = b24_run_dino(c_img)
                preds_ttt = [
                    {
                        "image_id":    img_id,
                        "category_id": COCO_MAP[l],
                        "bbox": [b[0].item(), b[1].item(),
                                 (b[2]-b[0]).item(), (b[3]-b[1]).item()],
                        "score": s.item(),
                    }
                    for b, s, l in zip(boxes_t, scores_t, labels_t)
                    if l in COCO_MAP
                ]
                mAP_ttt, _ = compute_map(preds_ttt, coco_gt, [img_id])

                lora_delta = max(
                    p.data.abs().max().item()
                    for n, p in dino_model.named_parameters()
                    if "lora_B" in n
                )
                weight_log.append(lora_delta)

        cat_rows.append({
            "category":     cat,
            "corruption":   corruption,
            "img_id":       img_id,
            "mAP_baseline": round(float(mAP_base),   4),
            "mAP_crattt":   round(float(mAP_crattt), 4),
            "mAP_ttt":      round(float(mAP_ttt),    4),
            "TTT_gain":     round(float(mAP_ttt - mAP_crattt), 4),
            "n_verified":   len(v_boxes),
            "ttt_fired":    ttt_fired,
        })

    all_rows_24b.extend(cat_rows)

    df_cat     = pd.DataFrame(cat_rows)
    mean_gain  = df_cat["TTT_gain"].mean()
    pos_imgs   = int((df_cat["TTT_gain"] > 0).sum())
    fired_n    = int(df_cat["ttt_fired"].sum())
    mean_ver   = df_cat["n_verified"].mean()
    mean_wt    = float(np.mean(weight_log)) if weight_log else 0.0

    print(f"[{cat:7s} | {corruption:15s} | sev{severity}]  "
          f"gain: {mean_gain:+.4f}  |  "
          f"+ve: {pos_imgs}/{B24['n_images']}  |  "
          f"Fired: {fired_n}/{B24['n_images']}  |  "
          f"VerifiedAvg: {mean_ver:.1f}  |  "
          f"LoRA Δ: {mean_wt:.5f}")

# ─────────────────────────────────────────────────────────────
# 24b.6  RESULTS SUMMARY (unchanged)
# ─────────────────────────────────────────────────────────────
df_24b = pd.DataFrame(all_rows_24b)

summary_24b = (
    df_24b
    .groupby("corruption")
    .agg(
        Baseline   = ("mAP_baseline", "mean"),
        CRATTT_Int = ("mAP_crattt",   "mean"),
        CRATTT_TTT = ("mAP_ttt",      "mean"),
        TTT_gain   = ("TTT_gain",     "mean"),
        Verified   = ("n_verified",   "mean"),
        Fired      = ("ttt_fired",    "sum"),
    )
    .round(4)
)

overall_gain = float(df_24b["TTT_gain"].mean())
pos_total    = int((df_24b["TTT_gain"] > 0).sum())
total        = len(df_24b)
mean_ver     = df_24b["n_verified"].mean()

print()
print("=" * 65)
print("BLOCK 24b RESULTS — INTERNAL TTRV (FIXED CALIB + EPISODIC RESET)")
print("=" * 65)
display(summary_24b)

print(f"\nOverall TTT gain         : {overall_gain:+.4f}")
print(f"Images with TTT gain > 0 : {pos_total}/{total}")
print(f"Mean verified / image    : {mean_ver:.2f}")
print(f"τ_internal used          : {TAU_INTERNAL_24b:.4f}")
print(f"SNR baseline used        : {SNR_BASELINE_24b:.3f}")

os.makedirs("/kaggle/working/tables",  exist_ok=True)
os.makedirs("/kaggle/working/results", exist_ok=True)

df_24b.to_csv(
    "/kaggle/working/tables/table_block24b_episodic_fixed.csv",
    index=False,
)
with open("/kaggle/working/results/block24b_summary.json", "w") as f:
    json.dump({
        "block":            "24b",
        "method":           "Internal_TTRV_Score_SNR_FixedCalib_EpisodicReset",
        "tau_internal":     round(TAU_INTERNAL_24b, 4),
        "snr_baseline":     round(SNR_BASELINE_24b, 3),
        "overall_gain":     round(overall_gain, 4),
        "positive_images":  pos_total,
        "total_images":     total,
        "mean_verified":    round(float(mean_ver), 2),
    }, f, indent=2)

print(f"\n✅ Saved: table_block24b_episodic_fixed.csv")
print()
print("=" * 65)
print("BLOCK 24b COMPLETE")
print("=" * 65)
