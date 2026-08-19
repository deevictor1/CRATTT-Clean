# ============================================================
# BLOCK 24d (fix): SEVERITY SWEEP — N=20, Severities 1, 3, 5
# (Episodic Reset + Seeded Corruption + Fallback-to-Baseline Fix)
# Dada Victor Damilare | MRES7015 | University of Greater Manchester
# ============================================================
#
# PURPOSE
# ───────
# This is the complete, corrected severity sweep, combining all
# three fixes discovered across this debugging process. Every
# prior result from this block (and from 24b/24c before the
# fixes) is now known to be invalid in one or more of these ways
# and should not be cited in the dissertation.
#
# SCALE NOTE: this run uses N=20 (the current image pool), not the
# originally-planned N=50. This was a deliberate runtime decision,
# not a missed step — Block 4b (image expansion) was intentionally
# skipped. Report this table's sample size as N=20 when writing up.
#
# FIX 1 — Episodic reset:
# LoRA's B matrices and the optimiser are reset before EVERY
# image, not once per (severity, corruption) combination —
# preventing adaptation from image i leaking into image i+1's
# "Baseline" measurement.
#
# FIX 2 — Deterministic corruption seeding:
# imagecorruptions.corrupt() draws from the global RNG with no
# seed control. Seeded from (img_id, corruption, severity)
# immediately before every corruption call — confirmed via a
# run-twice reproducibility test (Baseline matched exactly).
#
# FIX 3 — Fallback to baseline, not to the gated/empty set:
# When the internal gate verifies ZERO detections, TTT never
# runs — but mAP_ttt was previously defaulting to mAP_crattt
# (≈0 for an empty prediction set), creating catastrophic
# negative vs_baseline values on images where the frozen model
# may have been performing fine. Diagnosed via per-image analysis
# in an earlier debugging pass: worst outliers had n_verified=0.
# Fixed by defaulting mAP_ttt to mAP_baseline whenever TTT
# doesn't fire, for any reason.
#
# REQUIRES: Block 24 (helper functions) + Block 24c (calibrated
# SNR_BASELINE_24c, TAU_INTERNAL_24c) already run this session.
# ============================================================

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import os
import random as _py_random

from imagecorruptions import corrupt as ic_corrupt
from tqdm.notebook import tqdm
from IPython.display import display

print("=" * 65)
print("BLOCK 24d: Severity Sweep — All Three Fixes Combined (N=20)")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────
# 24d.0  Deterministic seeding + reset helpers (self-contained)
# ─────────────────────────────────────────────────────────────
CORRUPTION_SEED_OFFSET = {
    "gaussian_noise": 1_000,
    "motion_blur":    2_000,
    "snow":           3_000,
    "contrast":       4_000,
}

def seed_for_corruption(img_id: int, corruption: str, severity: int) -> int:
    offset = CORRUPTION_SEED_OFFSET.get(corruption, 9_000)
    return (img_id * 100 + offset + severity) % (2**31)

def apply_corruption_deterministic(raw_img, img_id, corruption, severity):
    seed_val = seed_for_corruption(img_id, corruption, severity)
    np.random.seed(seed_val)
    _py_random.seed(seed_val)
    return ic_corrupt(raw_img, corruption_name=corruption, severity=severity)

def reset_lora_b():
    for name, param in dino_model.named_parameters():
        if "lora_B" in name:
            param.data.zero_()

print("✅ Deterministic corruption seeding + reset helpers defined")
print()

# ─────────────────────────────────────────────────────────────
# 24d.1  Dependency check
# ─────────────────────────────────────────────────────────────
required = ["b24_compute_snr", "b24_apply_gate",
            "b24_confidence_loss", "b24_run_dino",
            "B24", "SNR_BASELINE_24c", "TAU_INTERNAL_24c"]
missing = [f for f in required if f not in globals()]
if missing:
    raise RuntimeError(f"Missing: {missing}\nRe-run Block 24 → Block 24c first.")

n_images = len(image_files)
print(f"✅ Helper functions present")
print(f"   SNR_BASELINE : {SNR_BASELINE_24c:.3f}")
print(f"   τ_internal   : {TAU_INTERNAL_24c:.4f}")
print(f"   N images     : {n_images}")
print()

# Updated: N=20 is the deliberate scale for this run 
if n_images == 20:
    print(f"✅ Running at N=20 (deliberate scale choice for this session)")
else:
    print(f"⚠️  image_files has {n_images} entries — unexpected size, "
          f"double-check which setup blocks ran this session.")

# ─────────────────────────────────────────────────────────────
# 24d.2  Reset LoRA first, before anything else
# ─────────────────────────────────────────────────────────────
reset_lora_b()
n_trainable = sum(p.numel() for p in dino_model.parameters()
                  if p.requires_grad)
print(f"✅ LoRA reset — trainable params: {n_trainable:,}")
print()

# ─────────────────────────────────────────────────────────────
# 24d.3  Configuration
# ─────────────────────────────────────────────────────────────
SEVERITIES_TO_TEST = [1, 3, 5]
CORRUPTIONS_D = [
    ("Noise",   "gaussian_noise"),
    ("Blur",    "motion_blur"),
    ("Weather", "snow"),
    ("Digital", "contrast"),
]
TTT_STEPS_D = 5
TTT_LR_D    = 1e-4

print(f"Severities    : {SEVERITIES_TO_TEST}")
print(f"Corruptions   : {[c for _, c in CORRUPTIONS_D]}")
print(f"TTT steps     : {TTT_STEPS_D}")
print(f"TTT lr        : {TTT_LR_D}")
print(f"Total runs    : {n_images * len(CORRUPTIONS_D) * len(SEVERITIES_TO_TEST)}")
print()

# ─────────────────────────────────────────────────────────────
# 24d.4  Main loop
# ─────────────────────────────────────────────────────────────
print("─" * 65)
print("24d.4  Main Evaluation")
print("─" * 65)
print()

all_rows_24d = []

for severity in SEVERITIES_TO_TEST:
    print(f"{'='*55}")
    print(f"SEVERITY {severity}")
    print(f"{'='*55}")

    for (cat, corruption) in CORRUPTIONS_D:

        cat_rows   = []
        weight_log = []

        for img_path in tqdm(
            image_files[:n_images],
            desc=f"  {corruption} sev{severity}",
            leave=False,
        ):
            reset_lora_b()
            opt_24d = torch.optim.AdamW(
                [p for p in dino_model.parameters() if p.requires_grad],
                lr=TTT_LR_D, weight_decay=1e-4,
            )

            img_id  = img_id_map[os.path.basename(img_path)]
            raw_img = loaded_images[img_path]
            c_img   = apply_corruption_deterministic(
                raw_img, img_id, corruption, severity
            )

            # ── A: Baseline ───────────────────────────────────
            dino_model.eval()
            boxes_b, scores_b, labels_b = b24_run_dino(c_img)
            preds_base = [
                {"image_id": img_id, "category_id": COCO_MAP[l],
                 "bbox": [b[0].item(), b[1].item(),
                          (b[2]-b[0]).item(), (b[3]-b[1]).item()],
                 "score": s.item()}
                for b, s, l in zip(boxes_b, scores_b, labels_b)
                if l in COCO_MAP
            ]
            mAP_base, _ = compute_map(preds_base, coco_gt, [img_id])

            # ── B: Internal TTRV gate ─────────────────────────
            dets = b24_compute_snr(c_img)
            v_boxes, v_labels, v_rjoints = b24_apply_gate(dets)
            preds_crattt = [
                {"image_id": img_id, "category_id": COCO_MAP[l],
                 "bbox": [b[0].item(), b[1].item(),
                          (b[2]-b[0]).item(), (b[3]-b[1]).item()],
                 "score": rj}
                for b, l, rj in zip(v_boxes, v_labels, v_rjoints)
                if l in COCO_MAP
            ]
            mAP_crattt, _ = compute_map(preds_crattt, coco_gt, [img_id])

            # ── C: TTT — FIX 3: fall back to BASELINE, not crattt ──
            mAP_ttt   = mAP_base
            ttt_fired = False

            if len(v_boxes) > 0:
                dino_model.train()
                for _ in range(TTT_STEPS_D):
                    opt_24d.zero_grad()
                    loss = b24_confidence_loss(c_img, v_boxes, device)
                    if loss is not None and loss.requires_grad:
                        loss.backward()
                        nn.utils.clip_grad_norm_(
                            [p for p in dino_model.parameters()
                             if p.requires_grad],
                            max_norm=B24["max_norm"],
                        )
                        opt_24d.step()
                        ttt_fired = True
                dino_model.eval()

                if ttt_fired:
                    boxes_t, scores_t, labels_t = b24_run_dino(c_img)
                    preds_ttt = [
                        {"image_id": img_id, "category_id": COCO_MAP[l],
                         "bbox": [b[0].item(), b[1].item(),
                                  (b[2]-b[0]).item(), (b[3]-b[1]).item()],
                         "score": s.item()}
                        for b, s, l in zip(boxes_t, scores_t, labels_t)
                        if l in COCO_MAP
                    ]
                    mAP_ttt, _ = compute_map(preds_ttt, coco_gt, [img_id])
                    weight_log.append(max(
                        p.data.abs().max().item()
                        for n, p in dino_model.named_parameters()
                        if "lora_B" in n
                    ))
                # else: mAP_ttt remains mAP_base (correct fallback)

            cat_rows.append({
                "severity":      severity,
                "category":      cat,
                "corruption":    corruption,
                "img_id":        img_id,
                "mAP_baseline":  round(float(mAP_base),   4),
                "mAP_crattt":    round(float(mAP_crattt), 4),
                "mAP_ttt":       round(float(mAP_ttt),    4),
                "TTT_gain":      round(float(mAP_ttt - mAP_crattt), 4),
                "vs_baseline":   round(float(mAP_ttt - mAP_base),   4),
                "beats_baseline": bool(mAP_ttt > mAP_base),
                "n_verified":    len(v_boxes),
                "ttt_fired":     ttt_fired,
            })

        all_rows_24d.extend(cat_rows)

        df_c       = pd.DataFrame(cat_rows)
        gain       = df_c["TTT_gain"].mean()
        vs_base    = df_c["vs_baseline"].mean()
        beats_n    = int(df_c["beats_baseline"].sum())
        fired_n    = int(df_c["ttt_fired"].sum())
        mean_ver   = df_c["n_verified"].mean()
        mean_wt    = float(np.mean(weight_log)) if weight_log else 0.0
        beats_flag = "✅" if vs_base > 0 else "  "

        print(f"  [{cat:7s} | {corruption:15s} | sev{severity}]  "
              f"gain: {gain:+.4f}  |  "
              f"vs baseline: {vs_base:+.4f} {beats_flag} |  "
              f"beats: {beats_n}/{n_images}  |  "
              f"fired: {fired_n}/{n_images}  |  "
              f"VerAvg: {mean_ver:.1f}")

    print()

# ─────────────────────────────────────────────────────────────
# 24d.5  Summary table
# ─────────────────────────────────────────────────────────────
df_24d = pd.DataFrame(all_rows_24d)

summary_d = (
    df_24d
    .groupby(["severity", "corruption"])
    .agg(
        Baseline    = ("mAP_baseline", "mean"),
        CRATTT_Int  = ("mAP_crattt",   "mean"),
        CRATTT_TTT  = ("mAP_ttt",      "mean"),
        TTT_gain    = ("TTT_gain",     "mean"),
        vs_Baseline = ("vs_baseline",  "mean"),
        Beats       = ("beats_baseline", "sum"),
        Coverage    = ("ttt_fired",    "sum"),
        VerifiedAvg = ("n_verified",   "mean"),
    )
    .round(4)
)

print()
print("=" * 65)
print(f"BLOCK 24d RESULTS — SEVERITY SWEEP, N={n_images} (ALL THREE FIXES)")
print("=" * 65)
display(summary_d)

# ─────────────────────────────────────────────────────────────
# 24d.6  Severity comparison
# ─────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("Severity Comparison — Overall Across All 4 Corruption Types")
print("─" * 65)
print(f"  {'Sev':4s}  {'TTT gain':10s}  {'vs Baseline':12s}  "
      f"{'Coverage':10s}  {'VerifiedAvg':12s}  Corruption types > baseline")

for sev in SEVERITIES_TO_TEST:
    df_s       = df_24d[df_24d["severity"] == sev]
    gain       = df_s["TTT_gain"].mean()
    vs_base    = df_s["vs_baseline"].mean()
    coverage   = df_s["ttt_fired"].sum()
    total      = len(df_s)
    ver_avg    = df_s["n_verified"].mean()
    beats_by_type = (df_s.groupby("corruption")["vs_baseline"].mean() > 0).sum()

    print(f"  {sev:<4d}  {gain:+10.4f}  {vs_base:+12.4f}  "
          f"{coverage:>4}/{total:<5}  {ver_avg:12.2f}  "
          f"{beats_by_type}/4 types")

# ─────────────────────────────────────────────────────────────
# 24d.7  Honest generalisation verdict — checks vs_baseline,
# not just TTT_gain (TTT_gain alone is misleading — see Block
# 24c's analysis: it mostly reflects recovery from the gate's
# own filtering cost, not genuine TTT improvement)
# ─────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("Generalisation Assessment (vs_baseline-based, not TTT_gain-based)")
print("─" * 65)

df_sev3  = df_24d[df_24d["severity"] == 3]
vs_s3    = df_sev3["vs_baseline"].mean()
cov_s3   = df_sev3["ttt_fired"].sum() / len(df_sev3)
beats_s3 = (df_sev3.groupby("corruption")["vs_baseline"].mean() > 0).sum()

df_sev5  = df_24d[df_24d["severity"] == 5]
vs_s5    = df_sev5["vs_baseline"].mean()
cov_s5   = df_sev5["ttt_fired"].sum() / len(df_sev5)
beats_s5 = (df_sev5.groupby("corruption")["vs_baseline"].mean() > 0).sum()

print(f"Severity 3: vs_baseline={vs_s3:+.4f}  coverage={cov_s3:.1%}  types>baseline={beats_s3}/4")
print(f"Severity 5: vs_baseline={vs_s5:+.4f}  coverage={cov_s5:.1%}  types>baseline={beats_s5}/4")
print()

if vs_s3 > 0.005 and beats_s3 >= 2:
    verdict = f"✅ BEATS BASELINE — vs_baseline={vs_s3:+.4f} at severity 3, {beats_s3}/4 types positive."
elif vs_s3 > -0.005:
    verdict = (f"➖ NEAR-NEUTRAL — vs_baseline={vs_s3:+.4f} at severity 3. "
               f"Gap is small, comparable to measurement noise. "
               f"Needs formal statistical testing (larger N) to determine "
               f"if this is genuinely zero, positive, or negative.")
else:
    verdict = f"❌ Still below baseline — vs_baseline={vs_s3:+.4f} at severity 3."

print(verdict)

# ─────────────────────────────────────────────────────────────
# 24d.8  Save
# ─────────────────────────────────────────────────────────────
os.makedirs("/kaggle/working/tables",  exist_ok=True)
os.makedirs("/kaggle/working/results", exist_ok=True)

# Filename now explicitly tagged n20 so it isn't confused with a
# future N=50 re-run of this same block.
csv_path = f"/kaggle/working/tables/table_block24d_severity_sweep_n{n_images}_fallback_fix.csv"
df_24d.to_csv(csv_path, index=False)

sev_summary = {}
for sev in SEVERITIES_TO_TEST:
    df_s = df_24d[df_24d["severity"] == sev]
    sev_summary[f"sev{sev}"] = {
        "ttt_gain":    round(df_s["TTT_gain"].mean(), 4),
        "vs_baseline": round(df_s["vs_baseline"].mean(), 4),
        "coverage_pct": round(df_s["ttt_fired"].sum() / len(df_s) * 100, 1),
        "verified_avg": round(df_s["n_verified"].mean(), 2),
        "types_beating_baseline": int(
            (df_s.groupby("corruption")["vs_baseline"].mean() > 0).sum()
        ),
    }

json_path = f"/kaggle/working/results/block24d_summary_n{n_images}.json"
with open(json_path, "w") as f:
    json.dump({
        "block":    "24d",
        "method":   "Internal_TTRV_SeveritySweep_AllThreeFixes",
        "n_images": n_images,
        "results":  sev_summary,
        "verdict":  verdict,
    }, f, indent=2)

print()
print(f"✅ Saved: {csv_path}")
print(f"✅ Saved: {json_path}")
print()
print("=" * 65)
print("BLOCK 24d COMPLETE")
print("=" * 65)
