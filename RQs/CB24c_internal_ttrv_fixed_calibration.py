# ============================================================
# BLOCK 24c: INTERNAL TTRV — Fixed Calibration + Episodic Reset
#            + DETERMINISTIC CORRUPTION SEEDING
# ============================================================
#
# WHY THIS BLOCK EXISTS
# ─────────────────────
# This replaces the earlier Block 24c (which tested ONLINE/
# CONTINUAL domain accumulation across a 50-image sequence —
# a different, secondary research question). That version is
# set aside for now; this slot is repurposed for the block that
# produces the primary, reproducible N=20 result.
#
# Three issues found and fixed across this and earlier sessions:
#
#   FIX 1 — Episodic reset:
#   LoRA's B matrices were only reset once per corruption TYPE,
#   not once per IMAGE — so adaptation from image i leaked into
#   image i+1's "Baseline" measurement. Fixed by resetting LoRA
#   + reinitialising the optimiser before every single image.
#
#   FIX 2 — Deterministic corruption seeding:
#   imagecorruptions.corrupt() draws from the global NumPy/Python
#   RNG with no seed control. Most corruption types involve
#   internal random sampling, so the SAME image/corruption/severity
#   produced a DIFFERENT corrupted image every run. Confirmed by
#   comparing Baseline mAP across runs: 'contrast' (deterministic
#   pixel rescale) matched exactly; 'gaussian_noise', 'motion_blur',
#   'snow' (all internally randomised) did not. Fixed by seeding
#   both RNGs from (img_id, corruption, severity) immediately
#   before every ic_corrupt() call.
#
#   FIX 3 — Reset-before-calibration ordering:
#   The LoRA reset previously ran AFTER calibration. If a prior
#   run left LoRA's weights non-zero (because TTT fired on the
#   last image processed), the NEXT run's calibration would be
#   computed against that contaminated state — making "run it
#   twice and compare" an invalid test. Fixed by moving the reset
#   to the very top of the block, before anything else runs.
#
#   FIX 4 — TTT_vs_baseline diagnostic (NEW, this version):
#   Block 24b's headline "TTT_gain" is mAP_ttt - mAP_crattt,
#   measured against the GATE-FILTERED subset, not the untouched
#   model. Since mAP_ttt is computed on the full, unfiltered
#   detection set (same as Baseline), and the LoRA weight delta
#   in 24b was nearly identical across all 4 very different
#   corruptions (~0.0005 every time — the signature of gradient
#   clipping dominating the update, not real signal), there's a
#   real risk "TTT_gain" was mostly just "full detections vs
#   gate-filtered subset", not genuine evidence TTT improved
#   anything. TTT_vs_baseline = mAP_ttt - mAP_baseline isolates
#   the real question: does TTT beat doing nothing at all?
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
print("BLOCK 24c: INTERNAL TTRV — Episodic Reset + Deterministic Corruption Seeding")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────
# 24c.0  DETERMINISTIC SEEDING HELPER
#
# Built from plain integer arithmetic, NOT Python's hash() —
# hash() on strings is randomised per-session by default
# (PYTHONHASHSEED), which would silently reintroduce the exact
# non-determinism we're trying to eliminate.
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

print("✅ Deterministic corruption seeding helper defined")
print()

# ─────────────────────────────────────────────────────────────
# 24c.1  VERIFY BLOCK 24 FUNCTIONS EXIST
# ─────────────────────────────────────────────────────────────
required = [
    "b24_augment", "b24_run_dino", "b24_compute_snr",
    "b24_apply_gate", "b24_confidence_loss", "B24",
]
missing = [f for f in required if f not in globals()]
if missing:
    raise RuntimeError(
        f"These functions from Block 24 are missing: {missing}\n"
        "Re-run Block 24 first."
    )
print("✅ All Block 24 helper functions present")
print()

# ─────────────────────────────────────────────────────────────
# 24c.2  RESET LORA — MOVED TO THE TOP (FIX 3)
# Runs BEFORE calibration so every run of this block, including
# back-to-back re-runs without restarting the kernel, starts
# from an identical, known-clean state.
# ─────────────────────────────────────────────────────────────
print("─" * 50)
print("24c.2  Reset LoRA (runs first, before calibration)")
print("─" * 50)

def reset_lora_b():
    for name, param in dino_model.named_parameters():
        if "lora_B" in name:
            param.data.zero_()

reset_lora_b()
n_trainable = sum(p.numel() for p in dino_model.parameters()
                  if p.requires_grad)
print(f"✅ LoRA reset — trainable params: {n_trainable:,}")
print()

# ─────────────────────────────────────────────────────────────
# 24c.3  FIXED TWO-PASS CALIBRATION
# (Calibration runs on CLEAN images, no corruption involved,
# so it was never affected by the seeding bug — but it WAS
# affected by the reset-ordering bug, now fixed above.)
# ─────────────────────────────────────────────────────────────
print("─" * 50)
print("24c.3  Fixed Two-Pass Calibration")
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
    SNR_BASELINE_24c = 3.0
    print("⚠️  Fewer than 5 calibration points — using fallback SNR_BASELINE = 3.0")
else:
    SNR_BASELINE_24c = float(np.percentile(raw_snr_vals, 25))

print(f"  Pass 1 — SNR values collected : {len(raw_snr_vals)}")
print(f"  SNR mean  (clean)             : {np.mean(raw_snr_vals):.2f}")
print(f"  SNR median (clean)            : {np.median(raw_snr_vals):.2f}")
print(f"  SNR p25 → SNR_BASELINE        : {SNR_BASELINE_24c:.3f}")

norm_rjoint_vals = []
for dets in raw_det_cache:
    for d in dets:
        if d["iou_consensus"] >= min_cons:
            snr_norm = min(d["s_snr"] / max(SNR_BASELINE_24c, 1e-6), 2.0)
            rj = (d["iou_consensus"] ** B24["alpha"] *
                  snr_norm           ** B24["beta"])
            norm_rjoint_vals.append(rj)

if len(norm_rjoint_vals) < 5:
    TAU_INTERNAL_24c = 0.30
    print("⚠️  Fewer than 5 normalised Rjoint values — using fallback τ = 0.30")
else:
    TAU_INTERNAL_24c = float(np.percentile(norm_rjoint_vals, 25))

print()
print(f"  Pass 2 — Normalised Rjoint values : {len(norm_rjoint_vals)}")
print(f"  Rjoint mean                       : {np.mean(norm_rjoint_vals):.4f}")
print(f"  Rjoint median                     : {np.median(norm_rjoint_vals):.4f}")
print(f"  τ_internal (p25) → TAU_INTERNAL   : {TAU_INTERNAL_24c:.4f}")
print()

B24["snr_baseline"] = SNR_BASELINE_24c
B24["tau_internal"] = TAU_INTERNAL_24c

# ─────────────────────────────────────────────────────────────
# 24c.4  GATE DIAGNOSTIC (now using deterministic corruption)
# ─────────────────────────────────────────────────────────────
print("─" * 50)
print("24c.4  Gate Diagnostic — Corrupted Sample Image")
print("─" * 50)

_test_img_path = image_files[0]
_test_img_id   = img_id_map[os.path.basename(_test_img_path)]
_test_raw      = loaded_images[_test_img_path]
_test_c_img    = apply_corruption_deterministic(
    _test_raw, _test_img_id, "gaussian_noise", 5
)
_test_dets  = b24_compute_snr(_test_c_img)
_vb, _vl, _vr = b24_apply_gate(_test_dets)

print(f"  DINO proposals (corrupted)   : {len(_test_dets)}")
print(f"  Gate passed (verified)       : {len(_vb)}")
if _test_dets:
    _snr = [d["s_snr"] for d in _test_dets]
    print(f"  SNR range (corrupted)        : [{min(_snr):.1f}, {max(_snr):.1f}]")

if len(_vb) == 0:
    print()
    print("  ⚠️  Still zero verified detections after calibration fix.")
    print("  Applying emergency tau reduction: τ × 0.6")
    B24["tau_internal"] = round(TAU_INTERNAL_24c * 0.6, 4)
    TAU_INTERNAL_24c    = B24["tau_internal"]
    _vb, _vl, _vr       = b24_apply_gate(_test_dets)
    print(f"  New τ_internal: {TAU_INTERNAL_24c:.4f}")
    print(f"  Gate passed after reduction: {len(_vb)}")
else:
    print(f"  ✅ Gate is active — {len(_vb)} verified on sample corrupted image")
print()

# ─────────────────────────────────────────────────────────────
# 24c.5  MAIN EVALUATION LOOP
# Per-image episodic reset + deterministic corruption seeding
# ─────────────────────────────────────────────────────────────
print("─" * 50)
print("24c.5  Main Evaluation (episodic reset + seeded corruption)")
print("─" * 50)
print()

all_rows_24c = []

for (cat, corruption, severity) in B24["corruptions"]:

    cat_rows   = []
    weight_log = []

    for img_path in tqdm(image_files[: B24["n_images"]],
                          desc=f"{corruption} sev{severity}"):

        reset_lora_b()
        opt_img = torch.optim.AdamW(
            [p for p in dino_model.parameters() if p.requires_grad],
            lr=B24["ttt_lr"],
            weight_decay=1e-4,
        )

        img_id  = img_id_map[os.path.basename(img_path)]
        raw_img = loaded_images[img_path]
        c_img   = apply_corruption_deterministic(
            raw_img, img_id, corruption, severity
        )

        # ── A: Baseline ───────────────────────────────────────
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

        # ── B: Internal TTRV (no TTT) ───────────────────────────
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

        # ── C: TTT ───────────────────────────────────────────
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
            "category":        cat,
            "corruption":      corruption,
            "img_id":          img_id,
            "mAP_baseline":    round(float(mAP_base),   4),
            "mAP_crattt":      round(float(mAP_crattt), 4),
            "mAP_ttt":         round(float(mAP_ttt),    4),
            "TTT_gain":        round(float(mAP_ttt - mAP_crattt), 4),
            # FIX 4: the real test — does TTT beat the untouched
            # model, not just the gate-filtered subset?
            "TTT_vs_baseline": round(float(mAP_ttt - mAP_base), 4),
            "n_verified":      len(v_boxes),
            "ttt_fired":       ttt_fired,
        })

    all_rows_24c.extend(cat_rows)

    df_cat     = pd.DataFrame(cat_rows)
    mean_gain  = df_cat["TTT_gain"].mean()
    mean_vbase = df_cat["TTT_vs_baseline"].mean()
    pos_imgs   = int((df_cat["TTT_gain"] > 0).sum())
    fired_n    = int(df_cat["ttt_fired"].sum())
    mean_ver   = df_cat["n_verified"].mean()
    mean_wt    = float(np.mean(weight_log)) if weight_log else 0.0

    print(f"[{cat:7s} | {corruption:15s} | sev{severity}]  "
          f"gain(vs gate): {mean_gain:+.4f}  |  "
          f"gain(vs base): {mean_vbase:+.4f}  |  "
          f"+ve: {pos_imgs}/{B24['n_images']}  |  "
          f"Fired: {fired_n}/{B24['n_images']}  |  "
          f"VerifiedAvg: {mean_ver:.1f}  |  "
          f"LoRA Δ: {mean_wt:.5f}")

# ─────────────────────────────────────────────────────────────
# 24c.6  RESULTS SUMMARY
# ─────────────────────────────────────────────────────────────
df_24c = pd.DataFrame(all_rows_24c)

summary_24c = (
    df_24c
    .groupby("corruption")
    .agg(
        Baseline        = ("mAP_baseline",    "mean"),
        CRATTT_Int      = ("mAP_crattt",       "mean"),
        CRATTT_TTT      = ("mAP_ttt",          "mean"),
        TTT_gain        = ("TTT_gain",         "mean"),
        TTT_vs_Baseline = ("TTT_vs_baseline",  "mean"),
        Verified        = ("n_verified",       "mean"),
        Fired           = ("ttt_fired",        "sum"),
    )
    .round(4)
)

overall_gain    = float(df_24c["TTT_gain"].mean())
overall_vbase   = float(df_24c["TTT_vs_baseline"].mean())
pos_total       = int((df_24c["TTT_gain"] > 0).sum())
pos_vbase_total = int((df_24c["TTT_vs_baseline"] > 0).sum())
total           = len(df_24c)
mean_ver        = df_24c["n_verified"].mean()

print()
print("=" * 65)
print("BLOCK 24c RESULTS — EPISODIC RESET + SEEDED CORRUPTION")
print("=" * 65)
display(summary_24c)

print(f"\nOverall TTT gain (vs gate-filtered) : {overall_gain:+.4f}")
print(f"Images with gain(vs gate) > 0       : {pos_total}/{total}")
print()
print(f"Overall TTT gain (vs UNTOUCHED BASELINE) : {overall_vbase:+.4f}")
print(f"  ↑ This is the real test: does TTT beat doing nothing at all?")
print(f"Images with gain(vs baseline) > 0        : {pos_vbase_total}/{total}")
print()
print(f"Mean verified / image    : {mean_ver:.2f}")
print(f"τ_internal used          : {TAU_INTERNAL_24c:.4f}")
print(f"SNR baseline used        : {SNR_BASELINE_24c:.3f}")

if abs(overall_vbase) < 0.005:
    print()
    print("⚠️  TTT_vs_baseline is near zero — this suggests the headline")
    print("   'TTT_gain' figure above is mostly recovering mAP lost to")
    print("   gate filtering, not evidence that TTT-adapted weights")
    print("   genuinely improve detection quality over the untouched")
    print("   model. Worth reporting TTT_vs_baseline as the primary")
    print("   number, not TTT_gain, if writing this up as a finding.")

os.makedirs("/kaggle/working/tables",  exist_ok=True)
os.makedirs("/kaggle/working/results", exist_ok=True)

df_24c.to_csv(
    "/kaggle/working/tables/table_block24c_seeded.csv",
    index=False,
)
with open("/kaggle/working/results/block24c_summary.json", "w") as f:
    json.dump({
        "block":              "24c",
        "method":             "Internal_TTRV_EpisodicReset_SeededCorruption",
        "tau_internal":       round(TAU_INTERNAL_24c, 4),
        "snr_baseline":       round(SNR_BASELINE_24c, 3),
        "overall_gain_vs_gate":     round(overall_gain, 4),
        "overall_gain_vs_baseline": round(overall_vbase, 4),
        "positive_images_vs_gate":     pos_total,
        "positive_images_vs_baseline": pos_vbase_total,
        "total_images":       total,
        "mean_verified":      round(float(mean_ver), 2),
    }, f, indent=2)

print(f"\n✅ Saved: table_block24c_seeded.csv")
print()
print("=" * 65)
print("BLOCK 24c COMPLETE")
print("=" * 65)
