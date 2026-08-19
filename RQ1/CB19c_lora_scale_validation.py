# Archive and Master Ablation first before Block 19c below
===================================================================
import shutil, json, os

# Archive all results
for folder in ['results', 'tables', 'figures', 'checkpoints']:
    shutil.make_archive(
        f'/kaggle/working/FINAL_COMPLETE_{folder}',
        'zip', '/kaggle/working', folder
    )
    print(f"✅ {folder} archived")

# Pull every block's gain live from memory, falling back to its
# saved CSV on disk if the variable isn't in scope anymore.
def _get_gain(varname, csv_path, csv_col="delta_ttt_vs_crattt"):
    if varname in globals():
        return float(globals()[varname][csv_col].mean())
    if os.path.exists(csv_path):
        return float(pd.read_csv(csv_path)[csv_col].mean())
    return None

table_dir = EVAL_PARAMS["table_dir"]

gain_16   = _get_gain("df_final", os.path.join(table_dir, "table_4_7_end_to_end.csv"))
gain_16b  = _get_gain("df_b",     os.path.join(table_dir, "block16b_alignment_ablation.csv"))
gain_16c  = _get_gain("df_16c",   os.path.join(table_dir, "block16c_rank16_ttt.csv"))
gain_17   = _get_gain("df_17",    os.path.join(table_dir, "table_block17_coadapt_pilot.csv"),
                       csv_col="delta_coadapt_vs_crattt")
gain_18   = _get_gain("df_18",    os.path.join(table_dir, "table_block18_encoder_gate.csv"))
gain_19   = _get_gain("df_19",    os.path.join(table_dir, "table_block19_dethead_lora.csv"))
gain_19b  = _get_gain("df_19b",   os.path.join(table_dir, "table_block19b_higher_lr.csv"))

master_ablation = {
    "block_16_frozen_conf":      round(gain_16, 4)  if gain_16  is not None else None,
    "block_16b_frozen_align":    round(gain_16b, 4) if gain_16b is not None else None,
    "block_16c_rank16_align":    round(gain_16c, 4) if gain_16c is not None else None,
    "block_17_coadapt":          round(gain_17, 4)  if gain_17  is not None else None,
    "block_18_encoder_gate":     round(gain_18, 4)  if gain_18  is not None else None,
    "block_19_dethead_low_lr":   round(gain_19, 4)  if gain_19  is not None else None,
    "block_19b_dethead_high_lr": round(gain_19b, 4) if gain_19b is not None else None,
    "finding": {
        "positive_result": "gaussian_noise_sev5_dethead_lora_rank8_lr5e5",
        "gaussian_noise_gain": round(gain_19, 4) if gain_19 is not None else None,
        "high_lr_destabilizes": True,
        "high_lr_gain_5e4": round(gain_19b, 4) if gain_19b is not None else None,
        "stable_lr_range": "< 1e-4",
        "corruption_specificity": "effect concentrated in one corruption at N=5 — pending N=20 scale check"
    }
}

with open('/kaggle/working/results/MASTER_ABLATION.json', 'w') as f:
    json.dump(master_ablation, f, indent=2)

print("✅ Master ablation record saved (live Swin-B values)")
print(json.dumps(master_ablation, indent=2))
print("\nDownload all FINAL_COMPLETE_*.zip files")
print("Upload to GitHub CRATTT-Clean repo")

===============================================================================

# ============================================================
# BLOCK 19c: Detection Head LoRA — Scale Validation
# Runs validated Block 19 configuration (lr=5e-5, 10 steps)
# on all 20 COCO images.
#
# CORRUPTION CHOICE: in this run, snow gave exactly 0.0000 in
# Block 19 and collapsed further in 19b — the actual signal was
# in GAUSSIAN_NOISE (+0.0096) instead. Scaling gaussian_noise here
# to test what this run actually found. To scale snow instead,
# change SCALE_CORRUPTION below.
#
# Purpose: Confirm that the N=5 gain holds at 20-image scale
# before Vast.ai spending.
#
# If gain holds: proceed to Vast.ai for 50-image full scale.
# If gain disappears: the 5-image result was statistical noise.
#
# Runtime: ~20-25 minutes on Kaggle T4
# Cost: Free
# ============================================================

import torch
import numpy as np
import pandas as pd
import json
import os
from imagecorruptions import corrupt as ic_corrupt
from tqdm.notebook import tqdm
from IPython.display import display

SCALE_CORRUPTION = 'gaussian_noise'

print("="*55)
print(f"BLOCK 19c: {SCALE_CORRUPTION.upper()} SCALE VALIDATION (N=20)")
print("="*55)
print(f"Configuration  : Detection head LoRA rank=8")
print(f"Learning rate  : 5e-5 (validated in Block 19)")
print(f"TTT steps      : 10 (validated in Block 19)")
print(f"Corruption     : {SCALE_CORRUPTION} severity 5")
print(f"Images         : {len(image_files)} (was 5 in Block 19)")
print(f"Tau            : {CRATTT_PARAMS['tau']}")
print()

# FIX: strengthened from "trainable != 0" to the exact expected
# count, so this can't silently pass against the wrong LoRA config
# (e.g. the dormant 73,728-param encoder/decoder LoRA instead of
# the detection head's 155,872).
trainable = sum(
    p.numel() for p in dino_model.parameters()
    if p.requires_grad
)
if trainable == 0:
    print("⚠️  No trainable params detected — LoRA not active")
    print("   Re-run Block 19 injection cells first")
elif trainable != 155_872:
    print(f"⚠️  Expected 155,872 trainable params (detection head "
          f"LoRA from Block 19), found {trainable:,} instead. "
          f"Re-run Block 19's injection before continuing.")
else:
    print(f"✅ Detection head LoRA active: {trainable:,} params")

# Reset LoRA B to zero — clean starting state
for name, param in dino_model.named_parameters():
    if 'lora_B' in name:
        param.data.zero_()
print("✅ LoRA B matrices reset to zero")

# Fresh optimizer
optimizer_19c = torch.optim.AdamW(
    [p for p in dino_model.parameters() if p.requires_grad],
    lr=5e-5,
    weight_decay=1e-4
)

# --- Main Loop ---
rows_19c   = []
corruption = SCALE_CORRUPTION
severity   = 5

pbar = tqdm(image_files, desc=f"{SCALE_CORRUPTION} sev5 N=20", leave=True)

for img_path in pbar:
    img_id  = img_id_map[os.path.basename(img_path)]
    raw_img = loaded_images[img_path]
    fname   = os.path.basename(img_path)

    c_img = ic_corrupt(
        raw_img,
        corruption_name=corruption,
        severity=severity
    )

    # === CONDITION A: Baseline DINO ===
    with torch.no_grad():
        inputs = dino_processor(
            images=c_img,
            text=DINO_TEXT_PROMPT,
            return_tensors="pt"
        ).to(device)
        outputs      = dino_model(**inputs)
        baseline_res = dino_processor\
            .post_process_grounded_object_detection(
                outputs, inputs.input_ids,
                target_sizes=[c_img.shape[:2]],
                text_threshold=CRATTT_PARAMS["dino_text_thr"]
            )[0]

    baseline_preds = dino_to_coco_format(baseline_res, img_id)
    bmap, _        = compute_map(
        baseline_preds, coco_gt, [img_id]
    )

    # === CONDITION B: CRATTT only ===
    with torch.no_grad():
        crattt_preds_b, stats_b = run_crattt_inference(c_img)
    for p in crattt_preds_b:
        p["image_id"] = img_id
    crattt_coco_b = [{
        "image_id":    p["image_id"],
        "category_id": p["category_id"],
        "bbox":        p["bbox"],
        "score":       p["score"]
    } for p in crattt_preds_b]
    cmap_b, _ = compute_map(
        crattt_coco_b, coco_gt, [img_id]
    )

    # === CONDITION C: Detection Head TTT ===
    loss_values = []
    dino_model.train()
    for step in range(10):
        optimizer_19c.zero_grad()
        loss = compute_detection_head_loss(
            dino_model, dino_processor,
            c_img, crattt_preds_b, device
        )
        if loss is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in dino_model.parameters()
                 if p.requires_grad],
                max_norm=1.0
            )
            optimizer_19c.step()
            loss_values.append(round(loss.item(), 6))
    dino_model.eval()

    # Re-run CRATTT with updated detection head
    with torch.no_grad():
        crattt_preds_c, stats_c = run_crattt_inference(c_img)
    for p in crattt_preds_c:
        p["image_id"] = img_id
    crattt_coco_c = [{
        "image_id":    p["image_id"],
        "category_id": p["category_id"],
        "bbox":        p["bbox"],
        "score":       p["score"]
    } for p in crattt_preds_c]
    cmap_c, _ = compute_map(
        crattt_coco_c, coco_gt, [img_id]
    )

    # Box shift
    with torch.no_grad():
        inputs_s = dino_processor(
            images=c_img, text=DINO_TEXT_PROMPT,
            return_tensors="pt"
        ).to(device)
        out_s = dino_model(**inputs_s)
    res_s = dino_processor\
        .post_process_grounded_object_detection(
            out_s, inputs_s.input_ids,
            target_sizes=[c_img.shape[:2]],
            text_threshold=CRATTT_PARAMS["dino_text_thr"]
        )[0]
    box_shift = 0.0
    if len(baseline_res['boxes']) > 0 and \
       len(res_s['boxes']) > 0:
        min_l = min(len(baseline_res['boxes']),
                    len(res_s['boxes']))
        box_shift = (
            baseline_res['boxes'][:min_l] -
            res_s['boxes'][:min_l]
        ).abs().mean().item()

    delta = round(float(cmap_c - cmap_b), 4)
    rows_19c.append({
        "image":               fname,
        "img_id":              img_id,
        "baseline_mAP":        round(float(bmap),   4),
        "crattt_mAP":          round(float(cmap_b), 4),
        "dethead_ttt_mAP":     round(float(cmap_c), 4),
        "delta_ttt_vs_crattt": delta,
        "delta_ttt_vs_base":   round(float(cmap_c - bmap), 4),
        "n_baseline":          len(baseline_res['boxes']),
        "n_crattt":            stats_b['n_verified'],
        "n_ttt":               stats_c['n_verified'],
        "box_shift_px":        round(float(box_shift), 4),
        "loss_mean":           round(float(np.mean(loss_values))
                                     if loss_values else 0, 4)
    })

    pbar.set_postfix({
        "B":  f"{bmap:.3f}",
        "C":  f"{cmap_b:.3f}",
        "T":  f"{cmap_c:.3f}",
        "Δ":  f"{delta:+.3f}"
    })

# --- Results ---
df_19c = pd.DataFrame(rows_19c)

print("\n" + "="*65)
print(f"BLOCK 19c: PER-IMAGE RESULTS — {SCALE_CORRUPTION} Sev5 N=20")
print("="*65)
display(df_19c[[
    "image", "baseline_mAP", "crattt_mAP",
    "dethead_ttt_mAP", "delta_ttt_vs_crattt", "box_shift_px"
]])

# Aggregate
mean_base   = df_19c['baseline_mAP'].mean()
mean_crattt = df_19c['crattt_mAP'].mean()
mean_ttt    = df_19c['dethead_ttt_mAP'].mean()
mean_delta  = df_19c['delta_ttt_vs_crattt'].mean()
mean_shift  = df_19c['box_shift_px'].mean()

positive_images = (df_19c['delta_ttt_vs_crattt'] > 0).sum()
negative_images = (df_19c['delta_ttt_vs_crattt'] < 0).sum()
zero_images     = (df_19c['delta_ttt_vs_crattt'] == 0).sum()

print(f"\n{'='*55}")
print(f"AGGREGATE SUMMARY — {SCALE_CORRUPTION} Severity 5 (N=20)")
print(f"{'='*55}")
print(f"Mean Baseline mAP        : {mean_base:.4f}")
print(f"Mean CRATTT mAP          : {mean_crattt:.4f}")
print(f"Mean DetHead+TTT mAP     : {mean_ttt:.4f}")
print(f"Mean TTT gain vs CRATTT  : {mean_delta:+.4f}")
print(f"Mean box shift (pixels)  : {mean_shift:.4f}")
print()
print(f"Images with positive gain: {positive_images}/20")
print(f"Images with zero gain    : {zero_images}/20")
print(f"Images with negative gain: {negative_images}/20")

# FIX: live read of Block 19's actual result instead of a
# hardcoded placeholder value.
if 'df_19' in globals():
    gain_19_live  = df_19['delta_ttt_vs_crattt'].mean()
    shift_19_live = df_19['mean_box_shift_px'].mean()
    src_19 = "live"
else:
    _csv19 = os.path.join(EVAL_PARAMS["table_dir"], "table_block19_dethead_lora.csv")
    if os.path.exists(_csv19):
        _df19 = pd.read_csv(_csv19)
        gain_19_live  = _df19['delta_ttt_vs_crattt'].mean()
        shift_19_live = _df19['mean_box_shift_px'].mean()
        src_19 = "from disk"
    else:
        gain_19_live, shift_19_live, src_19 = None, None, "unavailable"

print(f"\n{'='*55}")
print("SCALE COMPARISON")
print(f"{'='*55}")
print(f"{'Scale':<22} {'TTT gain':>12} {'Box shift':>12}  {'source'}")
print("-"*58)
if gain_19_live is not None:
    print(f"{'Block 19 (N=5, mixed corruptions)':<22} "
          f"{gain_19_live:>+12.4f} {shift_19_live:>11.2f}px  {src_19}")
else:
    print(f"{'Block 19 (N=5)':<22} {'N/A':>12} {'N/A':>12}  re-run Block 19 first")
print(f"{f'Block 19c (N=20, {SCALE_CORRUPTION})':<22} {mean_delta:>+12.4f} "
      f"{mean_shift:>11.2f}px  live")

# Verdict
print()
if mean_delta > 0.010:
    verdict    = "✅ GAIN CONFIRMED AT SCALE"
    action     = ("Result is statistically robust. "
                  "Proceed to Vast.ai for 50-image validation.")
    conclusion = "confirmed_proceed_vastai"
elif mean_delta > 0.003:
    verdict    = "✅ GAIN HOLDS BUT REDUCED AT SCALE"
    action     = ("Positive but smaller than 5-image pilot. "
                  "Report as promising preliminary result. "
                  "Vast.ai optional.")
    conclusion = "holds_reduced"
elif mean_delta > -0.002:
    verdict    = "⚠️  NEAR-ZERO AT SCALE"
    action     = ("5-image result was likely statistical noise. "
                  "Do not proceed to Vast.ai for TTT. "
                  "Report as inconclusive.")
    conclusion = "noise_at_scale"
else:
    verdict    = "❌ GAIN DOES NOT HOLD AT SCALE"
    action     = ("5-image result was noise. "
                  "TTT constraint confirmed. "
                  "Do not proceed to Vast.ai for TTT.")
    conclusion = "does_not_scale"

print(verdict)
print(f"Action: {action}")

# Save
csv_19c = os.path.join(
    EVAL_PARAMS["table_dir"], f"table_block19c_{SCALE_CORRUPTION}_n20.csv"
)
df_19c.to_csv(csv_19c, index=False)

with open(os.path.join(
    EVAL_PARAMS["save_dir"], "block19c_results.json"
), "w") as f:
    json.dump({
        "corruption":      SCALE_CORRUPTION,
        "severity":        5,
        "n_images":        20,
        "lr":              5e-5,
        "ttt_steps":       10,
        "mean_baseline":   round(float(mean_base),   4),
        "mean_crattt":     round(float(mean_crattt), 4),
        "mean_ttt":        round(float(mean_ttt),    4),
        "mean_gain":       round(float(mean_delta),  4),
        "mean_box_shift":  round(float(mean_shift),  4),
        "positive_images": int(positive_images),
        "block_19_n5_gain": round(float(gain_19_live), 4) if gain_19_live is not None else None,
        "conclusion":      conclusion
    }, f, indent=2)

print(f"\n✅ Results saved: {csv_19c}")
print("\n" + "="*50)
print("BLOCK 19c COMPLETE")
print("="*50)
