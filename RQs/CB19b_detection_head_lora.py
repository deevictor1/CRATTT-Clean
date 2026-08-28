# ============================================================
# BLOCK 19b: Detection Head LoRA — Higher LR + More Steps
# Tests whether lr=5e-4 and 20 steps generalises Block 19's
# positive signal to other corruption types.
#
# NOTE: this run's positive effect was concentrated in
# GAUSSIAN_NOISE (snow gave exactly 0.0000 here). The test
# below checks whether more lr/steps produces a broader,
# less corruption-specific gain.
# ============================================================

import torch
import numpy as np
import pandas as pd
import json
import os
from imagecorruptions import corrupt as ic_corrupt
from tqdm.notebook import tqdm
from IPython.display import display

print("="*50)
print("BLOCK 19b: DETECTION HEAD LoRA — HIGHER LR")
print("="*50)

# --- Pre-flight: confirm detection-head LoRA state survived ---
# Block 19 injected 23 layers / 155,872 trainable params in this
# session. Nothing between Block 19 and here should have touched
# bbox_embed/reference_points_head, so this should still hold.
_current_trainable = sum(
    p.numel() for p in dino_model.parameters() if p.requires_grad
)
if _current_trainable != 155_872:
    print(f"⚠️  Expected 155,872 trainable params (Block 19's "
          f"detection-head LoRA), found {_current_trainable:,}. "
          f"Re-run Block 19 before continuing.")
else:
    print(f"✅ Detection-head LoRA state confirmed "
          f"({_current_trainable:,} trainable params)")

TTT_STEPS_19B = 20
TTT_LR_19B    = 5e-4  # 10x higher than Block 19

print(f"TTT steps : {TTT_STEPS_19B} (was 10)")
print(f"TTT lr    : {TTT_LR_19B} (was 5e-5)")
print(f"All other settings identical to Block 19")
print()

EVAL_CORRUPTIONS_19B = [
    ("Noise",   "gaussian_noise", 5),
    ("Blur",    "motion_blur",    5),
    ("Weather", "snow",           5),
    ("Digital", "contrast",       5),
]
PILOT_IMAGES_19B = image_files[:5]

all_rows_19b = []

for cat_name, corruption, severity in EVAL_CORRUPTIONS_19B:
    print(f"\n▶  {cat_name}: {corruption} sev{severity}")

    # Reset LoRA B to zero
    for name, param in dino_model.named_parameters():
        if 'lora_B' in name:
            param.data.zero_()

    # Fresh optimizer with higher lr
    optimizer_19b = torch.optim.AdamW(
        [p for p in dino_model.parameters() if p.requires_grad],
        lr=TTT_LR_19B,
        weight_decay=1e-4
    )

    corruption_rows = []
    pbar = tqdm(
        PILOT_IMAGES_19B, desc=f"  {corruption}", leave=False
    )

    for img_path in pbar:
        img_id  = img_id_map[os.path.basename(img_path)]
        raw_img = loaded_images[img_path]
        fname   = os.path.basename(img_path)

        c_img = ic_corrupt(
            raw_img,
            corruption_name=corruption,
            severity=severity
        )

        # Baseline
        with torch.no_grad():
            inputs = dino_processor(
                images=c_img, text=DINO_TEXT_PROMPT,
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
        bmap, _ = compute_map(baseline_preds, coco_gt, [img_id])

        # CRATTT only
        with torch.no_grad():
            crattt_preds_b, stats_b = run_crattt_inference(c_img)
        for p in crattt_preds_b:
            p["image_id"] = img_id
        crattt_coco_b = [{
            "image_id": p["image_id"],
            "category_id": p["category_id"],
            "bbox": p["bbox"], "score": p["score"]
        } for p in crattt_preds_b]
        cmap_b, _ = compute_map(crattt_coco_b, coco_gt, [img_id])

        # Detection Head TTT — 20 steps at higher lr
        loss_values = []
        dino_model.train()
        for step in range(TTT_STEPS_19B):
            optimizer_19b.zero_grad()
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
                optimizer_19b.step()
                loss_values.append(round(loss.item(), 6))
        dino_model.eval()

        # Re-run CRATTT
        with torch.no_grad():
            crattt_preds_c, stats_c = run_crattt_inference(c_img)
        for p in crattt_preds_c:
            p["image_id"] = img_id
        crattt_coco_c = [{
            "image_id": p["image_id"],
            "category_id": p["category_id"],
            "bbox": p["bbox"], "score": p["score"]
        } for p in crattt_preds_c]
        cmap_c, _ = compute_map(crattt_coco_c, coco_gt, [img_id])

        # Box shift
        with torch.no_grad():
            inputs_c2 = dino_processor(
                images=c_img, text=DINO_TEXT_PROMPT,
                return_tensors="pt"
            ).to(device)
            out_c2 = dino_model(**inputs_c2)
        res_c2 = dino_processor\
            .post_process_grounded_object_detection(
                out_c2, inputs_c2.input_ids,
                target_sizes=[c_img.shape[:2]],
                text_threshold=CRATTT_PARAMS["dino_text_thr"]
            )[0]
        box_shift = 0.0
        if len(baseline_res['boxes']) > 0 and \
           len(res_c2['boxes']) > 0:
            min_l = min(len(baseline_res['boxes']),
                        len(res_c2['boxes']))
            box_shift = (
                baseline_res['boxes'][:min_l] -
                res_c2['boxes'][:min_l]
            ).abs().mean().item()

        row = {
            "category":            cat_name,
            "corruption":          corruption,
            "baseline_mAP":        round(float(bmap),   4),
            "crattt_mAP":          round(float(cmap_b), 4),
            "dethead_ttt_mAP":     round(float(cmap_c), 4),
            "delta_ttt_vs_crattt": round(float(cmap_c-cmap_b), 4),
            "mean_box_shift_px":   round(float(box_shift), 4),
            "loss_first_5":        loss_values[:5]
        }
        corruption_rows.append(row)
        pbar.set_postfix({
            "B": f"{bmap:.3f}", "C": f"{cmap_b:.3f}",
            "T": f"{cmap_c:.3f}", "Δ": f"{box_shift:.1f}px"
        })

    all_rows_19b.extend(corruption_rows)
    c_df = pd.DataFrame(corruption_rows)
    print(f"  Baseline mAP       : {c_df['baseline_mAP'].mean():.4f}")
    print(f"  CRATTT mAP         : {c_df['crattt_mAP'].mean():.4f}")
    print(f"  DetHead+TTT mAP    : {c_df['dethead_ttt_mAP'].mean():.4f}")
    print(f"  TTT gain vs CRATTT : "
          f"{c_df['delta_ttt_vs_crattt'].mean():+.4f}")
    print(f"  Mean box shift (px): "
          f"{c_df['mean_box_shift_px'].mean():.4f}")
    sample = next((r['loss_first_5'] for r in corruption_rows
                   if r['loss_first_5']), [])
    if sample:
        print(f"  Loss (first 5)     : {sample}")

df_19b = pd.DataFrame(all_rows_19b)
overall_gain_19b   = df_19b['delta_ttt_vs_crattt'].mean()
mean_box_shift_19b = df_19b['mean_box_shift_px'].mean()

print(f"\n{'='*60}")
print("BLOCK 19b SUMMARY")
print(f"{'='*60}")
print(f"Overall TTT gain vs CRATTT: {overall_gain_19b:+.4f}")
print(f"Mean box shift (pixels)   : {mean_box_shift_19b:.4f}")
print()

# FIXED: reads Block 19's live result from df_19 (or its saved
# CSV) instead of a hardcoded placeholder value.
if 'df_19' in globals():
    gain_19      = df_19['delta_ttt_vs_crattt'].mean()
    boxshift_19  = df_19['mean_box_shift_px'].mean()
    src_19       = "live"
else:
    _csv19 = os.path.join(EVAL_PARAMS["table_dir"], "table_block19_dethead_lora.csv")
    if os.path.exists(_csv19):
        _df19 = pd.read_csv(_csv19)
        gain_19      = _df19['delta_ttt_vs_crattt'].mean()
        boxshift_19  = _df19['mean_box_shift_px'].mean()
        src_19       = "from disk"
    else:
        gain_19, boxshift_19, src_19 = None, None, "unavailable"

print(f"{'Configuration':<35} {'TTT gain':>10} {'Box shift':>10}  {'source'}")
print("-"*68)
if gain_19 is not None:
    print(f"{'Blk 19 (rank=8, 10 steps, lr=5e-5)':<35} "
          f"{gain_19:>+10.4f} {boxshift_19:>9.2f}px  {src_19}")
else:
    print(f"{'Blk 19 (rank=8, 10 steps, lr=5e-5)':<35} "
          f"{'N/A':>10} {'N/A':>10}  re-run Block 19 first")
print(f"{'Blk 19b(rank=8, 20 steps, lr=5e-4)':<35} "
      f"{overall_gain_19b:>+10.4f} "
      f"{mean_box_shift_19b:>9.2f}px  live")

if overall_gain_19b > 0.01:
    print("\n✅ MEANINGFUL POSITIVE GAIN")
    print("   Detection head LoRA generalises beyond a single corruption.")
    print("   Proceed to Vast.ai for full-scale evaluation.")
    conclusion = "proceed_vastai"
elif overall_gain_19b > 0.003:
    print("\n✅ CONSISTENT MARGINAL GAIN")
    print("   Improvement is real but modest.")
    print("   Proceed to Vast.ai — full scale may show more.")
    conclusion = "proceed_vastai_cautious"
else:
    print("\n⚠️  GAIN LIMITED TO ONE CORRUPTION TYPE")
    print("   Detection head TTT is corruption-type specific.")
    print("   (This run: gaussian_noise. Not a broad effect.)")
    conclusion = "corruption_specific_finding"

# Save
csv_19b = os.path.join(
    EVAL_PARAMS["table_dir"], "table_block19b_higher_lr.csv"
)
df_19b.to_csv(csv_19b, index=False)
with open(os.path.join(
    EVAL_PARAMS["save_dir"], "block19b_results.json"
), "w") as f:
    json.dump({
        "lr": TTT_LR_19B, "steps": TTT_STEPS_19B,
        "overall_gain": round(float(overall_gain_19b), 4),
        "mean_box_shift": round(float(mean_box_shift_19b), 4),
        "block_19_gain": round(float(gain_19), 4) if gain_19 is not None else None,
        "conclusion": conclusion
    }, f, indent=2)

print(f"\n✅ Saved: {csv_19b}")
print("\n" + "="*50)
print("BLOCK 19b COMPLETE")
print("="*50)
