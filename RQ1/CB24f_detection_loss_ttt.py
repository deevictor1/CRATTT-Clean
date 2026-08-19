# ============================================================
# BLOCK 24f: Detection Loss TTT — Diagnostic at N=10
# Dada Victor Damilare | MRES7015 | University of Greater Manchester
# ============================================================
#
# WHY THIS BLOCK EXISTS
# ─────────────────────
# Block 24e (N=200, formally powered) showed the confidence-only
# TTT loss produces a small but statistically robust NEGATIVE
# vs_baseline (mean=-0.0055, p<0.0001). Root cause diagnosed:
# the confidence-only loss has no box regression or class
# supervision — it pushes confidence up on gate-verified queries
# without ever teaching the model where the correct box is or
# what class it should predict. This means the adaptation signal
# is not grounded in correctness, only in confidence magnitude.
#
# THIS BLOCK tests a fundamentally different loss that uses the
# gate-verified detections as actual pseudo-ground-truth:
#
#   Loss = λ1 * L1(pred_cxcywh, gt_cxcywh)       [box regression]
#        + λ2 * (1 - GIoU(pred_xyxy, gt_xyxy))    [spatial overlap]
#        + λ3 * (-mean(scores_matched))            [classification]
#
# The richer supervision gives the model something substantively
# different to learn from — not "be confident here" but "predict
# this specific box at this specific location." This directly
# addresses the identified root limitation, within the same TTT
# framework, without changing any other part of CRATTT.
#
# DESIGN: Variant A (confidence-only, control) and Variant B
# (detection loss, new) are run on the SAME images with the SAME
# gate decisions — so any difference in vs_baseline is purely
# attributable to the loss function change, nothing else.
#
# Starts at N=10, 2 corruptions, severity 5 for a quick first
# read before committing to a larger run.
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import random as _py_random

from imagecorruptions import corrupt as ic_corrupt
from torchvision.ops import box_iou, generalized_box_iou
from tqdm.notebook import tqdm

print("=" * 65)
print("BLOCK 24f: Detection Loss TTT — Diagnostic (N=10)")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────
# Pre-flight check
# ─────────────────────────────────────────────────────────────
required_24f = ["b24_run_dino", "b24_compute_snr", "b24_apply_gate",
                "b24_confidence_loss", "B24",
                "SNR_BASELINE_24c", "TAU_INTERNAL_24c",
                "dino_model", "dino_processor", "DINO_TEXT_PROMPT",
                "device", "compute_map", "coco_gt", "COCO_MAP",
                "image_files", "loaded_images", "img_id_map"]

missing_24f = [f for f in required_24f if f not in globals()]
if missing_24f:
    print(f"❌ Missing before Block 24f: {missing_24f}")
else:
    print("✅ All Block 24f dependencies present, safe to run")

import inspect
_src = inspect.getsource(b24_run_dino)
if "post_process_grounded_object_detection" in _src:
    print("❌ WARNING: b24_run_dino still calling library post-processing — patch not active!")
elif "BOX_THRESHOLD" in _src:
    print("✅ Confirmed: b24_run_dino is the patched clean-argmax version")
else:
    print("⚠️  Could not confirm patch status — inspect manually")

n_lora = sum(1 for n, p in dino_model.named_parameters() if "lora_B" in n)
print(f"LoRA B matrices currently in model: {n_lora}  (rank=4 pass expects 36)")

print(f"τ_internal (24c) : {TAU_INTERNAL_24c:.4f}")
print(f"SNR_BASELINE (24c): {SNR_BASELINE_24c:.3f}")
print()

# ─────────────────────────────────────────────────────────────
# Seeding + reset helpers
# ─────────────────────────────────────────────────────────────
CORRUPTION_SEED_OFFSET = {
    "gaussian_noise": 1_000, "motion_blur": 2_000,
    "snow": 3_000, "contrast": 4_000,
}

def seed_for_corruption(img_id, corruption, severity):
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

reset_lora_b()
print("✅ Dependencies present, LoRA reset to clean state")
print()

# ─────────────────────────────────────────────────────────────
# 24f.0  NEW DETECTION LOSS
#
# verified_boxes : list of tensors, each [4] in xyxy PIXEL coords
#                  (same format as b24_apply_gate returns)
# verified_labels: list of str (not used for loss directly but
#                  available if you later add per-class weighting)
# weights        : (box_w, giou_w, cls_w) — tunable per ablation
# ─────────────────────────────────────────────────────────────
def b24f_detection_loss(image_np, verified_boxes, device,
                         box_w=2.0, giou_w=1.0, cls_w=1.0):
    """
    Full detection loss using verified detections as pseudo-GT.

    Three loss terms:
      L1   — penalises coordinate distance in normalised cxcywh space
      GIoU — penalises poor spatial overlap (geometry-aware)
      Cls  — pushes confidence up for matched queries (same as before,
             kept to preserve the original signal alongside the new ones)
    """
    if len(verified_boxes) == 0:
        return None

    img_h, img_w = image_np.shape[:2]

    inputs = dino_processor(
        images=image_np,
        text=DINO_TEXT_PROMPT,
        return_tensors="pt"
    ).to(device)
    outputs = dino_model(**inputs)

    # ── Predicted boxes ──────────────────────────────────────
    pred_cxcywh = outputs.pred_boxes[0]

    cx = pred_cxcywh[:, 0] * img_w
    cy = pred_cxcywh[:, 1] * img_h
    pw = pred_cxcywh[:, 2] * img_w
    ph = pred_cxcywh[:, 3] * img_h
    pred_xyxy_px = torch.stack(
        [cx - pw/2, cy - ph/2, cx + pw/2, cy + ph/2], dim=-1
    )

    # ── Verified boxes ────────────────────────────────────────
    vb_px = torch.stack(verified_boxes).to(device)

    vb_cx_n = (vb_px[:, 0] + vb_px[:, 2]) / 2 / img_w
    vb_cy_n = (vb_px[:, 1] + vb_px[:, 3]) / 2 / img_h
    vb_w_n  = (vb_px[:, 2] - vb_px[:, 0]) / img_w
    vb_h_n  = (vb_px[:, 3] - vb_px[:, 1]) / img_h
    vb_cxcywh_n = torch.stack([vb_cx_n, vb_cy_n, vb_w_n, vb_h_n], dim=-1)

    # ── Matching: each verified box → nearest predicted query ──
    with torch.no_grad():
        iou_mat = box_iou(vb_px, pred_xyxy_px.detach())
    best_iou_vals, best_q_ids = iou_mat.max(dim=1)
    valid_mask = best_iou_vals > 0.25

    if valid_mask.sum() == 0:
        return None

    matched_q_ids   = best_q_ids[valid_mask]
    matched_pred_cxcywh = pred_cxcywh[matched_q_ids]
    matched_vb_cxcywh   = vb_cxcywh_n[valid_mask]
    matched_pred_xyxy   = pred_xyxy_px[matched_q_ids]
    matched_vb_xyxy     = vb_px[valid_mask]

    l1_loss = F.l1_loss(matched_pred_cxcywh, matched_vb_cxcywh,
                         reduction="mean")

    scale = torch.tensor([img_w, img_h, img_w, img_h],
                          dtype=torch.float32, device=device)
    pred_xyxy_n = matched_pred_xyxy / scale
    vb_xyxy_n   = matched_vb_xyxy   / scale

    pred_xyxy_n = pred_xyxy_n.clamp(0.0, 1.0)
    vb_xyxy_n   = vb_xyxy_n.clamp(0.0, 1.0)

    giou_mat  = generalized_box_iou(pred_xyxy_n, vb_xyxy_n)
    giou_loss = (1.0 - giou_mat.diag()).mean()

    scores_raw = outputs.logits.sigmoid().max(dim=-1).values[0]
    scores_matched = scores_raw[matched_q_ids]
    cls_loss = -scores_matched.mean()

    total = box_w * l1_loss + giou_w * giou_loss + cls_w * cls_loss
    return total


print("✅ b24f_detection_loss defined")
print("   Terms: L1 (box_w=2.0)  +  GIoU (giou_w=1.0)  +  Cls (cls_w=1.0)")
print()

# ─────────────────────────────────────────────────────────────
# 24f.1  Configuration
# ─────────────────────────────────────────────────────────────
DIAG_CORRUPTIONS = ["motion_blur", "contrast"]
DIAG_SEVERITY    = 5
DIAG_N           = 10
TTT_STEPS        = 5
TTT_LR           = 1e-4

print(f"Corruptions : {DIAG_CORRUPTIONS}")
print(f"Severity    : {DIAG_SEVERITY}")
print(f"N images    : {DIAG_N}")
print(f"Steps / lr  : {TTT_STEPS} / {TTT_LR}")
print()

# ─────────────────────────────────────────────────────────────
# 24f.2  Main loop
# Same images, same gate — only loss function differs.
# Variant A = confidence-only (control, matches Block 24c)
# Variant B = detection loss  (new)
# ─────────────────────────────────────────────────────────────
results = []

for corruption in DIAG_CORRUPTIONS:
    for img_path in tqdm(image_files[:DIAG_N],
                          desc=f"{corruption} sev{DIAG_SEVERITY}"):

        img_id  = img_id_map[os.path.basename(img_path)]
        raw_img = loaded_images[img_path]
        c_img   = apply_corruption_deterministic(
            raw_img, img_id, corruption, DIAG_SEVERITY
        )

        reset_lora_b()
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

        dets = b24_compute_snr(c_img)
        v_boxes, v_labels, v_rjoints = b24_apply_gate(dets)

        if len(v_boxes) == 0:
            for variant in ["confidence_only", "detection_loss"]:
                results.append({
                    "corruption": corruption, "img_id": img_id,
                    "variant": variant,
                    "mAP_baseline": mAP_base, "mAP_ttt": mAP_base,
                    "vs_baseline": 0.0, "beats_baseline": False,
                    "ttt_fired": False, "n_verified": 0,
                })
            continue

        reset_lora_b()
        opt_a = torch.optim.AdamW(
            [p for p in dino_model.parameters() if p.requires_grad],
            lr=TTT_LR, weight_decay=1e-4,
        )
        dino_model.train()
        fired_a = False
        for _ in range(TTT_STEPS):
            opt_a.zero_grad()
            loss_a = b24_confidence_loss(c_img, v_boxes, device)
            if loss_a is not None and loss_a.requires_grad:
                loss_a.backward()
                nn.utils.clip_grad_norm_(
                    [p for p in dino_model.parameters() if p.requires_grad],
                    max_norm=B24["max_norm"])
                opt_a.step()
                fired_a = True
        dino_model.eval()

        if fired_a:
            boxes_t, scores_t, labels_t = b24_run_dino(c_img)
            preds_a = [
                {"image_id": img_id, "category_id": COCO_MAP[l],
                 "bbox": [b[0].item(), b[1].item(),
                          (b[2]-b[0]).item(), (b[3]-b[1]).item()],
                 "score": s.item()}
                for b, s, l in zip(boxes_t, scores_t, labels_t)
                if l in COCO_MAP
            ]
            mAP_a, _ = compute_map(preds_a, coco_gt, [img_id])
        else:
            mAP_a = mAP_base

        del opt_a
        results.append({
            "corruption": corruption, "img_id": img_id,
            "variant": "confidence_only",
            "mAP_baseline": mAP_base, "mAP_ttt": mAP_a,
            "vs_baseline": mAP_a - mAP_base,
            "beats_baseline": mAP_a > mAP_base,
            "ttt_fired": fired_a, "n_verified": len(v_boxes),
        })

        reset_lora_b()
        opt_b = torch.optim.AdamW(
            [p for p in dino_model.parameters() if p.requires_grad],
            lr=TTT_LR, weight_decay=1e-4,
        )
        dino_model.train()
        fired_b = False
        for _ in range(TTT_STEPS):
            opt_b.zero_grad()
            loss_b = b24f_detection_loss(c_img, v_boxes, device)
            if loss_b is not None and loss_b.requires_grad:
                loss_b.backward()
                nn.utils.clip_grad_norm_(
                    [p for p in dino_model.parameters() if p.requires_grad],
                    max_norm=B24["max_norm"])
                opt_b.step()
                fired_b = True
        dino_model.eval()

        if fired_b:
            boxes_t, scores_t, labels_t = b24_run_dino(c_img)
            preds_b = [
                {"image_id": img_id, "category_id": COCO_MAP[l],
                 "bbox": [b[0].item(), b[1].item(),
                          (b[2]-b[0]).item(), (b[3]-b[1]).item()],
                 "score": s.item()}
                for b, s, l in zip(boxes_t, scores_t, labels_t)
                if l in COCO_MAP
            ]
            mAP_b, _ = compute_map(preds_b, coco_gt, [img_id])
        else:
            mAP_b = mAP_base

        del opt_b
        results.append({
            "corruption": corruption, "img_id": img_id,
            "variant": "detection_loss",
            "mAP_baseline": mAP_base, "mAP_ttt": mAP_b,
            "vs_baseline": mAP_b - mAP_base,
            "beats_baseline": mAP_b > mAP_base,
            "ttt_fired": fired_b, "n_verified": len(v_boxes),
        })

# ─────────────────────────────────────────────────────────────
# 24f.3  Results
# ─────────────────────────────────────────────────────────────
df_24f = pd.DataFrame(results)

print()
print("=" * 65)
print("RESULTS — Detection Loss vs Confidence-Only")
print("=" * 65)

summary = (
    df_24f.groupby(["corruption", "variant"])
    .agg(
        vs_Baseline = ("vs_baseline", "mean"),
        Beats       = ("beats_baseline", "sum"),
        N           = ("beats_baseline", "count"),
        Fired       = ("ttt_fired",   "sum"),
        VerifiedAvg = ("n_verified",  "mean"),
    )
    .round(4)
)
print(summary.to_string())

print()
print("─" * 65)
print("OVERALL (both corruptions combined)")
print("─" * 65)
overall = (
    df_24f.groupby("variant")
    .agg(
        vs_Baseline = ("vs_baseline", "mean"),
        Beats       = ("beats_baseline", "sum"),
        N           = ("beats_baseline", "count"),
    )
    .round(4)
)
print(overall.to_string())

print()
improvement = (
    overall.loc["detection_loss", "vs_Baseline"]
    - overall.loc["confidence_only", "vs_Baseline"]
)
print(f"Detection loss improvement over confidence-only: {improvement:+.4f}")
if improvement > 0.005:
    print("✅ Detection loss shows meaningful improvement — worth scaling up.")
elif improvement > 0:
    print("➖ Detection loss marginally better — consider adjusting loss weights before scaling.")
else:
    print("❌ Detection loss no better or worse — deeper investigation needed.")

os.makedirs("/kaggle/working/tables", exist_ok=True)
df_24f.to_csv("/kaggle/working/tables/table_block24f_detection_loss.csv", index=False)
print("\n✅ Saved: table_block24f_detection_loss.csv")
print()
print("=" * 65)
print("BLOCK 24f COMPLETE")
print("=" * 65)
