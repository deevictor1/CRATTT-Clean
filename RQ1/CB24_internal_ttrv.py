# ============================================================
# BLOCK 24: INTERNAL TTRV — Score-SNR Gated TTT
# Dada Victor Damilare | MRES7015 | University of Greater Manchester
# ============================================================
#
# FIXES APPLIED (this version):
# 1. dino_model.eval() forced in two places to guarantee
#    deterministic behaviour regardless of what mode a prior
#    block left the model in (top-level guard + inside
#    b24_run_dino() itself, the single choke point all other
#    helpers route through).
# 2. LoRA gradient re-enable: Block 23's adapter-head experiments
#    froze ALL of dino_model's parameters (including lora_A/
#    lora_B injected back in Block 22), so requires_grad would
#    be False on everything without this fix — opt_b24 would be
#    built from an empty parameter list and TTT would silently
#    do nothing.
# 3. LoRA B zeroed BEFORE calibration (24.5) and the gate
#    diagnostic (24.8), not just before the main loop (24.9).
#    Without this, lora_B still carries leftover weights from
#    Block 22's last-trained corruption (contrast) — meaning the
#    clean-image SNR_BASELINE and TAU_INTERNAL calibration, and
#    the gate diagnostic, would be computed against a model
#    silently contaminated by an unrelated experiment, not the
#    intended clean rank=4-but-untrained baseline.
# ============================================================
#
# SCIENTIFIC MOTIVATION
# ─────────────────────
# Blocks 15–23c established a definitive architectural constraint:
# the frozen external CLIP Oracle (Soracle) is DECOUPLED from
# LoRA updates inside GroundingDINO. When LoRA adapts DINO's
# encoder, the Oracle cannot see the change — the gate stays shut.
# This is the Representational Asymmetry Problem (Section 5.5).
#
# BLOCK 24 SOLUTION — INTERNAL TTRV:
# Replace the external CLIP Oracle with GroundingDINO's own
# multi-view Score Signal-to-Noise Ratio (Score-SNR).
#
#   S_snr = mean(score across k views) / (std(score) + ε)
#
# Internal Gate:
#   Rjoint_int = IoU_consensus^α × S_snr_norm^β ≥ τ_internal
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import os
import random as _rnd

from PIL import Image as PILImage
from imagecorruptions import corrupt as ic_corrupt
from torchvision.ops import box_iou
from tqdm.notebook import tqdm
from IPython.display import display

print("=" * 65)
print("BLOCK 24: INTERNAL TTRV — Score-SNR Gated TTT")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────
# 24.0  FORCE EVAL MODE + RE-ENABLE LORA GRADIENTS + ZERO LORA B
# ─────────────────────────────────────────────────────────────
was_training = dino_model.training
dino_model.eval()
print(f"  Model mode before fix : {'TRAIN' if was_training else 'eval'}")
print(f"  Model mode now        : eval (forced)")
if was_training:
    print("  ⚠️  Model was left in TRAIN mode by a prior block.")
    print("     This would have caused irreproducible SNR statistics.")
print()

# FIX 2: re-enable gradients on the existing LoRA tensors.
# Block 23.1 froze ALL of dino_model's parameters. The LoRA
# layers themselves are untouched (still rank=4, 36 of them,
# from Block 22's injection) — only requires_grad was turned
# off. Re-enabling it, not re-injecting: re-injection on an
# already-wrapped model would silently match 0 layers (the same
# bug pattern hit repeatedly in Blocks 16c/17/18/20/21/22).
n_lora_reenabled = 0
for name, param in dino_model.named_parameters():
    if 'lora_A' in name or 'lora_B' in name:
        param.requires_grad = True
        n_lora_reenabled += 1

trainable_check = sum(
    p.numel() for p in dino_model.parameters() if p.requires_grad
)
print(f"  ✅ Re-enabled grad on {n_lora_reenabled} LoRA tensors")
print(f"     Trainable params: {trainable_check:,} (should be 73,728)")
assert trainable_check == 73_728, (
    f"Expected 73,728 trainable params, got {trainable_check:,} — "
    f"check for leftover state from another block before continuing"
)

# FIX 3: zero lora_B NOW, before calibration/diagnostic run.
# Block 22's training left lora_B carrying weights from its last
# corruption (contrast). Block 23 only froze gradients, never
# zeroed or retrained these tensors — so without this, 24.5's
# calibration and 24.8's diagnostic would both run against a
# contaminated model, not a clean rank=4-but-untrained baseline.
n_zeroed = 0
for name, param in dino_model.named_parameters():
    if 'lora_B' in name:
        param.data.zero_()
        n_zeroed += 1
print(f"  ✅ Zeroed {n_zeroed} lora_B tensors "
      f"(clearing leftover weights from Block 22)")
print()

# ─────────────────────────────────────────────────────────────
# 24.1  CONFIGURATION
# ─────────────────────────────────────────────────────────────
B24 = {
    "n_views":         5,
    "iou_thr":         0.40,
    "consistency_thr": 2,

    "snr_eps":   1e-4,
    "alpha":     0.40,
    "beta":      0.60,

    "ttt_steps": 5,
    "ttt_lr":    1e-4,
    "max_norm":  1.0,

    "n_images":  20,
    "corruptions": [
        ("Noise",   "gaussian_noise", 5),
        ("Blur",    "motion_blur",    5),
        ("Weather", "snow",           5),
        ("Digital", "contrast",       5),
    ],

    "n_calib": 10,
}

print("Configuration loaded:")
for k, v in B24.items():
    if k != "corruptions":
        print(f"  {k:20s}: {v}")
print()


# ─────────────────────────────────────────────────────────────
# 24.2  AUGMENTATION (photometric only — boxes stay valid)
# ─────────────────────────────────────────────────────────────
import torchvision.transforms.functional as TF
import torchvision.transforms as T


def b24_augment(image_np: np.ndarray, seed: int) -> np.ndarray:
    _rnd.seed(seed * 31 + 7)
    img = PILImage.fromarray(image_np.astype(np.uint8))

    brightness = 0.85 + _rnd.random() * 0.30
    contrast   = 0.85 + _rnd.random() * 0.30
    img = TF.adjust_brightness(img, brightness)
    img = TF.adjust_contrast(img, contrast)

    if seed % 3 == 0:
        img = TF.adjust_saturation(img, 0.80 + _rnd.random() * 0.40)
    elif seed % 3 == 1:
        img = T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.4))(img)
    else:
        img = TF.adjust_hue(img, (-0.05 + _rnd.random() * 0.10))

    return np.array(img)


# ─────────────────────────────────────────────────────────────
# 24.3  SINGLE DINO INFERENCE (no gradient)
# ─────────────────────────────────────────────────────────────
def b24_run_dino(image_np: np.ndarray):
    """
    Runs GroundingDINO inference without gradients.
    Forces eval() mode before every forward pass, regardless of
    what mode the model was in beforehand. This is the single
    choke point all other helpers route through, so fixing it
    here also fixes b24_compute_snr, the calibration pass, and
    the TTT loop's post-update re-inference.
    """
    dino_model.eval()
    with torch.no_grad():
        inputs = dino_processor(
            images=image_np,
            text=DINO_TEXT_PROMPT,
            return_tensors="pt"
        ).to(device)
        outputs = dino_model(**inputs)
        results = dino_processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            target_sizes=[image_np.shape[:2]],
            text_threshold=CRATTT_PARAMS["dino_text_thr"]
        )[0]

    label_key = "text_labels" if "text_labels" in results else "labels"
    raw_labels = results[label_key]
    labels = [str(l).lower().replace(".", "").strip() for l in raw_labels]

    return results["boxes"], results["scores"], labels


# ─────────────────────────────────────────────────────────────
# 24.4  MULTI-VIEW SCORE-SNR COMPUTATION
# ─────────────────────────────────────────────────────────────
def b24_compute_snr(image_np: np.ndarray) -> list:
    ref_boxes, ref_scores, ref_labels = b24_run_dino(image_np)

    if len(ref_boxes) == 0:
        return []

    view_outputs = []
    for v in range(1, B24["n_views"]):
        aug = b24_augment(image_np, seed=v)
        vb, vs, vl = b24_run_dino(aug)
        view_outputs.append((vb, vs, vl))

    detections = []
    for ref_box, ref_score, ref_label in zip(ref_boxes, ref_scores, ref_labels):
        clean_label = ref_label
        if clean_label not in COCO_MAP:
            continue

        scores_seen = [ref_score.item()]
        match_count = 1

        rb_exp = ref_box.unsqueeze(0)

        for (vb, vs, _) in view_outputs:
            if len(vb) == 0:
                continue
            ious = box_iou(rb_exp, vb)
            max_iou, best_j = ious[0].max(0)
            if max_iou.item() >= B24["iou_thr"]:
                scores_seen.append(vs[best_j].item())
                match_count += 1

        arr        = np.array(scores_seen)
        score_mean = float(arr.mean())
        score_std  = float(arr.std())
        s_snr      = score_mean / (score_std + B24["snr_eps"])
        iou_cons   = match_count / B24["n_views"]

        detections.append({
            "box":           ref_box,
            "label":         clean_label,
            "score_mean":    score_mean,
            "score_std":     score_std,
            "s_snr":         s_snr,
            "iou_consensus": iou_cons,
            "match_count":   match_count,
        })

    return detections


# ─────────────────────────────────────────────────────────────
# 24.5  CLEAN BASELINE CALIBRATION
# ─────────────────────────────────────────────────────────────
print("─" * 50)
print("24.5  Calibrating Internal Gate on Clean Images")
print("─" * 50)
print(f"  Model mode entering calibration: "
      f"{'TRAIN ⚠️' if dino_model.training else 'eval ✅'}")
print(f"  lora_B state entering calibration: zeroed (see 24.0)")

calib_snr    = []
calib_rjoint = []
min_cons     = B24["consistency_thr"] / B24["n_views"]

for img_path in image_files[: B24["n_calib"]]:
    raw = loaded_images[img_path]
    dets = b24_compute_snr(raw)
    for d in dets:
        if d["iou_consensus"] >= min_cons:
            calib_snr.append(d["s_snr"])
            rj = (d["iou_consensus"] ** B24["alpha"] *
                  d["s_snr"]         ** B24["beta"])
            calib_rjoint.append(rj)

if len(calib_rjoint) < 5:
    print("⚠️  Fewer than 5 calibration detections — using fallback values")
    SNR_BASELINE = 3.0
    TAU_INTERNAL = 0.45
else:
    SNR_BASELINE = float(np.percentile(calib_snr, 25))
    TAU_INTERNAL = float(np.percentile(calib_rjoint, 25))

    print(f"  Calib detections     : {len(calib_rjoint)}")
    print(f"  SNR mean (clean)     : {np.mean(calib_snr):.3f}")
    print(f"  SNR median (clean)   : {np.median(calib_snr):.3f}")
    print(f"  SNR p25  (baseline)  : {SNR_BASELINE:.3f}")
    print(f"  Rjoint mean (clean)  : {np.mean(calib_rjoint):.4f}")
    print(f"  τ_internal (p25)     : {TAU_INTERNAL:.4f}")

B24["snr_baseline"] = SNR_BASELINE
B24["tau_internal"] = TAU_INTERNAL
print()


# ─────────────────────────────────────────────────────────────
# 24.6  INTERNAL TTRV GATE APPLICATION
# ─────────────────────────────────────────────────────────────
def b24_apply_gate(detections: list):
    v_boxes, v_labels, v_rjoints = [], [], []

    for d in detections:
        snr_norm = min(d["s_snr"] / max(B24["snr_baseline"], 1e-6), 2.0)
        rjoint   = (d["iou_consensus"] ** B24["alpha"] *
                    snr_norm           ** B24["beta"])

        if rjoint >= B24["tau_internal"]:
            v_boxes.append(d["box"])
            v_labels.append(d["label"])
            v_rjoints.append(rjoint)

    return v_boxes, v_labels, v_rjoints


# ─────────────────────────────────────────────────────────────
# 24.7  TTT LOSS — COUPLED CONFIDENCE MAXIMISATION
# ─────────────────────────────────────────────────────────────
def b24_confidence_loss(image_np: np.ndarray,
                         verified_boxes: list,
                         device: torch.device):
    if len(verified_boxes) == 0:
        return None

    inputs = dino_processor(
        images=image_np,
        text=DINO_TEXT_PROMPT,
        return_tensors="pt"
    ).to(device)

    outputs = dino_model(**inputs)

    scores_raw = outputs.logits.sigmoid().max(dim=-1).values[0]

    img_h, img_w = image_np.shape[:2]
    pred_cxcywh  = outputs.pred_boxes[0]
    cx = pred_cxcywh[:, 0] * img_w
    cy = pred_cxcywh[:, 1] * img_h
    pw = pred_cxcywh[:, 2] * img_w
    ph = pred_cxcywh[:, 3] * img_h
    pred_xyxy = torch.stack(
        [cx - pw / 2, cy - ph / 2, cx + pw / 2, cy + ph / 2], dim=-1
    )

    vb_tensor = torch.stack(verified_boxes).to(device)

    with torch.no_grad():
        iou_mat = box_iou(vb_tensor, pred_xyxy.detach())

    best_iou_vals, best_q_ids = iou_mat.max(dim=1)
    valid_mask = best_iou_vals > 0.25

    if valid_mask.sum() == 0:
        return None

    matched_ids    = best_q_ids[valid_mask]
    scores_matched = scores_raw[matched_ids]

    loss = -scores_matched.mean()
    return loss


# ─────────────────────────────────────────────────────────────
# 24.8  DIAGNOSTIC — Verify gate fires on corrupted image
# ─────────────────────────────────────────────────────────────
print("─" * 50)
print("24.8  Gate Diagnostic on Sample Corrupted Image")
print("─" * 50)

_test_raw   = loaded_images[image_files[0]]
_test_c_img = ic_corrupt(_test_raw, corruption_name="gaussian_noise", severity=5)
_test_dets  = b24_compute_snr(_test_c_img)
_vb, _vl, _vr = b24_apply_gate(_test_dets)

print(f"  DINO proposals (corrupted) : {len(_test_dets)}")
print(f"  Gate passed (verified)     : {len(_vb)}")
if len(_test_dets) > 0:
    _snr_vals = [d["s_snr"] for d in _test_dets]
    print(f"  SNR range (corrupted)      : [{min(_snr_vals):.2f}, {max(_snr_vals):.2f}]")
    print(f"  SNR mean  (corrupted)      : {np.mean(_snr_vals):.2f}  (clean baseline: {SNR_BASELINE:.2f})")

if len(_vb) == 0:
    print()
    print("  ⚠️  Gate yielded zero verified detections.")
    print(f"  → Current tau_internal = {TAU_INTERNAL:.4f}")
    print("  → Try: B24['tau_internal'] *= 0.7  then re-run 24.8")
else:
    print(f"  ✅ Gate is active — {len(_vb)} detections verified")
print()


# ─────────────────────────────────────────────────────────────
# 24.9  RESET LORA + INITIALISE OPTIMISER
# (lora_B already zeroed in 24.0; re-zeroed here too, harmless
#  and keeps this section self-contained as a clean reset point
#  immediately before the main loop)
# ─────────────────────────────────────────────────────────────
print("─" * 50)
print("24.9  Resetting LoRA B matrices to zero")
print("─" * 50)

for name, param in dino_model.named_parameters():
    if "lora_B" in name:
        param.data.zero_()

n_trainable = sum(p.numel() for p in dino_model.parameters()
                  if p.requires_grad)
print(f"  ✅ LoRA reset — trainable params: {n_trainable:,}")
assert n_trainable == 73_728, (
    f"Expected 73,728 trainable params, got {n_trainable:,}"
)

opt_b24 = torch.optim.AdamW(
    [p for p in dino_model.parameters() if p.requires_grad],
    lr=B24["ttt_lr"],
    weight_decay=1e-4,
)
print(f"  ✅ AdamW initialised (lr={B24['ttt_lr']})")
print()


# ─────────────────────────────────────────────────────────────
# 24.10  MAIN EVALUATION LOOP
# ─────────────────────────────────────────────────────────────
print("─" * 50)
print("24.10  Main Evaluation")
print("─" * 50)
print()

all_rows = []

for (cat, corruption, severity) in B24["corruptions"]:

    for name, param in dino_model.named_parameters():
        if "lora_B" in name:
            param.data.zero_()

    cat_rows   = []
    weight_log = []

    for img_path in tqdm(image_files[: B24["n_images"]],
                          desc=f"{corruption} sev{severity}"):

        img_id  = img_id_map[os.path.basename(img_path)]
        raw_img = loaded_images[img_path]
        c_img   = ic_corrupt(raw_img,
                              corruption_name=corruption,
                              severity=severity)

        dino_model.eval()
        boxes_b, scores_b, labels_b = b24_run_dino(c_img)

        preds_baseline = [
            {
                "image_id":    img_id,
                "category_id": COCO_MAP[l],
                "bbox": [b[0].item(), b[1].item(),
                         (b[2] - b[0]).item(), (b[3] - b[1]).item()],
                "score": s.item(),
            }
            for b, s, l in zip(boxes_b, scores_b, labels_b)
            if l in COCO_MAP
        ]
        mAP_base, _ = compute_map(preds_baseline, coco_gt, [img_id])

        dets = b24_compute_snr(c_img)
        v_boxes, v_labels, v_rjoints = b24_apply_gate(dets)

        preds_crattt = [
            {
                "image_id":    img_id,
                "category_id": COCO_MAP[l],
                "bbox": [b[0].item(), b[1].item(),
                         (b[2] - b[0]).item(), (b[3] - b[1]).item()],
                "score": rj,
            }
            for b, l, rj in zip(v_boxes, v_labels, v_rjoints)
            if l in COCO_MAP
        ]
        mAP_crattt, _ = compute_map(preds_crattt, coco_gt, [img_id])

        mAP_ttt    = mAP_crattt
        ttt_fired  = False
        pre_loss_v = None

        if len(v_boxes) > 0:
            dino_model.train()

            for step in range(B24["ttt_steps"]):
                opt_b24.zero_grad()
                loss = b24_confidence_loss(c_img, v_boxes, device)

                if loss is not None and loss.requires_grad:
                    if pre_loss_v is None:
                        pre_loss_v = round(loss.item(), 5)
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        [p for p in dino_model.parameters()
                         if p.requires_grad],
                        max_norm=B24["max_norm"]
                    )
                    opt_b24.step()
                    ttt_fired = True

            dino_model.eval()

            if ttt_fired:
                boxes_t, scores_t, labels_t = b24_run_dino(c_img)
                preds_ttt = [
                    {
                        "image_id":    img_id,
                        "category_id": COCO_MAP[l],
                        "bbox": [b[0].item(), b[1].item(),
                                 (b[2] - b[0]).item(),
                                 (b[3] - b[1]).item()],
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
            "pre_loss":     pre_loss_v,
        })

    all_rows.extend(cat_rows)

    df_cat    = pd.DataFrame(cat_rows)
    mean_gain = df_cat["TTT_gain"].mean()
    pos_imgs  = int((df_cat["TTT_gain"] > 0).sum())
    fired_n   = int(df_cat["ttt_fired"].sum())
    mean_ver  = df_cat["n_verified"].mean()
    mean_wt   = float(np.mean(weight_log)) if weight_log else 0.0

    print(f"[{cat:7s} | {corruption:15s} | sev{severity}]  "
          f"TTT gain: {mean_gain:+.4f}  |  "
          f"+ve: {pos_imgs}/{B24['n_images']}  |  "
          f"Fired: {fired_n}/{B24['n_images']}  |  "
          f"VerifiedAvg: {mean_ver:.1f}  |  "
          f"LoRA Δ: {mean_wt:.5f}")


# ─────────────────────────────────────────────────────────────
# 24.11  RESULTS SUMMARY
# ─────────────────────────────────────────────────────────────
df_b24 = pd.DataFrame(all_rows)

summary = (
    df_b24
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

print()
print("=" * 65)
print("BLOCK 24 RESULTS — INTERNAL TTRV + COUPLED CONFIDENCE TTT")
print("=" * 65)
display(summary)

overall_gain = float(df_b24["TTT_gain"].mean())
pos_total    = int((df_b24["TTT_gain"] > 0).sum())
total        = len(df_b24)
mean_ver     = df_b24["n_verified"].mean()

print(f"\nOverall TTT gain         : {overall_gain:+.4f}")
print(f"Images with TTT gain > 0 : {pos_total}/{total}")
print(f"Mean verified / image    : {mean_ver:.2f}")
print(f"τ_internal used          : {TAU_INTERNAL:.4f}")
print(f"SNR baseline used        : {SNR_BASELINE:.3f}")

os.makedirs("/kaggle/working/tables", exist_ok=True)
os.makedirs("/kaggle/working/results", exist_ok=True)
csv_out = "/kaggle/working/tables/table_block24_internal_ttrv.csv"
df_b24.to_csv(csv_out, index=False)

with open("/kaggle/working/results/block24_summary.json", "w") as f:
    json.dump({
        "block":          "24",
        "method":         "Internal_TTRV_Score_SNR",
        "model_mode_fix_applied": True,
        "was_training_before_fix": was_training,
        "lora_grad_reenabled": n_lora_reenabled,
        "lora_b_zeroed_pre_calibration": n_zeroed,
        "n_images":       B24["n_images"],
        "n_views":        B24["n_views"],
        "tau_internal":   round(TAU_INTERNAL, 4),
        "snr_baseline":   round(SNR_BASELINE, 3),
        "overall_gain":   round(overall_gain, 4),
        "positive_images": pos_total,
        "total_images":   total,
        "mean_verified":  round(float(mean_ver), 2),
    }, f, indent=2)

print(f"\n✅ Saved: {csv_out}")
print()
print("=" * 65)
print("BLOCK 24 COMPLETE")
print("=" * 65)
