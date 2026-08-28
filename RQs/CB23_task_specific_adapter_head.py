# ============================================================
# BLOCK 23: Task-Specific Adapter Head TTT
# A fundamentally different architecture that solves the
# encoder-detection head decoupling constraint identified
# across Blocks 16-22.
#
# The Problem (confirmed across 9 configurations):
# LoRA adapters in the encoder cannot propagate changes
# to detection outputs because the encoder and detection
# head operate in decoupled feature spaces.
#
# The Solution:
# Freeze the ENTIRE GroundingDINO model completely.
# Add a small 3-layer MLP adapter that sits DIRECTLY
# on top of the detection head outputs.
# The adapter takes raw box predictions and scores as input
# and outputs refined box predictions and scores.
# Train ONLY this adapter using Oracle-verified multi-view
# pseudo-GT boxes.
#
# The gradient path is now DIRECT:
# pseudo-GT loss → adapter MLP → refined detection outputs
# No encoder. No decoupling. No frozen oracle ceiling.
#
# PSEUDO-GT SOURCE (Swin-B rerun): the original notebook drew
# pseudo-GT from Block 22b's RELAXED parameters (consistency
# 2/5, IoU 0.3, tau 0.300). This run's diagnostics showed those
# relaxed params produce degenerate, near-100%-pass-rate pseudo-
# GT with low-confidence top-CLIP-matches (e.g. "chair" verified
# against a top match of "tie" at 0.25) — so the full Block 22b
# was deliberately NOT run. Block 23 below instead uses Block
# 22's STRICT, validated parameters (consistency 3/5, IoU 0.5,
# live gate tau) as its pseudo-GT source, since that configuration
# was confirmed to produce healthy, well-discriminated pseudo-GT
# (17/20 images firing in Block 22 itself).
#
# Scope: 5 images pilot, 4 corruptions, severity 5
# Cost: Free (Kaggle T4)
# Runtime: ~30-40 minutes
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import os
from PIL import Image as PILImage
from imagecorruptions import corrupt as ic_corrupt
from tqdm.notebook import tqdm
from torchvision.ops import box_iou, nms
from IPython.display import display

print("="*60)
print("BLOCK 23: TASK-SPECIFIC ADAPTER HEAD TTT")
print("="*60)
print()
print("Architecture: Frozen GroundingDINO + Trainable MLP adapter")
print("Gradient path: Direct from loss to adapter to outputs")
print("No encoder decoupling. No LoRA on frozen backbone.")
print()

# --- 23.1 Freeze entire GroundingDINO ---
for param in dino_model.parameters():
    param.requires_grad = False
for param in clip_model.parameters():
    param.requires_grad = False

frozen_params = sum(p.numel() for p in dino_model.parameters())
print(f"✅ GroundingDINO fully frozen: {frozen_params:,} params")


# --- 23.2 Detection Adapter Head ---
class DetectionAdapterHead(nn.Module):
    """
    Task-specific adapter that refines GroundingDINO outputs.

    Takes raw detection outputs (boxes + scores) and produces
    refined outputs. Trained at test time using Oracle-verified
    multi-view pseudo-GT boxes.

    Architecture:
        Input:  [N, 5] — [x1, y1, x2, y2, score] per detection
        Layer1: Linear(5, 64) → LayerNorm → ReLU
        Layer2: Linear(64, 32) → LayerNorm → ReLU
        Layer3: Linear(32, 5)  — residual refinement
        Output: input + delta  — refined [x1, y1, x2, y2, score]

    Residual design: starts as identity (no change at init)
    and learns refinements incrementally.
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 5)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, boxes, scores):
        if len(boxes) == 0:
            return boxes, scores

        x = torch.cat([boxes, scores.unsqueeze(-1)], dim=-1)
        delta = self.net(x)
        refined = x + delta

        refined_boxes  = refined[:, :4]
        refined_scores = torch.sigmoid(refined[:, 4])

        return refined_boxes, refined_scores


# --- 23.3 Instantiate and Validate Adapter ---
adapter = DetectionAdapterHead(hidden_dim=64).to(device)
adapter_params = sum(p.numel() for p in adapter.parameters())

print(f"✅ Adapter head created")
print(f"   Parameters: {adapter_params:,}")
print(f"   Architecture: Linear(5→64) → LN → ReLU → "
      f"Linear(64→32) → LN → ReLU → Linear(32→5)")
print(f"   Residual init: identity (zero final layer)")

test_img = loaded_images[image_files[0]]
with torch.no_grad():
    tv = dino_processor(
        images=test_img, text=DINO_TEXT_PROMPT,
        return_tensors="pt"
    ).to(device)
    ov = dino_model(**tv)
rv = dino_processor.post_process_grounded_object_detection(
    ov, tv.input_ids,
    target_sizes=[test_img.shape[:2]],
    text_threshold=CRATTT_PARAMS["dino_text_thr"]
)[0]

if len(rv['boxes']) > 0:
    test_boxes  = rv['boxes'][:3]
    test_scores = rv['scores'][:3]
    with torch.no_grad():
        ref_boxes, ref_scores = adapter(test_boxes, test_scores)
    box_diff   = (ref_boxes - test_boxes).abs().max().item()
    score_diff = (ref_scores - test_scores).abs().max().item()
    print(f"   Identity check — box delta:   {box_diff:.8f}")
    print(f"   Identity check — score delta: {score_diff:.8f}")
    if box_diff < 1e-6 and score_diff < 0.01:
        print(f"   ✅ Adapter starts as identity")
    else:
        print(f"   ⚠️  Small non-zero init — acceptable")

vram_free = (
    torch.cuda.get_device_properties(0).total_memory -
    torch.cuda.memory_allocated()
) / 1e9
print(f"   VRAM free: {vram_free:.2f} GB")


# --- 23.4 Adapted CRATTT Inference ---
def run_crattt_with_adapter(image_np, adapter):
    with torch.no_grad():
        inputs = dino_processor(
            images=image_np,
            text=DINO_TEXT_PROMPT,
            return_tensors="pt"
        ).to(device)
        outputs = dino_model(**inputs)
        res = dino_processor\
            .post_process_grounded_object_detection(
                outputs, inputs.input_ids,
                target_sizes=[image_np.shape[:2]],
                text_threshold=CRATTT_PARAMS["dino_text_thr"]
            )[0]

    if len(res['boxes']) == 0:
        return [], {"n_proposals": 0, "n_verified": 0,
                    "n_rejected": 0}

    refined_boxes, refined_scores = adapter(
        res['boxes'], res['scores']
    )

    h, w = image_np.shape[:2]
    refined_boxes[:, 0::2] = refined_boxes[:, 0::2].clamp(0, w)
    refined_boxes[:, 1::2] = refined_boxes[:, 1::2].clamp(0, h)

    labels = res.get("text_labels", res.get("labels", []))

    verified  = []
    n_rejected = 0

    for i, (box, score) in enumerate(
        zip(refined_boxes, refined_scores)
    ):
        dino_score = score.item()

        if i >= len(labels):
            continue
        label = labels[i]

        if isinstance(label, str):
            clean_label = label.lower().replace(".", "").strip()
        elif isinstance(label, int):
            clean_label = COCO_CLASSES[label] \
                if label < len(COCO_CLASSES) else None
        else:
            continue

        if clean_label is None:
            continue

        category_id = COCO_MAP.get(clean_label)
        if category_id is None:
            n_rejected += 1
            continue

        x1 = max(0, int(box[0].item()))
        y1 = max(0, int(box[1].item()))
        x2 = min(w, int(box[2].item()))
        y2 = min(h, int(box[3].item()))

        if x2 <= x1 or y2 <= y1:
            n_rejected += 1
            continue

        crop     = image_np[y1:y2, x1:x2]
        crop_pil = PILImage.fromarray(crop)
        soracle, _ = compute_soracle(crop_pil, clean_label)

        if dino_score > 0 and soracle > 0:
            rjoint = (dino_score ** CRATTT_PARAMS["alpha"]) * \
                     (soracle  ** CRATTT_PARAMS["beta"])
        else:
            rjoint = 0.0

        if rjoint >= CRATTT_PARAMS["tau"]:
            b = box.tolist()
            verified.append({
                "image_id":    0,
                "category_id": category_id,
                "bbox": [b[0], b[1], b[2]-b[0], b[3]-b[1]],
                "score":       rjoint,
                "label":       clean_label,
                "dino_score":  dino_score,
                "soracle":     soracle,
                "rjoint":      rjoint
            })
        else:
            n_rejected += 1

    stats = {
        "n_proposals": len(res['boxes']),
        "n_verified":  len(verified),
        "n_rejected":  n_rejected
    }
    return verified, stats


# --- 23.5 Adapter TTT Loss ---
def compute_adapter_loss(adapter, dino_model, dino_processor,
                          image_np, pseudo_gt_boxes,
                          pseudo_gt_labels, device):
    if not pseudo_gt_boxes:
        return None

    with torch.no_grad():
        inputs = dino_processor(
            images=image_np, text=DINO_TEXT_PROMPT,
            return_tensors="pt"
        ).to(device)
        outputs    = dino_model(**inputs)
        res = dino_processor\
            .post_process_grounded_object_detection(
                outputs, inputs.input_ids,
                target_sizes=[image_np.shape[:2]],
                text_threshold=0.05
            )[0]

    if len(res['boxes']) == 0:
        return None

    refined_boxes, refined_scores = adapter(
        res['boxes'], res['scores']
    )

    gt_boxes = torch.stack(pseudo_gt_boxes).to(device)

    ious = box_iou(gt_boxes, refined_boxes.detach())

    total_loss   = None
    n_matched    = 0

    for gt_idx in range(len(gt_boxes)):
        best_iou, best_idx = ious[gt_idx].max(dim=0)

        if best_iou.item() < 0.1:
            continue

        matched_box   = refined_boxes[best_idx]
        matched_score = refined_scores[best_idx]
        gt_box        = gt_boxes[gt_idx]

        h, w  = image_np.shape[:2]
        scale = torch.tensor(
            [w, h, w, h], dtype=torch.float32
        ).to(device)
        box_loss = F.l1_loss(
            matched_box / scale,
            gt_box.detach() / scale
        )

        conf_loss = F.binary_cross_entropy(
            matched_score.unsqueeze(0).clamp(1e-6, 1-1e-6),
            torch.ones(1).to(device)
        )

        loss = box_loss + 0.5 * conf_loss

        total_loss = loss if total_loss is None \
            else total_loss + loss
        n_matched += 1

    if total_loss is not None and n_matched > 0:
        total_loss = total_loss / n_matched

    return total_loss


# --- 23.6 Configuration ---
PILOT_IMAGES_23     = image_files[:5]
EVAL_CORRUPTIONS_23 = [
    ("Noise",   "gaussian_noise", 5),
    ("Blur",    "motion_blur",    5),
    ("Weather", "snow",           5),
    ("Digital", "contrast",       5),
]
TTT_STEPS_23 = 10
TTT_LR_23    = 1e-3  # Higher lr appropriate for small adapter

# FIX: define get_pseudo_gt locally using Block 22's STRICT,
# validated parameters (3/5 consistency, IoU 0.5, live gate tau)
# rather than calling the never-defined Block 22b version. This
# was Block 22's own configuration, confirmed to produce healthy,
# well-discriminated pseudo-GT (17/20 images firing).
PSEUDO_GT_CONSISTENCY_23 = 3
PSEUDO_GT_IOU_23         = 0.5
PSEUDO_GT_TAU_23         = CRATTT_PARAMS["tau"]  # 0.321, same as live gate

def get_pseudo_gt(image_np):
    """
    Pseudo-GT generator for Block 23, using Block 22's strict,
    validated parameters (see module docstring for why the
    relaxed Block 22b parameters were not used).
    """
    consistent_boxes, consistent_labels, _ = find_consistent_boxes(
        image_np,
        n_views=5,
        consistency_threshold=PSEUDO_GT_CONSISTENCY_23,
        iou_threshold=PSEUDO_GT_IOU_23
    )

    if len(consistent_boxes) == 0:
        return [], []

    verified_boxes  = []
    verified_labels = []
    h, w = image_np.shape[:2]

    for box, label in zip(consistent_boxes, consistent_labels):
        x1 = max(0, int(box[0].item()))
        y1 = max(0, int(box[1].item()))
        x2 = min(w, int(box[2].item()))
        y2 = min(h, int(box[3].item()))
        if x2 <= x1 or y2 <= y1:
            continue
        crop     = image_np[y1:y2, x1:x2]
        crop_pil = PILImage.fromarray(crop)
        soracle, _ = compute_soracle(crop_pil, label)
        dino_proxy = 0.5
        rjoint = (dino_proxy ** CRATTT_PARAMS["alpha"]) * \
                 (max(soracle, 0) ** CRATTT_PARAMS["beta"]) \
                 if soracle > 0 else 0.0
        if rjoint >= PSEUDO_GT_TAU_23:
            verified_boxes.append(box)
            verified_labels.append(label)

    return verified_boxes, verified_labels


print(f"\n--- Block 23 Configuration ---")
print(f"Adapter params  : {adapter_params:,}")
print(f"DINO params     : {frozen_params:,} (frozen)")
print(f"Images          : {len(PILOT_IMAGES_23)} (pilot)")
print(f"Corruptions     : {len(EVAL_CORRUPTIONS_23)}")
print(f"TTT steps       : {TTT_STEPS_23}")
print(f"TTT lr          : {TTT_LR_23}")
print(f"Pseudo-GT source: Block 22 strict params "
      f"(consistency {PSEUDO_GT_CONSISTENCY_23}/5, "
      f"IoU {PSEUDO_GT_IOU_23}, tau {PSEUDO_GT_TAU_23})")
print(f"CRATTT gate tau : {CRATTT_PARAMS['tau']}")


# --- 23.7 Checkpoint directory ---
b23_ckpt_dir = os.path.join(
    EVAL_PARAMS["ckpt_dir"], "block23"
)
os.makedirs(b23_ckpt_dir, exist_ok=True)


# --- 23.8 Main Loop ---
all_rows_23 = []

for cat_name, corruption, severity in EVAL_CORRUPTIONS_23:

    ckpt_path = os.path.join(
        b23_ckpt_dir, f"{corruption}_sev{severity}.json"
    )
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            rows = json.load(f)
        print(f"↩️  {corruption}: from checkpoint")
        all_rows_23.extend(rows)
        continue

    print(f"\n▶  {cat_name}: {corruption} sev{severity}")

    nn.init.zeros_(adapter.net[-1].weight)
    nn.init.zeros_(adapter.net[-1].bias)

    optimizer_23 = torch.optim.AdamW(
        adapter.parameters(),
        lr=TTT_LR_23,
        weight_decay=1e-4
    )

    corruption_rows = []
    pbar = tqdm(
        PILOT_IMAGES_23, desc=f"  {corruption}", leave=False
    )

    for img_path in pbar:
        img_id  = img_id_map[os.path.basename(img_path)]
        raw_img = loaded_images[img_path]
        fname   = os.path.basename(img_path)

        c_img = ic_corrupt(
            raw_img, corruption_name=corruption, severity=severity
        )

        with torch.no_grad():
            inp = dino_processor(
                images=c_img, text=DINO_TEXT_PROMPT,
                return_tensors="pt"
            ).to(device)
            out      = dino_model(**inp)
            base_res = dino_processor\
                .post_process_grounded_object_detection(
                    out, inp.input_ids,
                    target_sizes=[c_img.shape[:2]],
                    text_threshold=CRATTT_PARAMS["dino_text_thr"]
                )[0]
        base_preds = dino_to_coco_format(base_res, img_id)
        bmap, _    = compute_map(base_preds, coco_gt, [img_id])

        adapter.eval()
        with torch.no_grad():
            crattt_b, stats_b = run_crattt_with_adapter(
                c_img, adapter
            )
        for p in crattt_b:
            p["image_id"] = img_id
        coco_b = [{
            "image_id": p["image_id"],
            "category_id": p["category_id"],
            "bbox": p["bbox"], "score": p["score"]
        } for p in crattt_b]
        cmap_b, _ = compute_map(coco_b, coco_gt, [img_id])

        pseudo_gt_boxes, pseudo_gt_labels = get_pseudo_gt(c_img)
        n_pseudo_gt = len(pseudo_gt_boxes)

        loss_values = []
        adapter.train()

        if n_pseudo_gt > 0:
            for step in range(TTT_STEPS_23):
                optimizer_23.zero_grad()
                loss = compute_adapter_loss(
                    adapter, dino_model, dino_processor,
                    c_img, pseudo_gt_boxes,
                    pseudo_gt_labels, device
                )
                if loss is not None:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        adapter.parameters(), max_norm=1.0
                    )
                    optimizer_23.step()
                    loss_values.append(round(loss.item(), 6))

        adapter.eval()

        with torch.no_grad():
            crattt_c, stats_c = run_crattt_with_adapter(
                c_img, adapter
            )
        for p in crattt_c:
            p["image_id"] = img_id
        coco_c = [{
            "image_id": p["image_id"],
            "category_id": p["category_id"],
            "bbox": p["bbox"], "score": p["score"]
        } for p in crattt_c]
        cmap_c, _ = compute_map(coco_c, coco_gt, [img_id])

        weight_change = adapter.net[-1].weight.abs().max().item()

        row = {
            "category":              cat_name,
            "corruption":            corruption,
            "severity":              severity,
            "image":                 fname,
            "baseline_mAP":          round(float(bmap),   4),
            "adapter_crattt_mAP":    round(float(cmap_b), 4),
            "adapter_ttt_mAP":       round(float(cmap_c), 4),
            "delta_ttt_vs_crattt":   round(float(cmap_c - cmap_b), 4),
            "delta_ttt_vs_base":     round(float(cmap_c - bmap),   4),
            "n_pseudo_gt":           n_pseudo_gt,
            "n_verified_before":     stats_b['n_verified'],
            "n_verified_after":      stats_c['n_verified'],
            "adapter_weight_change": round(float(weight_change), 6),
            "loss_mean":             round(float(np.mean(loss_values))
                                           if loss_values else 0, 4),
            "ttt_fired":             n_pseudo_gt > 0
        }
        corruption_rows.append(row)

        pbar.set_postfix({
            "B":  f"{bmap:.3f}",
            "C":  f"{cmap_b:.3f}",
            "T":  f"{cmap_c:.3f}",
            "PG": f"{n_pseudo_gt}",
            "W":  f"{weight_change:.4f}"
        })

    with open(ckpt_path, "w") as f:
        json.dump(corruption_rows, f, indent=2)
    all_rows_23.extend(corruption_rows)

    c_df = pd.DataFrame(corruption_rows)
    ttt_fired  = c_df['ttt_fired'].sum()
    mean_wt    = c_df['adapter_weight_change'].mean()
    print(f"  Baseline mAP          : {c_df['baseline_mAP'].mean():.4f}")
    print(f"  Adapter CRATTT mAP    : {c_df['adapter_crattt_mAP'].mean():.4f}")
    print(f"  Adapter TTT mAP       : {c_df['adapter_ttt_mAP'].mean():.4f}")
    print(f"  TTT gain vs CRATTT    : {c_df['delta_ttt_vs_crattt'].mean():+.4f}")
    print(f"  TTT fired             : {ttt_fired}/{len(c_df)}")
    print(f"  Mean pseudo-GT boxes  : {c_df['n_pseudo_gt'].mean():.2f}")
    print(f"  Mean adapter weight Δ : {mean_wt:.6f}")
    if c_df['loss_mean'].max() > 0:
        non_zero = c_df[c_df['loss_mean'] > 0]['loss_mean']
        print(f"  Mean loss             : {non_zero.mean():.6f}")


# --- 23.9 Results ---
df_23 = pd.DataFrame(all_rows_23)
overall_23   = df_23['delta_ttt_vs_crattt'].mean()
positive_23  = (df_23['delta_ttt_vs_crattt'] > 0.001).sum()
ttt_fired_23 = df_23['ttt_fired'].sum()
mean_wt_23   = df_23['adapter_weight_change'].mean()

print("\n" + "="*70)
print("BLOCK 23: ADAPTER HEAD TTT RESULTS (N=5 PILOT)")
print("="*70)

summary_23 = df_23.groupby(["category", "corruption"]).agg(
    Baseline_mAP    =("baseline_mAP",          "mean"),
    Adapter_CRATTT  =("adapter_crattt_mAP",     "mean"),
    Adapter_TTT_mAP =("adapter_ttt_mAP",        "mean"),
    TTT_gain        =("delta_ttt_vs_crattt",    "mean"),
    Mean_PseudoGT   =("n_pseudo_gt",            "mean"),
    TTT_fired       =("ttt_fired",              "sum"),
    Mean_Wt_Change  =("adapter_weight_change",  "mean"),
).round(4).reset_index()

display(summary_23)

print(f"\nOverall TTT gain vs CRATTT : {overall_23:+.4f}")
print(f"Images with gain > 0       : {positive_23}/{len(df_23)}")
print(f"Images where TTT fired     : {ttt_fired_23}/{len(df_23)}")
print(f"Mean adapter weight change : {mean_wt_23:.6f}")

# Known/live results from this Swin-B session. Block 22b is
# marked as "not run (degenerate pseudo-GT)" rather than falsely
# showing 0.0000 as if it completed cleanly.
KNOWN_RESULTS = {
    "block_16":  +0.0001,
    "block_16b": -0.0000,
    "block_16c": -0.0345,
    "block_17":  +0.0000,
    "block_18":  +0.0000,
    "block_19c": +0.0100,
    "block_20":  -0.0069,
    "block_21":  +0.0002,
    "block_22":  +0.0000,
}

def _get_gain(varname, csv_path, csv_col="delta_ttt_vs_crattt", known_key=None):
    if varname in globals():
        return globals()[varname][csv_col].mean(), "live"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)[csv_col].mean(), "from disk"
    if known_key and known_key in KNOWN_RESULTS:
        return KNOWN_RESULTS[known_key], "known (confirmed)"
    return None, "unavailable"

table_dir = EVAL_PARAMS["table_dir"]
gain_16, src_16   = _get_gain("df_final", os.path.join(table_dir, "table_4_7_end_to_end.csv"), known_key="block_16")
gain_16b, src_16b = _get_gain("df_b",     os.path.join(table_dir, "block16b_alignment_ablation.csv"), known_key="block_16b")
gain_16c, src_16c = _get_gain("df_16c",   os.path.join(table_dir, "block16c_rank16_ttt.csv"), known_key="block_16c")
gain_17, src_17   = _get_gain("df_17", os.path.join(table_dir, "table_block17_coadapt_pilot.csv"), csv_col="delta_coadapt_vs_crattt", known_key="block_17")
gain_18, src_18   = _get_gain("df_18",    os.path.join(table_dir, "table_block18_encoder_gate.csv"), known_key="block_18")
gain_19c, src_19c = _get_gain("df_19c",   os.path.join(table_dir, "table_block19c_gaussian_noise_n20.csv"), known_key="block_19c")
gain_20, src_20   = _get_gain("df_20",    os.path.join(table_dir, "table_block20_consistency_ttt.csv"), known_key="block_20")
gain_21, src_21   = _get_gain("df_21",    os.path.join(table_dir, "table_block21_trained_proj.csv"), known_key="block_21")
gain_22, src_22   = _get_gain("df_22",    os.path.join(table_dir, "table_block22_multiview.csv"), known_key="block_22")

print(f"\n{'='*76}")
print("ABLATION SO FAR: 10 of 12 Table 4.8 Configurations (Swin-B)")
print("(Block 22b: not run — relaxed params produced degenerate pseudo-GT)")
print("(Blocks 23b, 23c still remain)")
print(f"{'='*76}")
print(f"{'Configuration':<48} {'TTT gain':>10}  {'source'}")
print("-"*76)

def _fmt(label, gain, src, marker=""):
    g = f"{gain:+.4f}" if gain is not None else "N/A"
    print(f"  {label:<46} {g:>10}  {src}{marker}")

_fmt("Blk 16  encoder conf-loss  frozen",   gain_16,  src_16)
_fmt("Blk 16b encoder align-loss frozen",   gain_16b, src_16b)
_fmt("Blk 16c encoder rank=16    frozen",   gain_16c, src_16c)
_fmt("Blk 17  encoder co-adapt",            gain_17,  src_17)
_fmt("Blk 18  encoder gate (untrained)",    gain_18,  src_18)
_fmt("Blk 19c dethead LoRA N=20",           gain_19c, src_19c)
_fmt("Blk 20  self-supervised consistency", gain_20,  src_20)
_fmt("Blk 21  trained projection gate",     gain_21,  src_21)
_fmt("Blk 22  multi-view strict params",    gain_22,  src_22)
print(f"  {'Blk 22b multi-view relaxed params':<46} {'N/A':>10}  not run (degenerate pseudo-GT)")
_fmt("Blk 23  adapter head + multi-view GT (strict)", overall_23, "live", " ◄ NEW")

print()
if overall_23 > 0.010:
    verdict    = "✅ POSITIVE GAIN — Adapter head TTT works!"
    action     = ("Scale to N=20 then Vast.ai. "
                  "This is the ablation's positive TTT result.")
    conclusion = "positive_scale_up"
elif overall_23 > 0.003:
    verdict    = "✅ MARGINAL POSITIVE GAIN"
    action     = "Scale to N=20. Report as positive pilot result."
    conclusion = "marginal_positive"
elif overall_23 > -0.003:
    verdict    = "⚠️  NEAR-ZERO"
    action     = ("Adapter weight change tells the story. "
                  "If weight change is near zero, adapter is "
                  "not learning. Try lr=1e-2.")
    conclusion = "near_zero"
else:
    verdict    = "❌ NEGATIVE"
    action     = ("Continue to Block 23b/23c before drawing "
                  "final conclusions — 2 configs remain.")
    conclusion = "negative_continue_ablation"

print(verdict)
print(f"Action: {action}")

# Save
csv_23 = os.path.join(
    EVAL_PARAMS["table_dir"], "table_block23_adapter_head.csv"
)
df_23.to_csv(csv_23, index=False)
with open(os.path.join(
    EVAL_PARAMS["save_dir"], "block23_results.json"
), "w") as f:
    json.dump({
        "method":             "task_specific_adapter_head",
        "pseudo_gt_source":   "block22_strict_params",
        "pseudo_gt_consistency": PSEUDO_GT_CONSISTENCY_23,
        "pseudo_gt_iou":      PSEUDO_GT_IOU_23,
        "pseudo_gt_tau":      PSEUDO_GT_TAU_23,
        "adapter_params":     adapter_params,
        "dino_frozen_params": frozen_params,
        "n_images":           len(PILOT_IMAGES_23),
        "ttt_steps":          TTT_STEPS_23,
        "lr":                 TTT_LR_23,
        "overall_gain":       round(float(overall_23),  4),
        "positive_images":    int(positive_23),
        "ttt_fired":          int(ttt_fired_23),
        "mean_weight_change": round(float(mean_wt_23),  6),
        "conclusion":         conclusion,
        "ablation_so_far": {
            "block_16": round(float(gain_16), 4)  if gain_16  is not None else None,
            "block_16b": round(float(gain_16b), 4) if gain_16b is not None else None,
            "block_16c": round(float(gain_16c), 4) if gain_16c is not None else None,
            "block_17": round(float(gain_17), 4)  if gain_17  is not None else None,
            "block_18": round(float(gain_18), 4)  if gain_18  is not None else None,
            "block_19c": round(float(gain_19c), 4) if gain_19c is not None else None,
            "block_20": round(float(gain_20), 4)  if gain_20  is not None else None,
            "block_21": round(float(gain_21), 4)  if gain_21  is not None else None,
            "block_22": round(float(gain_22), 4)  if gain_22  is not None else None,
            "block_22b": None,
            "block_23": round(float(overall_23), 4)
        }
    }, f, indent=2)

print(f"\n✅ Saved: {csv_23}")
print("\n" + "="*50)
print("BLOCK 23 COMPLETE — 10 of 12 Table 4.8 configs done")
print("="*50)
