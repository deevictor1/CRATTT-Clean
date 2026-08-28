# ============================================================
# BLOCK 16b: TTT with Alignment Loss (Ablation)
# Tests whether targeting cross-modal alignment directly
# (rather than confidence scores) produces TTT gain.
#
# Motivation: Block 16 showed zero TTT gain because the
# confidence-based loss did not affect the TTRV gate.
# The gate is dominated by Soracle (frozen CLIP), so the
# TTT update must target the DINO-CLIP alignment directly.
#
# This block replaces compute_pseudo_label_loss with
# compute_alignment_loss which maximises cosine similarity
# between LoRA-updated DINO features and CLIP text embeddings
# for verified classes.
#
# TABLE NOTE: this feeds one row of Table 4.8 (the 12-row
# Block 15-23c ablation summary), it does not produce its own
# numbered dissertation table. Earlier versions of this block
# mislabeled this "Table 4.4" -- that number already belongs to
# Block 11c's precision proxy table. Fixed below: outputs are
# named block16b_*, not table_4_4_*.
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import os
from imagecorruptions import corrupt as ic_corrupt
from tqdm.notebook import tqdm
from IPython.display import display

# --- 16b.1 Alignment Loss Function ---
def compute_alignment_loss(dino_model, dino_processor,
                            image_np, verified_preds, device):
    """
    Cross-modal alignment loss for TTT.

    Maximises cosine similarity between LoRA-updated DINO
    encoder features and CLIP text embeddings for verified
    classes. This directly targets the cross-modal alignment
    component measured by the TTRV gate's Soracle term.

    Args:
        dino_model     : GroundingDINO with LoRA adapters
        dino_processor : processor
        image_np       : corrupted image [H, W, 3]
        verified_preds : Oracle-verified detections from CRATTT
        device         : cuda/cpu

    Returns:
        loss : scalar tensor (1 - cosine_similarity)
               minimising this maximises alignment
    """
    if not verified_preds:
        return None

    inputs = dino_processor(
        images=image_np,
        text=DINO_TEXT_PROMPT,
        return_tensors="pt"
    ).to(device)

    outputs = dino_model(**inputs)

    # Extract encoder hidden states — these are what LoRA modifies
    # GroundingDINO encoder output shape: [1, num_tokens, hidden_dim]
    if hasattr(outputs, 'encoder_last_hidden_state') and \
       outputs.encoder_last_hidden_state is not None:
        visual_features = outputs.encoder_last_hidden_state
    elif hasattr(outputs, 'last_hidden_state') and \
         outputs.last_hidden_state is not None:
        visual_features = outputs.last_hidden_state
    else:
        # Fallback: use logits as proxy
        visual_features = outputs.logits if hasattr(
            outputs, 'logits'
        ) else None

    if visual_features is None:
        return None

    # Mean pool across spatial/token dimension
    # Shape: [1, hidden_dim]
    pooled = visual_features.mean(dim=1)
    hidden_dim = pooled.shape[-1]

    # Get target CLIP text embeddings for verified classes
    target_embeddings = []
    for pred in verified_preds:
        label = pred.get('label', '')
        if isinstance(label, str):
            clean = label.lower().replace(".", "").strip()
            if clean in COCO_CLASSES:
                idx = COCO_CLASSES.index(clean)
                target_embeddings.append(clip_text_features[idx])

    if not target_embeddings:
        return None

    # Mean target embedding across verified classes
    # Shape: [1, 512]
    target = torch.stack(target_embeddings).mean(
        dim=0, keepdim=True
    ).to(device)
    clip_dim = target.shape[-1]

    # Project DINO features to CLIP dimension if needed
    # This projection is part of LoRA's adaptation scope
    if hidden_dim != clip_dim:
        proj_key = f"proj_{hidden_dim}_{clip_dim}"
        if not hasattr(compute_alignment_loss, proj_key):
            proj = torch.nn.Linear(
                hidden_dim, clip_dim, bias=False
            ).to(device)
            # Initialise as approximate identity via SVD
            nn.init.orthogonal_(proj.weight)
            setattr(compute_alignment_loss, proj_key, proj)
        proj = getattr(compute_alignment_loss, proj_key)
        pooled = proj(pooled)

    # Normalise both for cosine similarity
    pooled_norm = F.normalize(pooled, p=2, dim=-1)
    target_norm = F.normalize(target, p=2, dim=-1)

    # Cosine similarity: 1.0 = perfect alignment
    similarity = (pooled_norm * target_norm).sum(dim=-1)

    # Loss: 1 - similarity
    # Minimising this maximises alignment with verified classes
    loss = (1.0 - similarity).mean()
    return loss


# --- 16b.2 Configuration ---
EVAL_CORRUPTIONS_B = [
    ("Noise",   "gaussian_noise", 5),
    ("Blur",    "motion_blur",    5),
    ("Weather", "snow",           5),
    ("Digital", "contrast",       5),
]

TTT_STEPS_B = 3
TTT_LR_B    = 5e-5

print("="*50)
print("BLOCK 16b: TTT ALIGNMENT LOSS ABLATION")
print("="*50)
print(f"Loss function : compute_alignment_loss")
print(f"Images        : {len(image_files)}")
print(f"Corruptions   : {len(EVAL_CORRUPTIONS_B)}")
print(f"TTT steps     : {TTT_STEPS_B}")
print(f"TTT lr        : {TTT_LR_B}")
print(f"Tau           : {CRATTT_PARAMS['tau']}")
print()

# --- 16b.3 Resume Logic ---
b16b_ckpt_dir = os.path.join(
    EVAL_PARAMS["ckpt_dir"], "block16b"
)
os.makedirs(b16b_ckpt_dir, exist_ok=True)

def load_b16b_checkpoint(corruption, severity):
    path = os.path.join(
        b16b_ckpt_dir, f"{corruption}_sev{severity}.json"
    )
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def save_b16b_checkpoint(corruption, severity, data):
    path = os.path.join(
        b16b_ckpt_dir, f"{corruption}_sev{severity}.json"
    )
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

# --- 16b.4 Main Loop ---
all_rows_b = []

for cat_name, corruption, severity in EVAL_CORRUPTIONS_B:

    # Check checkpoint
    ckpt = load_b16b_checkpoint(corruption, severity)
    if ckpt is not None:
        print(f"↩️  {corruption} sev{severity}: from checkpoint")
        all_rows_b.extend(ckpt)
        continue

    print(f"\n▶  {cat_name}: {corruption} sev{severity}")

    # Reset LoRA B weights to zero before each corruption
    # Ensures fair comparison — each starts from same state
    for name, param in dino_model.named_parameters():
        if 'lora_B' in name:
            param.data.zero_()

    # Clear alignment loss projection cache
    for attr in list(vars(compute_alignment_loss).keys()):
        if attr.startswith('proj_'):
            delattr(compute_alignment_loss, attr)

    # Fresh optimizer
    optimizer_b = torch.optim.AdamW(
        [p for p in dino_model.parameters() if p.requires_grad],
        lr=TTT_LR_B,
        weight_decay=1e-4
    )

    corruption_rows = []
    pbar = tqdm(
        image_files, desc=f"  {corruption}", leave=False
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

        # === CONDITION A: Baseline DINO ===
        with torch.no_grad():
            inputs = dino_processor(
                images=c_img,
                text=DINO_TEXT_PROMPT,
                return_tensors="pt"
            ).to(device)
            outputs = dino_model(**inputs)
            baseline_res = dino_processor\
                .post_process_grounded_object_detection(
                    outputs, inputs.input_ids,
                    target_sizes=[c_img.shape[:2]],
                    text_threshold=CRATTT_PARAMS["dino_text_thr"]
                )[0]

        baseline_preds = dino_to_coco_format(baseline_res, img_id)
        bmap, _ = compute_map(baseline_preds, coco_gt, [img_id])

        # === CONDITION B: CRATTT only (no TTT) ===
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

        # === CONDITION C: CRATTT + TTT (alignment loss) ===
        loss_values = []
        dino_model.train()

        for step in range(TTT_STEPS_B):
            optimizer_b.zero_grad()
            loss = compute_alignment_loss(
                dino_model, dino_processor,
                c_img, crattt_preds_b, device
            )
            if loss is not None:
                loss.backward()
                optimizer_b.step()
                loss_values.append(round(loss.item(), 6))

        dino_model.eval()

        # Re-run CRATTT with alignment-updated weights
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

        row = {
            "category":            cat_name,
            "corruption":          corruption,
            "severity":            severity,
            "image":               fname,
            "img_id":              img_id,
            "baseline_mAP":        round(float(bmap), 4),
            "crattt_mAP":          round(float(cmap_b), 4),
            "crattt_ttt_mAP":      round(float(cmap_c), 4),
            "delta_crattt":        round(float(cmap_b - bmap), 4),
            "delta_ttt":           round(float(cmap_c - bmap), 4),
            "delta_ttt_vs_crattt": round(float(cmap_c - cmap_b), 4),
            "n_baseline":          len(baseline_res['boxes']),
            "n_crattt":            stats_b['n_verified'],
            "n_ttt":               stats_c['n_verified'],
            "n_rejected_b":        stats_b['n_rejected'],
            "n_rejected_c":        stats_c['n_rejected'],
            "loss_trajectory":     loss_values
        }
        corruption_rows.append(row)

        pbar.set_postfix({
            "B": f"{bmap:.3f}",
            "C": f"{cmap_b:.3f}",
            "T": f"{cmap_c:.3f}"
        })

    save_b16b_checkpoint(corruption, severity, corruption_rows)
    all_rows_b.extend(corruption_rows)

    c_df = pd.DataFrame(corruption_rows)
    print(f"  Baseline mAP      : {c_df['baseline_mAP'].mean():.4f}")
    print(f"  CRATTT mAP        : {c_df['crattt_mAP'].mean():.4f}")
    print(f"  CRATTT+TTT mAP    : {c_df['crattt_ttt_mAP'].mean():.4f}")
    print(f"  TTT gain vs CRATTT: "
          f"{c_df['delta_ttt_vs_crattt'].mean():+.4f}")

    # Sample loss trajectory from first image with updates
    sample_loss = next(
        (r['loss_trajectory'] for r in corruption_rows
         if r['loss_trajectory']), []
    )
    if sample_loss:
        print(f"  Sample loss traj  : {sample_loss}")

# --- 16b.5 Results Table ---
df_b = pd.DataFrame(all_rows_b)

print("\n" + "="*70)
print("BLOCK 16b RESULTS: ALIGNMENT LOSS ABLATION")
print("(feeds one row of Table 4.8 -- see below)")
print("="*70)

summary_b = df_b.groupby(
    ["category", "corruption", "severity"]
).agg(
    Baseline_mAP        =("baseline_mAP",        "mean"),
    CRATTT_mAP          =("crattt_mAP",           "mean"),
    CRATTT_TTT_mAP      =("crattt_ttt_mAP",       "mean"),
    Delta_CRATTT        =("delta_crattt",          "mean"),
    Delta_TTT           =("delta_ttt",             "mean"),
    Delta_TTT_vs_CRATTT =("delta_ttt_vs_crattt",   "mean"),
    Mean_Rejected_B     =("n_rejected_b",          "mean"),
    Mean_Rejected_C     =("n_rejected_c",          "mean"),
).round(4).reset_index()

display(summary_b)

print(f"\n--- Overall Summary ---")
print(f"Mean Baseline mAP    : "
      f"{df_b['baseline_mAP'].mean():.4f}")
print(f"Mean CRATTT mAP      : "
      f"{df_b['crattt_mAP'].mean():.4f}")
print(f"Mean CRATTT+TTT mAP  : "
      f"{df_b['crattt_ttt_mAP'].mean():.4f}")
print(f"TTT gain vs Baseline : "
      f"{df_b['delta_ttt'].mean():+.4f}")
print(f"TTT gain vs CRATTT   : "
      f"{df_b['delta_ttt_vs_crattt'].mean():+.4f}")

# --- 16b.6 Comparison: Block 16 vs Block 16b ---
# FIXED: reads Block 16's live result from df_final (still in
# memory from that run) instead of a hardcoded "0.0000" string.
print(f"\n--- Ablation Comparison ---")
print(f"{'Condition':<30} {'TTT gain vs CRATTT':>20}")
print("-"*52)
print(f"{'Block 16 (confidence loss)':<30} "
      f"{df_final['delta_ttt_vs_crattt'].mean():>+20.4f}")
print(f"{'Block 16b (alignment loss)':<30} "
      f"{df_b['delta_ttt_vs_crattt'].mean():>+20.4f}")

# --- 16b.7 Save ---
# FIXED: renamed from table_4_4_* (that number belongs to
# Block 11c) to block16b_* — this isn't a standalone numbered
# dissertation table.
csv_b = os.path.join(
    EVAL_PARAMS["table_dir"], "block16b_alignment_ablation.csv"
)
df_b.to_csv(csv_b, index=False)

summary_b_path = os.path.join(
    EVAL_PARAMS["table_dir"], "block16b_summary.csv"
)
summary_b.to_csv(summary_b_path, index=False)

ablation_metrics = {
    "block_16_ttt_gain": round(
        float(df_final['delta_ttt_vs_crattt'].mean()), 4
    ),
    "block_16b_ttt_gain": round(
        float(df_b['delta_ttt_vs_crattt'].mean()), 4
    ),
    "loss_function_b16":  "compute_pseudo_label_loss",
    "loss_function_b16b": "compute_alignment_loss",
    "finding": (
        "Alignment loss directly targets DINO-CLIP cosine "
        "similarity, addressing the frozen Oracle dominance "
        "identified in Block 16"
    )
}

ablation_path = os.path.join(
    EVAL_PARAMS["save_dir"], "ablation_ttt_loss.json"
)
with open(ablation_path, "w") as f:
    json.dump(ablation_metrics, f, indent=2)

print(f"\n✅ Results saved        : {csv_b}")
print(f"✅ Ablation metrics saved: {ablation_path}")

print("\n" + "="*50)
print("BLOCK 16b COMPLETE — Alignment loss ablation done")
print("="*50)
