# ============================================================
# BLOCK 18: Encoder-Feature Gate (Option B)
# Replaces BARON crop-based Soracle with a gate that operates
# directly on DINO encoder hidden states.
#
# Key insight: Previous blocks (16, 16b, 16c, 17) produced
# zero TTT gain because Soracle was computed from region crops
# whose coordinates come from the FROZEN detection head.
# Box coordinates never changed, crops never changed, Soracle
# never changed.
#
# Option B fix: Compute Soracle by measuring cosine similarity
# between DINO's encoder hidden states (which LoRA modifies)
# and CLIP text embeddings for proposed classes.
# The gate now operates in the same feature space that LoRA
# modifies, guaranteeing gate sensitivity by construction.
#
# NOTE: the projection used here is shared between the gate
# (compute_encoder_soracle) and the TTT loss
# (compute_encoder_ttt_loss) — both reference the same cached
# instance, which is what makes "same feature space" literally
# true. It's still a fixed random rotation, never trained, just
# consistently fixed across both paths.
#
# Scope: 5 images, 4 corruptions, severity 5, 10 steps
# Cost: Free (Kaggle T4)
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import os
import math
import gc
from imagecorruptions import corrupt as ic_corrupt
from tqdm.notebook import tqdm
from IPython.display import display
from transformers import AutoModelForZeroShotObjectDetection

print("="*50)
print("BLOCK 18: ENCODER-FEATURE GATE (OPTION B)")
print("="*50)
print()

# --- 18.1 Confirm (or restore) clean rank=4 LoRA state ---
for param in clip_model.parameters():
    param.requires_grad = False
print("✅ CLIP fully frozen")

class LoRALinear18(nn.Module):
    def __init__(self, linear_layer, rank=4, lora_alpha=8):
        super().__init__()
        self.in_features  = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rank         = rank
        self.scale        = lora_alpha / rank
        self.weight = nn.Parameter(
            linear_layer.weight.data.clone(),
            requires_grad=False
        )
        self.bias = nn.Parameter(
            linear_layer.bias.data.clone(),
            requires_grad=False
        ) if linear_layer.bias is not None else None
        self.lora_A = nn.Parameter(
            torch.zeros(rank, self.in_features)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_features, rank)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base  = nn.functional.linear(x, self.weight, self.bias)
        delta = (x @ self.lora_A.T @ self.lora_B.T) * self.scale
        return base + delta


def inject_lora_b18(model, rank=4, lora_alpha=8):
    """Only matches nn.Linear — see the state check below for why
    this is only called on a confirmed-clean model."""
    n_injected = 0
    for name, module in list(model.named_modules()):
        if not (name.startswith('model.encoder') or
                name.startswith('model.decoder')):
            continue
        if not (name.endswith('.query') or
                name.endswith('.value')):
            continue
        if not isinstance(module, nn.Linear):
            continue
        parts  = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        lora_layer = LoRALinear18(
            module, rank=rank, lora_alpha=lora_alpha
        ).to(device)
        setattr(parent, parts[-1], lora_layer)
        n_injected += 1
    for param in model.parameters():
        param.requires_grad = False
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            param.requires_grad = True
            lora_params.append((name, param))
    return n_injected, lora_params


# FIX: check actual current state instead of assuming re-injection
# is needed. inject_lora_b18 only matches nn.Linear, so calling it
# on an already-wrapped model (the case here, carried from Block
# 14/17) silently injects 0 layers while still printing a correct-
# looking trainable-param count.
existing_lora_b = sum(
    1 for n, _ in dino_model.named_parameters() if 'lora_B' in n
)
existing_trainable = sum(
    p.numel() for p in dino_model.parameters() if p.requires_grad
)

if existing_lora_b == 36 and existing_trainable == 73_728:
    print("✅ dino_model already at clean rank=4 LoRA "
          "(reusing existing state, no re-injection needed)")
    n_inj = 36
else:
    print(f"   Current state: {existing_lora_b} lora_B matrices, "
          f"{existing_trainable:,} trainable params — not rank=4, "
          f"reloading fresh before injecting.")
    del dino_model
    gc.collect()
    torch.cuda.empty_cache()
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        DINO_MODEL_ID, token=hf_token
    ).to(device)
    dino_model.eval()
    n_inj, _ = inject_lora_b18(dino_model, rank=4, lora_alpha=8)

assert n_inj == 36, f"Expected 36 LoRA layers, got {n_inj}"
dino_trainable = sum(
    p.numel() for p in dino_model.parameters()
    if p.requires_grad
)
assert dino_trainable == 73_728, (
    f"Expected 73,728 trainable DINO params, got {dino_trainable:,}"
)
print(f"✅ DINO LoRA rank=4 confirmed ({n_inj} layers)")
print(f"   Trainable params: {dino_trainable:,} (0.0428%)")

# FIX: explicitly zero lora_B here, before the 18.4 clean-image
# check below. Without this, that diagnostic runs on whatever
# adaptation state Block 17's last corruption left behind, not a
# genuine clean baseline.
for name, param in dino_model.named_parameters():
    if 'lora_B' in name:
        param.data.zero_()
print("✅ LoRA B zeroed — clean state confirmed before diagnostics")

# Verify forward pass
test_img = loaded_images[image_files[0]]
inputs_v = dino_processor(
    images=test_img, text=DINO_TEXT_PROMPT,
    return_tensors="pt"
).to(device)
with torch.no_grad():
    out_v = dino_model(**inputs_v)
res_v = dino_processor.post_process_grounded_object_detection(
    out_v, inputs_v.input_ids,
    target_sizes=[test_img.shape[:2]],
    text_threshold=CRATTT_PARAMS["dino_text_thr"]
)[0]
print(f"✅ Forward pass OK — {len(res_v['boxes'])} detections")

# --- 18.2 Encoder-Feature Soracle ---
def compute_encoder_soracle(dino_outputs, proposed_labels,
                              clip_text_features, COCO_CLASSES,
                              device):
    """
    Computes Soracle directly from DINO encoder hidden states.
    """
    if hasattr(dino_outputs, 'encoder_last_hidden_state') and \
       dino_outputs.encoder_last_hidden_state is not None:
        enc_hidden = dino_outputs.encoder_last_hidden_state
    elif hasattr(dino_outputs, 'last_hidden_state') and \
         dino_outputs.last_hidden_state is not None:
        enc_hidden = dino_outputs.last_hidden_state
    else:
        return {}, None

    encoder_feat = enc_hidden.mean(dim=1)
    hidden_dim   = encoder_feat.shape[-1]
    clip_dim     = clip_text_features.shape[-1]

    proj_attr = f"_b18_proj_{hidden_dim}_{clip_dim}"
    if not hasattr(compute_encoder_soracle, proj_attr):
        proj = nn.Linear(hidden_dim, clip_dim, bias=False).to(device)
        nn.init.orthogonal_(proj.weight)
        setattr(compute_encoder_soracle, proj_attr, proj)
    proj         = getattr(compute_encoder_soracle, proj_attr)
    encoder_proj = proj(encoder_feat)
    encoder_norm = F.normalize(encoder_proj, p=2, dim=-1)

    soracle_scores = {}
    for label in proposed_labels:
        if isinstance(label, str):
            clean = label.lower().replace(".", "").strip()
            if clean in COCO_CLASSES:
                idx       = COCO_CLASSES.index(clean)
                text_feat = clip_text_features[idx:idx+1]
                text_norm = F.normalize(text_feat, p=2, dim=-1)
                sim       = (encoder_norm * text_norm).sum().item()
                soracle_scores[clean] = sim

    return soracle_scores, encoder_feat


# --- 18.3 Encoder-Feature CRATTT Inference ---
def run_crattt_encoder_gate(image_np, dino_outputs=None):
    if dino_outputs is None:
        inputs = dino_processor(
            images=image_np,
            text=DINO_TEXT_PROMPT,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            dino_outputs = dino_model(**inputs)
        inputs_ref = inputs
    else:
        inputs_ref = dino_processor(
            images=image_np,
            text=DINO_TEXT_PROMPT,
            return_tensors="pt"
        ).to(device)

    res = dino_processor.post_process_grounded_object_detection(
        dino_outputs,
        inputs_ref.input_ids,
        target_sizes=[image_np.shape[:2]],
        text_threshold=CRATTT_PARAMS["dino_text_thr"]
    )[0]

    boxes  = res["boxes"]
    scores = res["scores"]
    labels = res.get("text_labels", res.get("labels", []))

    if len(boxes) == 0:
        return [], {"n_proposals": 0, "n_verified": 0,
                    "n_rejected": 0}

    proposed = []
    for lbl in labels:
        if isinstance(lbl, str):
            proposed.append(lbl.lower().replace(".", "").strip())

    soracle_scores, _ = compute_encoder_soracle(
        dino_outputs, proposed,
        clip_text_features, COCO_CLASSES, device
    )

    verified  = []
    n_rejected = 0

    for i, box in enumerate(boxes):
        dino_score = scores[i].item()

        if i < len(labels):
            label = labels[i]
        else:
            continue

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

        soracle = soracle_scores.get(clean_label, 0.0)

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
        "n_proposals": len(boxes),
        "n_verified":  len(verified),
        "n_rejected":  n_rejected
    }
    return verified, stats


# --- 18.4 Verify Encoder Gate on Clean Image ---
# (now running on a confirmed-clean LoRA state, see 18.1's fix)
print("\nValidating encoder-feature gate on clean image...")
with torch.no_grad():
    clean_inputs = dino_processor(
        images=test_img, text=DINO_TEXT_PROMPT,
        return_tensors="pt"
    ).to(device)
    clean_outputs = dino_model(**clean_inputs)

verified_clean, stats_clean = run_crattt_encoder_gate(
    test_img, clean_outputs
)

# FIX: compute the original crop-gate's count on THIS run's clean
# image live, instead of printing a hardcoded "10" left over from
# a different run on a different backbone.
crop_gate_clean_preds, crop_gate_clean_stats = run_crattt_inference(test_img)
print(f"✅ Encoder gate: {stats_clean['n_verified']} verified, "
      f"{stats_clean['n_rejected']} rejected")
print(f"   (Original BARON crop gate, this run: "
      f"{crop_gate_clean_stats['n_verified']} verified — "
      f"for reference, not a strict comparison since the gates "
      f"score fundamentally different things)")

sample_labels = ['person', 'car', 'chair', 'tv']
sample_scores, _ = compute_encoder_soracle(
    clean_outputs, sample_labels,
    clip_text_features, COCO_CLASSES, device
)
print(f"\n   Sample encoder Soracle scores:")
for lbl, sc in sample_scores.items():
    print(f"   {lbl:<20}: {sc:.4f}")

# --- 18.5 TTT Loss using Encoder Features ---
def compute_encoder_ttt_loss(dino_model, dino_processor,
                              image_np, verified_preds,
                              clip_text_features, COCO_CLASSES,
                              device):
    if not verified_preds:
        return None

    inputs = dino_processor(
        images=image_np,
        text=DINO_TEXT_PROMPT,
        return_tensors="pt"
    ).to(device)

    outputs = dino_model(**inputs)

    if hasattr(outputs, 'encoder_last_hidden_state') and \
       outputs.encoder_last_hidden_state is not None:
        enc_hidden = outputs.encoder_last_hidden_state
    elif hasattr(outputs, 'last_hidden_state') and \
         outputs.last_hidden_state is not None:
        enc_hidden = outputs.last_hidden_state
    else:
        return None

    encoder_feat = enc_hidden.mean(dim=1)
    hidden_dim   = encoder_feat.shape[-1]
    clip_dim     = clip_text_features.shape[-1]

    # Same cached projection as compute_encoder_soracle —
    # this is what keeps the loss and the gate in the same space.
    proj_attr = f"_b18_proj_{hidden_dim}_{clip_dim}"
    if not hasattr(compute_encoder_soracle, proj_attr):
        proj = nn.Linear(hidden_dim, clip_dim, bias=False).to(device)
        nn.init.orthogonal_(proj.weight)
        setattr(compute_encoder_soracle, proj_attr, proj)
    proj         = getattr(compute_encoder_soracle, proj_attr)
    encoder_proj = proj(encoder_feat)
    encoder_norm = F.normalize(encoder_proj, p=2, dim=-1)

    target_embs = []
    for pred in verified_preds:
        label = pred.get('label', '')
        if label in COCO_CLASSES:
            idx = COCO_CLASSES.index(label)
            target_embs.append(clip_text_features[idx])

    if not target_embs:
        return None

    target      = torch.stack(target_embs).mean(dim=0, keepdim=True)
    target_norm = F.normalize(target, p=2, dim=-1)

    similarity = (encoder_norm * target_norm).sum(dim=-1)
    loss       = (1.0 - similarity).mean()
    return loss


# --- 18.6 Configuration ---
PILOT_IMAGES_18    = image_files[:5]
EVAL_CORRUPTIONS_18 = [
    ("Noise",   "gaussian_noise", 5),
    ("Blur",    "motion_blur",    5),
    ("Weather", "snow",           5),
    ("Digital", "contrast",       5),
]
TTT_STEPS_18 = 10
TTT_LR_18    = 1e-4

print(f"\n--- Block 18 Configuration ---")
print(f"Gate type   : Encoder-feature Soracle (Option B)")
print(f"Images      : {len(PILOT_IMAGES_18)}")
print(f"Corruptions : {len(EVAL_CORRUPTIONS_18)}")
print(f"TTT steps   : {TTT_STEPS_18}")
print(f"TTT lr      : {TTT_LR_18}")
print(f"Tau         : {CRATTT_PARAMS['tau']}")

# --- 18.7 Main Loop ---
all_rows_18 = []

for cat_name, corruption, severity in EVAL_CORRUPTIONS_18:
    print(f"\n▶  {cat_name}: {corruption} sev{severity}")

    for name, param in dino_model.named_parameters():
        if 'lora_B' in name:
            param.data.zero_()

    for attr in list(vars(compute_encoder_soracle).keys()):
        if attr.startswith('_b18_proj_'):
            delattr(compute_encoder_soracle, attr)

    optimizer_18 = torch.optim.AdamW(
        [p for p in dino_model.parameters() if p.requires_grad],
        lr=TTT_LR_18,
        weight_decay=1e-4
    )

    corruption_rows = []
    pbar = tqdm(
        PILOT_IMAGES_18, desc=f"  {corruption}", leave=False
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

        # === CONDITION B: CRATTT with encoder gate (no TTT) ===
        with torch.no_grad():
            inputs_b = dino_processor(
                images=c_img,
                text=DINO_TEXT_PROMPT,
                return_tensors="pt"
            ).to(device)
            outputs_b = dino_model(**inputs_b)

        crattt_preds_b, stats_b = run_crattt_encoder_gate(
            c_img, outputs_b
        )
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

        # === CONDITION C: Encoder Gate + TTT ===
        loss_values = []
        dino_model.train()

        for step in range(TTT_STEPS_18):
            optimizer_18.zero_grad()
            loss = compute_encoder_ttt_loss(
                dino_model, dino_processor,
                c_img, crattt_preds_b,
                clip_text_features, COCO_CLASSES, device
            )
            if loss is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in dino_model.parameters()
                     if p.requires_grad],
                    max_norm=1.0
                )
                optimizer_18.step()
                loss_values.append(round(loss.item(), 6))

        dino_model.eval()

        with torch.no_grad():
            inputs_c = dino_processor(
                images=c_img,
                text=DINO_TEXT_PROMPT,
                return_tensors="pt"
            ).to(device)
            outputs_c = dino_model(**inputs_c)

        crattt_preds_c, stats_c = run_crattt_encoder_gate(
            c_img, outputs_c
        )
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
            "category":               cat_name,
            "corruption":             corruption,
            "severity":               severity,
            "image":                  fname,
            "baseline_mAP":           round(float(bmap),   4),
            "encoder_crattt_mAP":     round(float(cmap_b), 4),
            "encoder_ttt_mAP":        round(float(cmap_c), 4),
            "delta_encoder_crattt":   round(float(cmap_b - bmap), 4),
            "delta_encoder_ttt":      round(float(cmap_c - bmap), 4),
            "delta_ttt_vs_crattt":    round(float(cmap_c - cmap_b), 4),
            "n_baseline":             len(baseline_res['boxes']),
            "n_encoder_crattt":       stats_b['n_verified'],
            "n_encoder_ttt":          stats_c['n_verified'],
            "loss_trajectory":        loss_values
        }
        corruption_rows.append(row)

        pbar.set_postfix({
            "B":  f"{bmap:.3f}",
            "EG": f"{cmap_b:.3f}",
            "T":  f"{cmap_c:.3f}"
        })

    all_rows_18.extend(corruption_rows)

    c_df = pd.DataFrame(corruption_rows)
    print(f"  Baseline mAP          : "
          f"{c_df['baseline_mAP'].mean():.4f}")
    print(f"  Encoder Gate mAP      : "
          f"{c_df['encoder_crattt_mAP'].mean():.4f}")
    print(f"  Encoder Gate+TTT mAP  : "
          f"{c_df['encoder_ttt_mAP'].mean():.4f}")
    print(f"  TTT gain vs gate      : "
          f"{c_df['delta_ttt_vs_crattt'].mean():+.4f}")

    sample = next(
        (r['loss_trajectory'] for r in corruption_rows
         if r['loss_trajectory']), []
    )
    if sample:
        print(f"  Loss (first 5 steps)  : {sample[:5]}")

# --- 18.8 Results ---
df_18 = pd.DataFrame(all_rows_18)

print("\n" + "="*70)
print("TABLE: BLOCK 18 ENCODER-FEATURE GATE RESULTS")
print("="*70)

summary_18 = df_18.groupby(
    ["category", "corruption"]
).agg(
    Baseline_mAP        =("baseline_mAP",        "mean"),
    Encoder_Gate_mAP    =("encoder_crattt_mAP",   "mean"),
    Encoder_TTT_mAP     =("encoder_ttt_mAP",      "mean"),
    TTT_gain_vs_gate    =("delta_ttt_vs_crattt",   "mean"),
).round(4).reset_index()

display(summary_18)

overall_gain_18 = df_18['delta_ttt_vs_crattt'].mean()
print(f"\nOverall TTT gain vs encoder gate: {overall_gain_18:+.4f}")

# --- 18.9 Complete Ablation Table ---
# FIXED: reads each prior block's gain live from memory if
# available, falling back to its saved CSV on disk.
def _get_gain(varname, csv_path, csv_col="delta_ttt_vs_crattt"):
    if varname in globals():
        return globals()[varname][csv_col].mean(), "live"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)[csv_col].mean(), "from disk"
    return None, "unavailable"

gain_16, _  = _get_gain("df_final", os.path.join(EVAL_PARAMS["table_dir"], "table_4_7_end_to_end.csv"))
gain_16b, _ = _get_gain("df_b",     os.path.join(EVAL_PARAMS["table_dir"], "block16b_alignment_ablation.csv"))
gain_16c, _ = _get_gain("df_16c",   os.path.join(EVAL_PARAMS["table_dir"], "block16c_rank16_ttt.csv"))
gain_17, _  = _get_gain("df_17",    os.path.join(EVAL_PARAMS["table_dir"], "table_block17_coadapt_pilot.csv"),
                         csv_col="delta_coadapt_vs_crattt")

print(f"\n{'='*68}")
print("COMPLETE ABLATION: All TTT Configurations Tested (Swin-B)")
print(f"{'='*68}")
print(f"{'Configuration':<46} {'TTT gain':>10} {'Gate type':>10}")
print("-"*68)

def _fmt(label, gain, gate):
    g = f"{gain:+.4f}" if gain is not None else "N/A"
    print(f"{label:<46} {g:>10} {gate:>10}")

_fmt("Blk 16  rank=4  conf-loss   crop-gate  frozen",  gain_16,  "crop")
_fmt("Blk 16b rank=4  align-loss  crop-gate  frozen",  gain_16b, "crop")
_fmt("Blk 16c rank=16 align-loss  crop-gate  frozen",  gain_16c, "crop")
_fmt("Blk 17  rank=4  align-loss  crop-gate  coadapt", gain_17,  "crop")
_fmt("Blk 18  rank=4  enc-loss    enc-gate   frozen",  overall_gain_18, "encoder")
print()

if overall_gain_18 > 0.005:
    verdict = "✅ POSITIVE GAIN — Encoder gate works!"
    action  = "Proceed to Vast.ai for full-scale evaluation."
    conclusion = "positive"
elif overall_gain_18 > 0.001:
    verdict = "✅ MARGINAL POSITIVE — Promising signal"
    action  = "Try 20 steps before Vast.ai."
    conclusion = "marginal_positive"
elif overall_gain_18 > -0.001:
    verdict = "⚠️  NEAR-ZERO — No gate improvement"
    action  = "Proceed to Option A (detection head LoRA)."
    conclusion = "near_zero"
else:
    verdict = "❌ NEGATIVE — TTT constraint is fundamental"
    action  = ("Consider Option A before concluding. "
                "Report as architectural limitation if A also fails.")
    conclusion = "negative"

print(verdict)
print(f"Action: {action}")

# --- 18.10 Save ---
csv_18 = os.path.join(
    EVAL_PARAMS["table_dir"], "table_block18_encoder_gate.csv"
)
df_18.to_csv(csv_18, index=False)

record_18 = {
    "gate_type":      "encoder_feature",
    "loss_type":      "encoder_alignment",
    "rank":           4,
    "ttt_steps":      TTT_STEPS_18,
    "lr":             TTT_LR_18,
    "n_images":       len(PILOT_IMAGES_18),
    "overall_gain":   round(float(overall_gain_18), 4),
    "conclusion":     conclusion,
    "next_action":    action,
    "ablation_summary": {
        "block_16_frozen_conf":   round(float(gain_16), 4)  if gain_16  is not None else None,
        "block_16b_frozen_align": round(float(gain_16b), 4) if gain_16b is not None else None,
        "block_16c_rank16_align": round(float(gain_16c), 4) if gain_16c is not None else None,
        "block_17_coadapt_align": round(float(gain_17), 4)  if gain_17  is not None else None,
        "block_18_encoder_gate":  round(float(overall_gain_18), 4)
    }
}
with open(os.path.join(
    EVAL_PARAMS["save_dir"], "block18_encoder_gate.json"
), "w") as f:
    json.dump(record_18, f, indent=2)

print(f"\n✅ Results saved: {csv_18}")
print("\n" + "="*50)
print("BLOCK 18 COMPLETE")
print("="*50)
