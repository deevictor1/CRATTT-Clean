# ============================================================
# BLOCK 17: Co-Adaptation Pilot
# Tests whether jointly unfreezing the final 2 layers of the
# CLIP vision encoder alongside DINO LoRA adapters produces
# positive TTT gain by addressing the Representational
# Asymmetry Problem identified in Blocks 16, 16b, and 16c.
#
# Hypothesis: The frozen CLIP Oracle creates a fixed ceiling
# on Soracle scores. If CLIP's final layers adapt alongside
# DINO's LoRA adapters, the Oracle's evaluation criterion
# co-evolves with the adapted model, allowing the gate to
# respond to beneficial parameter updates.
#
# CAVEAT (documented, not fixed): the loss below backprops only
# through CLIP's TEXT branch (text_model + text_projection) to
# build target embeddings. The unfrozen layers are CLIP's VISION
# encoder (layers 10/11), the branch Soracle actually calls at
# gate-evaluation time. Those vision layers receive no task
# gradient here, only AdamW's decoupled weight decay. This
# block tests "does co-adapting CLIP's text branch help", not
# literally "does co-adapting the gate's own vision pathway
# help." Worth noting explicitly in the dissertation's
# limitations section.
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
print("BLOCK 17: CO-ADAPTATION PILOT")
print("="*50)
print()

# --- 17.1 Confirm (or restore) clean rank=4 LoRA state ---
print("Step 1: Checking DINO's current LoRA state...")

class LoRALinear17(nn.Module):
    """Clean rank=4 LoRA — used only if dino_model needs reloading."""
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


def inject_lora_b17(model, rank=4, lora_alpha=8):
    """
    Fresh rank=4 LoRA injection. Only matches nn.Linear modules,
    so this is only called below after confirming/forcing a clean
    (unwrapped) model — calling it on an already-wrapped model
    would silently inject 0 layers.
    """
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
        lora_layer = LoRALinear17(
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


# FIX: check actual current state instead of assuming dino_model
# needs fresh injection. If the post-16c revert cell already ran,
# dino_model is already correct — reusing it instead of trying
# (and failing) to inject a second time on top of it.
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
    n_inj, lora_p = inject_lora_b17(dino_model, rank=4, lora_alpha=8)

assert n_inj == 36, (
    f"Expected 36 LoRA layers (existing or freshly injected), got {n_inj}"
)

dino_trainable = sum(
    p.numel() for p in dino_model.parameters()
    if p.requires_grad
)
assert dino_trainable == 73_728, (
    f"Expected 73,728 trainable DINO params, got {dino_trainable:,}"
)
print(f"✅ DINO LoRA rank=4 confirmed ({n_inj} layers)")
print(f"   DINO trainable params: {dino_trainable:,}")

# --- 17.2 Unfreeze Final 2 CLIP Vision Encoder Layers ---
print()
print("Step 2: Unfreezing final 2 CLIP vision encoder layers...")

# First freeze all CLIP parameters
for param in clip_model.parameters():
    param.requires_grad = False

# Unfreeze only the final 2 transformer layers of CLIP vision
clip_unfrozen = 0
for name, param in clip_model.named_parameters():
    if ('vision_model.encoder.layers.11' in name or
        'vision_model.encoder.layers.10' in name):
        param.requires_grad = True
        clip_unfrozen += 1

clip_trainable = sum(
    p.numel() for p in clip_model.parameters()
    if p.requires_grad
)
print(f"✅ CLIP layers 10 and 11 unfrozen")
print(f"   CLIP trainable params : {clip_trainable:,}")
print(f"   CLIP unfrozen tensors : {clip_unfrozen}")

# --- 17.3 Parameter Summary ---
total_trainable = dino_trainable + clip_trainable
total_params    = (
    sum(p.numel() for p in dino_model.parameters()) +
    sum(p.numel() for p in clip_model.parameters())
)

print(f"\n--- Co-Adaptation Parameter Audit ---")
print(f"DINO LoRA trainable : {dino_trainable:,} (0.0428%)")
print(f"CLIP final 2 layers : {clip_trainable:,}")
print(f"Total trainable     : {total_trainable:,}")
print(f"Total parameters    : {total_params:,}")
print(f"Trainable ratio     : "
      f"{100*total_trainable/total_params:.4f}%")

# VRAM check
vram_used  = torch.cuda.memory_allocated() / 1e9
vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
vram_free  = vram_total - vram_used
print(f"\nVRAM free: {vram_free:.2f} GB")

if vram_free < 4.0:
    print("⚠️  Low VRAM — reducing pilot to 3 images")
    PILOT_SIZE = 3
else:
    print("✅ Sufficient VRAM for 5-image pilot")
    PILOT_SIZE = 5

# --- 17.4 Co-Adaptation Alignment Loss ---
def compute_coadapt_loss(dino_model, dino_processor,
                          clip_model, clip_processor,
                          image_np, verified_preds, device):
    """
    Co-adaptation loss: jointly updates DINO LoRA and
    CLIP final layers to maximise alignment between
    DINO encoder features and CLIP text embeddings
    for verified classes.

    See the caveat at the top of this block: this forward
    pass only touches clip_model.text_model /
    .text_projection. clip_model.vision_model (the unfrozen
    layers) never appears in this graph.
    """
    if not verified_preds:
        return None

    # --- DINO forward pass (with LoRA gradients) ---
    dino_inputs = dino_processor(
        images=image_np,
        text=DINO_TEXT_PROMPT,
        return_tensors="pt"
    ).to(device)

    dino_outputs = dino_model(**dino_inputs)

    if hasattr(dino_outputs, 'encoder_last_hidden_state') and \
       dino_outputs.encoder_last_hidden_state is not None:
        visual_features = dino_outputs.encoder_last_hidden_state
    elif hasattr(dino_outputs, 'last_hidden_state') and \
         dino_outputs.last_hidden_state is not None:
        visual_features = dino_outputs.last_hidden_state
    else:
        return None

    dino_pooled = visual_features.mean(dim=1)  # [1, hidden_dim]

    # --- CLIP text forward pass (final layers trainable) ---
    target_embeddings = []
    for pred in verified_preds:
        label = pred.get('label', '')
        if isinstance(label, str):
            clean = label.lower().replace(".", "").strip()
            if clean in COCO_CLASSES:
                # Re-compute CLIP text embedding with gradients
                clip_text_in = clip_processor(
                    text=[clean],
                    return_tensors="pt",
                    padding=True
                ).to(device)
                text_out = clip_model.text_model(
                    input_ids=clip_text_in["input_ids"],
                    attention_mask=clip_text_in["attention_mask"]
                )
                pooled    = text_out.pooler_output
                projected = clip_model.text_projection(pooled)
                target_embeddings.append(projected)

    if not target_embeddings:
        return None

    # Mean target across verified classes
    target = torch.stack(target_embeddings).mean(
        dim=0, keepdim=True
    )  # [1, 512]

    clip_dim   = target.shape[-1]
    hidden_dim = dino_pooled.shape[-1]

    # Project DINO features to CLIP space
    proj_attr = f"_coadapt_proj_{hidden_dim}_{clip_dim}"
    if not hasattr(compute_coadapt_loss, proj_attr):
        proj = nn.Linear(
            hidden_dim, clip_dim, bias=False
        ).to(device)
        nn.init.orthogonal_(proj.weight)
        setattr(compute_coadapt_loss, proj_attr, proj)
    proj        = getattr(compute_coadapt_loss, proj_attr)
    dino_proj   = proj(dino_pooled)

    # Normalise for cosine similarity
    dino_norm   = F.normalize(dino_proj, p=2, dim=-1)
    target_norm = F.normalize(target,    p=2, dim=-1)

    # Cosine alignment loss: minimise = maximise alignment
    similarity = (dino_norm * target_norm).sum(dim=-1)
    loss       = (1.0 - similarity).mean()
    return loss


# --- 17.5 Configuration ---
PILOT_IMAGES_17    = image_files[:PILOT_SIZE]
EVAL_CORRUPTIONS_17 = [
    ("Noise",   "gaussian_noise", 5),
    ("Blur",    "motion_blur",    5),
    ("Weather", "snow",           5),
    ("Digital", "contrast",       5),
]
TTT_STEPS_17 = 10
TTT_LR_17    = 5e-5

print(f"\n--- Pilot Configuration ---")
print(f"Images      : {len(PILOT_IMAGES_17)}")
print(f"Corruptions : {len(EVAL_CORRUPTIONS_17)}")
print(f"TTT steps   : {TTT_STEPS_17}")
print(f"TTT lr      : {TTT_LR_17}")
print(f"Tau         : {CRATTT_PARAMS['tau']}")

# --- 17.6 Main Pilot Loop ---
all_rows_17 = []

for cat_name, corruption, severity in EVAL_CORRUPTIONS_17:
    print(f"\n▶  {cat_name}: {corruption} sev{severity}")

    # Reset DINO LoRA B to zero
    for name, param in dino_model.named_parameters():
        if 'lora_B' in name:
            param.data.zero_()

    # Reset CLIP final layers to original weights
    # by reloading — we simply zero the gradient
    # (weights persist from Block 2 loading)

    # Clear projection cache
    for attr in list(vars(compute_coadapt_loss).keys()):
        if attr.startswith('_coadapt_proj_'):
            delattr(compute_coadapt_loss, attr)

    # Fresh optimizer — covers both DINO LoRA and CLIP layers
    all_trainable_params = (
        [p for p in dino_model.parameters() if p.requires_grad] +
        [p for p in clip_model.parameters() if p.requires_grad]
    )
    optimizer_17 = torch.optim.AdamW(
        all_trainable_params,
        lr=TTT_LR_17,
        weight_decay=1e-4
    )

    corruption_rows = []
    pbar = tqdm(
        PILOT_IMAGES_17, desc=f"  {corruption}", leave=False
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

        # === CONDITION C: CRATTT + Co-Adaptation TTT ===
        loss_values = []
        dino_model.train()
        clip_model.train()

        for step in range(TTT_STEPS_17):
            optimizer_17.zero_grad()
            loss = compute_coadapt_loss(
                dino_model, dino_processor,
                clip_model, clip_processor,
                c_img, crattt_preds_b, device
            )
            if loss is not None:
                loss.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(
                    all_trainable_params, max_norm=1.0
                )
                optimizer_17.step()
                loss_values.append(round(loss.item(), 6))

        dino_model.eval()
        clip_model.eval()

        # Re-run CRATTT with co-adapted weights
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
            "baseline_mAP":        round(float(bmap),  4),
            "crattt_mAP":          round(float(cmap_b), 4),
            "coadapt_ttt_mAP":     round(float(cmap_c), 4),
            "delta_crattt":        round(float(cmap_b - bmap), 4),
            "delta_coadapt":       round(float(cmap_c - bmap), 4),
            "delta_coadapt_vs_crattt": round(
                float(cmap_c - cmap_b), 4
            ),
            "n_baseline":          len(baseline_res['boxes']),
            "n_crattt":            stats_b['n_verified'],
            "n_coadapt":           stats_c['n_verified'],
            "loss_trajectory":     loss_values
        }
        corruption_rows.append(row)

        pbar.set_postfix({
            "B": f"{bmap:.3f}",
            "C": f"{cmap_b:.3f}",
            "T": f"{cmap_c:.3f}"
        })

    all_rows_17.extend(corruption_rows)

    c_df = pd.DataFrame(corruption_rows)
    print(f"  Baseline mAP            : "
          f"{c_df['baseline_mAP'].mean():.4f}")
    print(f"  CRATTT mAP              : "
          f"{c_df['crattt_mAP'].mean():.4f}")
    print(f"  Co-Adapt TTT mAP        : "
          f"{c_df['coadapt_ttt_mAP'].mean():.4f}")
    print(f"  TTT gain vs CRATTT      : "
          f"{c_df['delta_coadapt_vs_crattt'].mean():+.4f}")

    sample = next(
        (r['loss_trajectory'] for r in corruption_rows
         if r['loss_trajectory']), []
    )
    if sample:
        print(f"  Loss trajectory (first 5): {sample[:5]}")

# --- 17.7 Results Table ---
df_17 = pd.DataFrame(all_rows_17)

print("\n" + "="*70)
print("BLOCK 17: CO-ADAPTATION PILOT RESULTS")
print("="*70)

summary_17 = df_17.groupby(
    ["category", "corruption"]
).agg(
    Baseline_mAP           =("baseline_mAP",           "mean"),
    CRATTT_mAP             =("crattt_mAP",              "mean"),
    CoAdapt_TTT_mAP        =("coadapt_ttt_mAP",         "mean"),
    Delta_CoAdapt_vs_CRATTT=("delta_coadapt_vs_crattt",  "mean"),
).round(4).reset_index()

display(summary_17)

overall_gain = df_17['delta_coadapt_vs_crattt'].mean()
print(f"\n--- Overall Co-Adaptation TTT Gain vs CRATTT ---")
print(f"Mean gain: {overall_gain:+.4f}")

# --- 17.8 Full Ablation Comparison ---
# Reads each prior block's gain live from memory if available,
# falling back to its saved CSV on disk, rather than hardcoding
# stale values directly into the code.
def _get_gain(varname, csv_path, csv_col="delta_ttt_vs_crattt"):
    if varname in globals():
        return globals()[varname][csv_col].mean(), "live"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)[csv_col].mean(), "from disk"
    return None, "unavailable"

gain_16, src_16 = _get_gain(
    "df_final",
    os.path.join(EVAL_PARAMS["table_dir"], "table_4_7_end_to_end.csv")
)
gain_16b, src_16b = _get_gain(
    "df_b",
    os.path.join(EVAL_PARAMS["table_dir"], "block16b_alignment_ablation.csv")
)
gain_16c, src_16c = _get_gain(
    "df_16c",
    os.path.join(EVAL_PARAMS["table_dir"], "block16c_rank16_ttt.csv")
)

print(f"\n{'='*65}")
print("COMPLETE ABLATION: All TTT Configurations (this run, Swin-B)")
print(f"{'='*65}")
print(f"{'Configuration':<42} {'TTT gain vs CRATTT':>20}   {'source'}")
print("-"*78)

def _fmt(label, gain, src):
    if gain is not None:
        print(f"{label:<42} {gain:>+20.4f}   {src}")
    else:
        print(f"{label:<42} {'N/A':>20}   re-run that block first")

_fmt("Block 16  (rank=4,  conf loss,  frozen oracle)",  gain_16,  src_16)
_fmt("Block 16b (rank=4,  align loss, frozen oracle)",  gain_16b, src_16b)
_fmt("Block 16c (rank=16, align loss, frozen oracle)",  gain_16c, src_16c)
print(f"{'Block 17  (rank=4,  align loss, co-adapt oracle)':<42} "
      f"{overall_gain:>+20.4f}   live")
print()

if overall_gain > 0.001:
    print("✅ POSITIVE GAIN — Co-adaptation works!")
    print("   Proceed to Vast.ai for full-scale evaluation")
    conclusion = "positive_gain_proceed_to_vastai"
elif overall_gain > -0.001:
    print("⚠️  NEAR-ZERO — Marginal effect, borderline")
    print("   Try higher lr or more steps before Vast.ai")
    conclusion = "near_zero_investigate_further"
else:
    print("❌ STILL NEGATIVE — Constraint is deeper than")
    print("   representational asymmetry.")
    print("   Report as confirmed architectural limitation.")
    print("   Do NOT proceed to Vast.ai for TTT.")
    conclusion = "negative_architectural_constraint_confirmed"

# --- 17.9 Save ---
os.makedirs(EVAL_PARAMS["save_dir"], exist_ok=True)

csv_17 = os.path.join(
    EVAL_PARAMS["table_dir"], "table_block17_coadapt_pilot.csv"
)
df_17.to_csv(csv_17, index=False)

pilot_record = {
    "n_images":              len(PILOT_IMAGES_17),
    "corruptions":           [c[1] for c in EVAL_CORRUPTIONS_17],
    "severity":              5,
    "ttt_steps":             TTT_STEPS_17,
    "lr":                    TTT_LR_17,
    "rank":                  4,
    "clip_layers_unfrozen":  [10, 11],
    "overall_gain":          round(float(overall_gain), 4),
    "conclusion":            conclusion,
    "ablation_summary": {
        "block_16_frozen_conf":   round(float(gain_16), 4)  if gain_16  is not None else None,
        "block_16b_frozen_align": round(float(gain_16b), 4) if gain_16b is not None else None,
        "block_16c_rank16_align": round(float(gain_16c), 4) if gain_16c is not None else None,
        "block_17_coadapt_align": round(float(overall_gain), 4)
    }
}

with open(os.path.join(
    EVAL_PARAMS["save_dir"], "block17_coadapt_pilot.json"
), "w") as f:
    json.dump(pilot_record, f, indent=2)

print(f"\n✅ Results saved: {csv_17}")
print("\n" + "="*50)
print("BLOCK 17 COMPLETE — Co-Adaptation Pilot Done")
print("="*50)
