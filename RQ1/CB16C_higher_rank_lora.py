# ============================================================
# BLOCK 16c: Higher-Rank LoRA TTT (Ablation)
# Tests whether rank=16 with 10 gradient steps produces
# positive TTT gain, addressing the capacity constraint
# identified in Blocks 16 and 16b.
#
# Hypothesis: The zero TTT gain in Blocks 16 and 16b was
# a capacity constraint (rank=4, 3 steps) rather than a
# fundamental architectural impossibility. Rank=16 with
# 10 steps provides 4x more adapter capacity and 3x more
# gradient steps to bridge the representational gap.
#
# TABLE NOTE: feeds one row of Table 4.8 (the 12-row Block
# 15-23c ablation summary). Table 4.5 already belongs to
# Block 12's category-level comparative results.
#
# STATE NOTE: this block assumes NOTHING about dino_model's
# current LoRA state. It always reloads a fresh base model
# before injecting rank=16, and always reverts to rank=4 at
# the end so Block 17 onward starts from the correct baseline.
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

# --- 16c.1 LoRALinear for rank=16 ---
class LoRALinear(nn.Module):
    """
    LoRA-augmented Linear layer.
    Identical to Block 14 but parameterised for rank=16.
    B=0 init ensures no change to output at start of TTT.
    """
    def __init__(self, linear_layer, rank=16, lora_alpha=32):
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


# --- 16c.2 Injection Function ---
def inject_lora_rank16(model, rank=16, lora_alpha=32):
    """
    Injects rank=16 LoRA into GroundingDINO encoder+decoder
    query and value projections. Identical scope to Block 14
    but with 4x the adapter capacity.

    NOTE: only matches nn.Linear modules. If the model already
    has LoRALinear wrappers from a previous injection, those
    won't match and this becomes a silent no-op — which is why
    16c.4 below always reloads a clean model first.
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

        lora_layer = LoRALinear(
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


# --- 16c.3 Alignment Loss (same as Block 16b) ---
def compute_alignment_loss_16c(dino_model, dino_processor,
                                image_np, verified_preds, device):
    """
    Cross-modal alignment loss targeting cosine similarity
    between DINO encoder features and CLIP text embeddings.
    """
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
        visual_features = outputs.encoder_last_hidden_state
    elif hasattr(outputs, 'last_hidden_state') and \
         outputs.last_hidden_state is not None:
        visual_features = outputs.last_hidden_state
    else:
        return None

    pooled     = visual_features.mean(dim=1)
    hidden_dim = pooled.shape[-1]

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

    target   = torch.stack(target_embeddings).mean(
        dim=0, keepdim=True
    ).to(device)
    clip_dim = target.shape[-1]

    if hidden_dim != clip_dim:
        proj_attr = f"_proj_{hidden_dim}_{clip_dim}"
        if not hasattr(compute_alignment_loss_16c, proj_attr):
            proj = nn.Linear(hidden_dim, clip_dim, bias=False).to(device)
            nn.init.orthogonal_(proj.weight)
            setattr(compute_alignment_loss_16c, proj_attr, proj)
        proj   = getattr(compute_alignment_loss_16c, proj_attr)
        pooled = proj(pooled)

    pooled_norm = F.normalize(pooled, p=2, dim=-1)
    target_norm = F.normalize(target, p=2, dim=-1)
    similarity  = (pooled_norm * target_norm).sum(dim=-1)
    loss        = (1.0 - similarity).mean()
    return loss


# --- 16c.4 Reload clean model, then inject Rank=16 LoRA ---
LORA_RANK_16C  = 16
LORA_ALPHA_16C = 32
TTT_STEPS_16C  = 10
TTT_LR_16C     = 1e-4  # Slightly higher lr for rank=16

print("="*50)
print("BLOCK 16c: RANK=16 LoRA TTT ABLATION")
print("="*50)
print(f"Rank       : {LORA_RANK_16C}")
print(f"Alpha      : {LORA_ALPHA_16C}")
print(f"Scale      : {LORA_ALPHA_16C/LORA_RANK_16C:.2f}")
print(f"TTT steps  : {TTT_STEPS_16C}")
print(f"TTT lr     : {TTT_LR_16C}")
print(f"Tau        : {CRATTT_PARAMS['tau']}")
print()

# Always reload, regardless of dino_model's current state.
# Whatever LoRA config is currently active (rank=4, rank=16
# from a prior run, or none), this guarantees a clean slate.
print("Reloading fresh GroundingDINO-base (clean slate, no LoRA)...")
del dino_model
gc.collect()
torch.cuda.empty_cache()
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
    DINO_MODEL_ID, token=hf_token
).to(device)
dino_model.eval()

print("Injecting rank=16 LoRA...")
n_injected, lora_params = inject_lora_rank16(
    dino_model,
    rank=LORA_RANK_16C,
    lora_alpha=LORA_ALPHA_16C
)

total_params     = sum(p.numel() for p in dino_model.parameters())
trainable_params = sum(
    p.numel() for p in dino_model.parameters()
    if p.requires_grad
)

print(f"✅ LoRA layers injected : {n_injected}")
print(f"✅ Trainable parameters : {trainable_params:,}")
print(f"   Trainable ratio      : "
      f"{100*trainable_params/total_params:.4f}%")
print(f"   vs Block 14 rank=4   : 73,728 (0.0428%)")

assert n_injected == 36, (
    f"Expected 36 LoRA layers, got {n_injected} — "
    f"model was not reloaded clean before injection"
)
assert trainable_params == 294_912, (
    f"Expected 294,912 trainable params, got {trainable_params:,}"
)
print("✅ Injection verified — genuinely at rank=16, not silently rank=4")

# Forward pass verification
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

vram_used = torch.cuda.memory_allocated() / 1e9
vram_free = torch.cuda.get_device_properties(0).total_memory/1e9 - vram_used
print(f"   VRAM free: {vram_free:.2f} GB")

if vram_free < 2.5:
    print("⚠️  Low VRAM — reduce TTT_STEPS_16C to 5 if OOM")

# --- 16c.5 Configuration ---
EVAL_CORRUPTIONS_16C = [
    ("Noise",   "gaussian_noise", 5),
    ("Blur",    "motion_blur",    5),
    ("Weather", "snow",           5),
    ("Digital", "contrast",       5),
]

# --- 16c.6 Resume Logic ---
c16c_ckpt_dir = os.path.join(
    EVAL_PARAMS["ckpt_dir"], "block16c"
)
os.makedirs(c16c_ckpt_dir, exist_ok=True)

def load_16c_ckpt(corruption, severity):
    path = os.path.join(
        c16c_ckpt_dir, f"{corruption}_sev{severity}.json"
    )
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def save_16c_ckpt(corruption, severity, data):
    path = os.path.join(
        c16c_ckpt_dir, f"{corruption}_sev{severity}.json"
    )
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

# --- 16c.7 Main Loop ---
all_rows_16c = []

for cat_name, corruption, severity in EVAL_CORRUPTIONS_16C:

    ckpt = load_16c_ckpt(corruption, severity)
    if ckpt is not None:
        print(f"↩️  {corruption} sev{severity}: from checkpoint")
        all_rows_16c.extend(ckpt)
        continue

    print(f"\n▶  {cat_name}: {corruption} sev{severity}")

    # Reset LoRA B to zero before each corruption
    for name, param in dino_model.named_parameters():
        if 'lora_B' in name:
            param.data.zero_()

    # Clear projection cache
    for attr in list(vars(compute_alignment_loss_16c).keys()):
        if attr.startswith('_proj_'):
            delattr(compute_alignment_loss_16c, attr)

    # Fresh optimizer
    optimizer_16c = torch.optim.AdamW(
        [p for p in dino_model.parameters() if p.requires_grad],
        lr=TTT_LR_16C,
        weight_decay=1e-4
    )

    corruption_rows = []
    pbar = tqdm(image_files, desc=f"  {corruption}", leave=False)

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

        # === CONDITION C: CRATTT + Rank=16 TTT ===
        loss_values = []
        dino_model.train()

        for step in range(TTT_STEPS_16C):
            optimizer_16c.zero_grad()
            loss = compute_alignment_loss_16c(
                dino_model, dino_processor,
                c_img, crattt_preds_b, device
            )
            if loss is not None:
                loss.backward()
                optimizer_16c.step()
                loss_values.append(round(loss.item(), 6))

        dino_model.eval()

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

    save_16c_ckpt(corruption, severity, corruption_rows)
    all_rows_16c.extend(corruption_rows)

    c_df = pd.DataFrame(corruption_rows)
    print(f"  Baseline mAP      : {c_df['baseline_mAP'].mean():.4f}")
    print(f"  CRATTT mAP        : {c_df['crattt_mAP'].mean():.4f}")
    print(f"  CRATTT+TTT mAP    : {c_df['crattt_ttt_mAP'].mean():.4f}")
    print(f"  TTT gain vs CRATTT: "
          f"{c_df['delta_ttt_vs_crattt'].mean():+.4f}")

    sample_loss = next(
        (r['loss_trajectory'] for r in corruption_rows
         if r['loss_trajectory']), []
    )
    if sample_loss:
        print(f"  Loss trajectory   : {sample_loss[:5]}...")

# --- 16c.8 Results ---
df_16c = pd.DataFrame(all_rows_16c)

print("\n" + "="*70)
print("BLOCK 16c RESULTS: RANK=16 LoRA TTT ABLATION")
print("(feeds one row of Table 4.8 -- see below)")
print("="*70)

summary_16c = df_16c.groupby(
    ["category", "corruption", "severity"]
).agg(
    Baseline_mAP        =("baseline_mAP",        "mean"),
    CRATTT_mAP          =("crattt_mAP",           "mean"),
    CRATTT_TTT_mAP      =("crattt_ttt_mAP",       "mean"),
    Delta_TTT_vs_CRATTT =("delta_ttt_vs_crattt",   "mean"),
).round(4).reset_index()

display(summary_16c)

print(f"\n--- Overall Summary ---")
print(f"Mean Baseline mAP    : {df_16c['baseline_mAP'].mean():.4f}")
print(f"Mean CRATTT mAP      : {df_16c['crattt_mAP'].mean():.4f}")
print(f"Mean CRATTT+TTT mAP  : {df_16c['crattt_ttt_mAP'].mean():.4f}")
print(f"TTT gain vs CRATTT   : {df_16c['delta_ttt_vs_crattt'].mean():+.4f}")

# --- 16c.9 Three-Way Ablation Comparison ---
# Does NOT assume df_final (Block 16) or df_b (Block 16b) are
# still in memory. Tries live variables first, falls back to
# their saved CSVs on disk, and skips with a clear note rather
# than crashing if neither is available.
def _get_gain(label, varname, csv_path):
    if varname in globals():
        val = globals()[varname]['delta_ttt_vs_crattt'].mean()
        return val, "live"
    if os.path.exists(csv_path):
        val = pd.read_csv(csv_path)['delta_ttt_vs_crattt'].mean()
        return val, "from disk"
    return None, "unavailable"

gain_16, src_16 = _get_gain(
    "Block 16", "df_final",
    os.path.join(EVAL_PARAMS["table_dir"], "table_4_7_end_to_end.csv")
)
gain_16b, src_16b = _get_gain(
    "Block 16b", "df_b",
    os.path.join(EVAL_PARAMS["table_dir"], "block16b_alignment_ablation.csv")
)

print(f"\n{'='*60}")
print("ABLATION SUMMARY: TTT Configuration Comparison")
print(f"{'='*60}")
print(f"{'Configuration':<35} {'TTT gain vs CRATTT':>20}   {'source'}")
print("-"*70)

if gain_16 is not None:
    print(f"{'Block 16  (rank=4,  3 steps, conf loss)':<35} "
          f"{gain_16:>+20.4f}   {src_16}")
else:
    print(f"{'Block 16  (rank=4,  3 steps, conf loss)':<35} "
          f"{'N/A':>20}   re-run Block 16 to fill this in")

if gain_16b is not None:
    print(f"{'Block 16b (rank=4,  3 steps, align loss)':<35} "
          f"{gain_16b:>+20.4f}   {src_16b}")
else:
    print(f"{'Block 16b (rank=4,  3 steps, align loss)':<35} "
          f"{'N/A':>20}   re-run Block 16b to fill this in")

print(f"{'Block 16c (rank=16, 10 steps, align loss)':<35} "
      f"{df_16c['delta_ttt_vs_crattt'].mean():>+20.4f}   live")

# --- 16c.10 Save ---
csv_16c = os.path.join(
    EVAL_PARAMS["table_dir"], "block16c_rank16_ttt.csv"
)
df_16c.to_csv(csv_16c, index=False)

ablation_update = {
    "block_16_rank4_conf_loss":   round(float(gain_16), 4) if gain_16 is not None else None,
    "block_16b_rank4_align_loss": round(float(gain_16b), 4) if gain_16b is not None else None,
    "block_16c_rank16_align_loss": round(
        float(df_16c['delta_ttt_vs_crattt'].mean()), 4
    ),
    "rank_16c":    LORA_RANK_16C,
    "steps_16c":   TTT_STEPS_16C,
    "lr_16c":      TTT_LR_16C,
    "conclusion": (
        "positive"
        if df_16c['delta_ttt_vs_crattt'].mean() > 0.001
        else "capacity_constraint_confirmed"
    )
}

with open(os.path.join(
    EVAL_PARAMS["save_dir"], "ablation_ttt_rank16.json"
), "w") as f:
    json.dump(ablation_update, f, indent=2)

print(f"\n✅ Results saved: {csv_16c}")
print("\n" + "="*50)
print("BLOCK 16c COMPLETE")
print("="*50)
