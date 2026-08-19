# ============================================================
# BLOCK 20: Self-Supervised Consistency TTT (Option C)
# Implements TTT without any CLIP Oracle or external verifier.
#
# Principle: Apply two different augmentations to the same
# corrupted image. Run GroundingDINO on both. Minimise the
# disagreement between the two detection outputs.
# Consistent detections across augmentations are more likely
# to be genuine objects than noise-induced hallucinations.
#
# CAVEAT (documented, not fixed, to stay comparable to the
# original): compute_consistency_loss aligns View A and View B
# score lists by position (truncated to min length), not by any
# IoU-based correspondence. If the two augmented views produce
# differently-ordered or differently-sized detection sets, this
# can compare unrelated detections rather than the same object
# seen twice.
#
# Reference: Inspired by CoTTA (Wang et al. 2022) augmentation
# consistency principle, adapted for OVD detection.
#
# Scope: 20 images, 4 corruptions, severity 5, 10 steps
# Cost: Free (Kaggle T4)
# Runtime: ~35-45 minutes
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
import cv2
from imagecorruptions import corrupt as ic_corrupt
from tqdm.notebook import tqdm
from IPython.display import display
from transformers import AutoModelForZeroShotObjectDetection

print("="*55)
print("BLOCK 20: SELF-SUPERVISED CONSISTENCY TTT")
print("="*55)
print()
print("Principle: Two augmented views → minimise detection")
print("disagreement → consistent detections survive gate")
print()

# --- 20.1 Reload clean model, then inject rank=4 LoRA ---
# FIX: dino_model currently carries BOTH Block 14's dormant
# rank=4 encoder/decoder LoRA AND Block 19's active rank=8
# detection-head LoRA (155,872 params, still live from 19c).
# inject_lora_b20's isinstance(module, nn.Linear) check would
# match 0 encoder/decoder layers (already wrapped), then its
# "mark every lora_A/lora_B trainable" fallback would silently
# re-activate Block 19's unrelated detection-head LoRA alongside
# whatever this block intends. Reloading fresh avoids both
# carryovers at once.
print("Reloading fresh GroundingDINO-base "
      "(clearing all prior LoRA — Block 14 encoder/decoder AND "
      "Block 19 detection-head)...")
del dino_model
gc.collect()
torch.cuda.empty_cache()
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
    DINO_MODEL_ID, token=hf_token
).to(device)
dino_model.eval()

for param in clip_model.parameters():
    param.requires_grad = False

class LoRALinear20(nn.Module):
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


def inject_lora_b20(model, rank=4, lora_alpha=8):
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
        lora_layer = LoRALinear20(
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


n_inj_20, _ = inject_lora_b20(dino_model, rank=4, lora_alpha=8)
trainable_20 = sum(
    p.numel() for p in dino_model.parameters()
    if p.requires_grad
)

assert n_inj_20 == 36, f"Expected 36 LoRA layers, got {n_inj_20}"
assert trainable_20 == 73_728, (
    f"Expected 73,728 trainable params (rank=4 only), got "
    f"{trainable_20:,} — check for leftover LoRA from another block"
)
print(f"✅ LoRA rank=4 injected: {n_inj_20} layers (verified fresh)")
print(f"   Trainable params: {trainable_20:,} (0.0428%)")

# Forward pass check
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
print(f"✅ Forward pass OK — {len(rv['boxes'])} detections")

vram_free = (
    torch.cuda.get_device_properties(0).total_memory -
    torch.cuda.memory_allocated()
) / 1e9
print(f"   VRAM free: {vram_free:.2f} GB")


# --- 20.2 Augmentation Functions ---
def augment_view_a(image_np):
    """View A: Mild brightness + contrast jitter."""
    img = image_np.copy().astype(np.float32)
    alpha = np.random.uniform(0.85, 1.15)
    img   = img * alpha
    beta  = np.random.uniform(-10, 10)
    img   = img + beta
    return np.clip(img, 0, 255).astype(np.uint8)


def augment_view_b(image_np):
    """View B: Mild Gaussian blur + slight hue shift."""
    img = image_np.copy()
    img = cv2.GaussianBlur(img, (3, 3), sigmaX=0.5)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + np.random.uniform(-5, 5)) % 180
    img = cv2.cvtColor(
        np.clip(hsv, 0, 255).astype(np.uint8),
        cv2.COLOR_HSV2RGB
    )
    return img


# --- 20.3 Consistency Loss ---
def compute_consistency_loss(dino_model, dino_processor,
                              image_np, device):
    """
    Memory-efficient consistency loss.
    View A: forward with gradients (trains the model)
    View B: forward without gradients (reference target)
    Loss: MSE between View A scores and View B scores (detached)

    See the module-level caveat: scores are aligned by position,
    not by IoU correspondence.
    """
    view_a = augment_view_a(image_np)
    view_b = augment_view_b(image_np)

    inputs_a = dino_processor(
        images=view_a,
        text=DINO_TEXT_PROMPT,
        return_tensors="pt"
    ).to(device)
    outputs_a = dino_model(**inputs_a)

    res_a = dino_processor.post_process_grounded_object_detection(
        outputs_a, inputs_a.input_ids,
        target_sizes=[image_np.shape[:2]],
        text_threshold=0.05
    )[0]

    if len(res_a['scores']) == 0:
        return None

    with torch.no_grad():
        inputs_b = dino_processor(
            images=view_b,
            text=DINO_TEXT_PROMPT,
            return_tensors="pt"
        ).to(device)
        outputs_b  = dino_model(**inputs_b)
        res_b = dino_processor.post_process_grounded_object_detection(
            outputs_b, inputs_b.input_ids,
            target_sizes=[image_np.shape[:2]],
            text_threshold=0.05
        )[0]

    if len(res_b['scores']) == 0:
        return None

    min_len   = min(len(res_a['scores']), len(res_b['scores']))
    scores_a  = res_a['scores'][:min_len]
    scores_b  = res_b['scores'][:min_len].detach()

    loss = F.mse_loss(scores_a, scores_b)
    return loss

# --- 20.4 Configuration ---
EVAL_CORRUPTIONS_20 = [
    ("Noise",   "gaussian_noise", 5),
    ("Blur",    "motion_blur",    5),
    ("Weather", "snow",           5),
    ("Digital", "contrast",       5),
]
TTT_STEPS_20 = 10
TTT_LR_20    = 1e-4

print(f"\n--- Block 20 Configuration ---")
print(f"TTT type    : Self-supervised consistency")
print(f"Images      : {len(image_files)}")
print(f"Corruptions : {len(EVAL_CORRUPTIONS_20)}")
print(f"TTT steps   : {TTT_STEPS_20}")
print(f"TTT lr      : {TTT_LR_20}")
print(f"Tau         : {CRATTT_PARAMS['tau']}")
print(f"Aug View A  : brightness+contrast jitter")
print(f"Aug View B  : Gaussian blur + hue shift")

# --- 20.5 Resume Logic ---
b20_ckpt_dir = os.path.join(
    EVAL_PARAMS["ckpt_dir"], "block20"
)
os.makedirs(b20_ckpt_dir, exist_ok=True)

def load_b20_ckpt(corruption):
    path = os.path.join(b20_ckpt_dir, f"{corruption}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def save_b20_ckpt(corruption, data):
    path = os.path.join(b20_ckpt_dir, f"{corruption}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# --- 20.6 Main Loop ---
all_rows_20 = []

for cat_name, corruption, severity in EVAL_CORRUPTIONS_20:

    ckpt = load_b20_ckpt(corruption)
    if ckpt is not None:
        print(f"↩️  {corruption}: from checkpoint")
        all_rows_20.extend(ckpt)
        continue

    print(f"\n▶  {cat_name}: {corruption} sev{severity}")

    for name, param in dino_model.named_parameters():
        if 'lora_B' in name:
            param.data.zero_()

    optimizer_20 = torch.optim.AdamW(
        [p for p in dino_model.parameters() if p.requires_grad],
        lr=TTT_LR_20,
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

        # === CONDITION C: Consistency TTT ===
        loss_values = []
        dino_model.train()

        for step in range(TTT_STEPS_20):
            optimizer_20.zero_grad()
            loss = compute_consistency_loss(
                dino_model, dino_processor,
                c_img, device
            )
            if loss is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in dino_model.parameters()
                     if p.requires_grad],
                    max_norm=1.0
                )
                optimizer_20.step()
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
            "baseline_mAP":        round(float(bmap),   4),
            "crattt_mAP":          round(float(cmap_b), 4),
            "consistency_ttt_mAP": round(float(cmap_c), 4),
            "delta_crattt":        round(float(cmap_b - bmap), 4),
            "delta_ttt":           round(float(cmap_c - bmap), 4),
            "delta_ttt_vs_crattt": round(float(cmap_c - cmap_b), 4),
            "n_baseline":          len(baseline_res['boxes']),
            "n_crattt":            stats_b['n_verified'],
            "n_ttt":               stats_c['n_verified'],
            "loss_mean":           round(float(np.mean(loss_values))
                                         if loss_values else 0, 4)
        }
        corruption_rows.append(row)

        pbar.set_postfix({
            "B": f"{bmap:.3f}",
            "C": f"{cmap_b:.3f}",
            "T": f"{cmap_c:.3f}"
        })

    save_b20_ckpt(corruption, corruption_rows)
    all_rows_20.extend(corruption_rows)

    c_df = pd.DataFrame(corruption_rows)
    print(f"  Baseline mAP          : "
          f"{c_df['baseline_mAP'].mean():.4f}")
    print(f"  CRATTT mAP            : "
          f"{c_df['crattt_mAP'].mean():.4f}")
    print(f"  Consistency TTT mAP   : "
          f"{c_df['consistency_ttt_mAP'].mean():.4f}")
    print(f"  TTT gain vs CRATTT    : "
          f"{c_df['delta_ttt_vs_crattt'].mean():+.4f}")

    sample = next(
        (r['loss_mean'] for r in corruption_rows
         if r['loss_mean'] > 0), None
    )
    if sample:
        print(f"  Mean loss             : {sample:.6f}")


# --- 20.7 Results ---
df_20 = pd.DataFrame(all_rows_20)

print("\n" + "="*70)
print("TABLE: BLOCK 20 CONSISTENCY TTT RESULTS (N=20)")
print("="*70)

summary_20 = df_20.groupby(
    ["category", "corruption"]
).agg(
    Baseline_mAP        =("baseline_mAP",        "mean"),
    CRATTT_mAP          =("crattt_mAP",           "mean"),
    Consistency_TTT_mAP =("consistency_ttt_mAP",  "mean"),
    TTT_gain_vs_CRATTT  =("delta_ttt_vs_crattt",  "mean"),
).round(4).reset_index()

display(summary_20)

overall_gain_20 = df_20['delta_ttt_vs_crattt'].mean()
positive_imgs   = (df_20['delta_ttt_vs_crattt'] > 0.001).sum()
negative_imgs   = (df_20['delta_ttt_vs_crattt'] < -0.001).sum()

print(f"\n--- Overall Summary ---")
print(f"Mean Baseline mAP      : {df_20['baseline_mAP'].mean():.4f}")
print(f"Mean CRATTT mAP        : {df_20['crattt_mAP'].mean():.4f}")
print(f"Mean Consistency TTT   : "
      f"{df_20['consistency_ttt_mAP'].mean():.4f}")
print(f"Overall TTT gain       : {overall_gain_20:+.4f}")
print(f"Images with gain > 0   : {positive_imgs}/{len(df_20)}")
print(f"Images with loss < 0   : {negative_imgs}/{len(df_20)}")

# --- 20.8 Complete Final Ablation ---
# FIXED: reads every prior block's gain live from memory if
# available, falling back to its saved CSV on disk, instead of
# hardcoded literals (several of which — Block 19, 19c — would
# now be flatly wrong for this run if hardcoded).
def _get_gain(varname, csv_path, csv_col="delta_ttt_vs_crattt"):
    if varname in globals():
        return globals()[varname][csv_col].mean(), "live"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)[csv_col].mean(), "from disk"
    return None, "unavailable"

table_dir = EVAL_PARAMS["table_dir"]

gain_16, _   = _get_gain("df_final", os.path.join(table_dir, "table_4_7_end_to_end.csv"))
gain_16b, _  = _get_gain("df_b",     os.path.join(table_dir, "block16b_alignment_ablation.csv"))
gain_16c, _  = _get_gain("df_16c",   os.path.join(table_dir, "block16c_rank16_ttt.csv"))
gain_17, _   = _get_gain("df_17",    os.path.join(table_dir, "table_block17_coadapt_pilot.csv"),
                          csv_col="delta_coadapt_vs_crattt")
gain_18, _   = _get_gain("df_18",    os.path.join(table_dir, "table_block18_encoder_gate.csv"))
gain_19, _   = _get_gain("df_19",    os.path.join(table_dir, "table_block19_dethead_lora.csv"))
gain_19c, _  = _get_gain("df_19c",   os.path.join(table_dir, "table_block19c_gaussian_noise_n20.csv"))

print(f"\n{'='*72}")
print("COMPLETE ABLATION: All TTT Configurations (Swin-B, this run)")
print(f"{'='*72}")
print(f"{'Configuration':<50} {'TTT gain':>10}")
print("-"*62)

def _fmt(label, gain, marker=""):
    g = f"{gain:+.4f}" if gain is not None else "N/A"
    print(f"  {label:<48} {g:>10}{marker}")

_fmt("Blk 16  encoder LoRA  conf-loss  frozen oracle",   gain_16)
_fmt("Blk 16b encoder LoRA  align-loss frozen oracle",   gain_16b)
_fmt("Blk 16c encoder LoRA  rank=16    frozen oracle",   gain_16c)
_fmt("Blk 17  encoder LoRA  align-loss co-adapt oracle", gain_17)
_fmt("Blk 18  encoder gate  (untrained projection)",     gain_18)
_fmt("Blk 19  dethead LoRA  N=5",                        gain_19)
_fmt("Blk 19c dethead LoRA  N=20 (1/20 images drove it)", gain_19c)
_fmt("Blk 20  consistency TTT (no oracle)",               overall_gain_20, " ◄")

print()
if overall_gain_20 > 0.010:
    verdict    = "✅ POSITIVE GAIN — Consistency TTT works!"
    action     = ("Proceed to Vast.ai for 50-image validation "
                  "across all 15 corruptions.")
    conclusion = "positive_proceed_vastai"
elif overall_gain_20 > 0.003:
    verdict    = "✅ MARGINAL POSITIVE GAIN"
    action     = ("Small but consistent improvement. "
                  "Check per-image distribution before reporting — "
                  "Block 19c's similar-looking gain was driven by "
                  "a single image, not a broad effect.")
    conclusion = "marginal_positive"
elif overall_gain_20 > -0.003:
    verdict    = "⚠️  NEAR-ZERO — No reliable improvement"
    action     = ("Try Option B trained projection next.")
    conclusion = "near_zero_try_option_b"
else:
    verdict    = "❌ NEGATIVE — Consistency TTT degrades"
    action     = ("Proceed to Option B trained projection "
                  "as final attempt.")
    conclusion = "negative_try_option_b"

print(verdict)
print(f"Action: {action}")

# --- 20.9 Save ---
csv_20 = os.path.join(
    EVAL_PARAMS["table_dir"], "table_block20_consistency_ttt.csv"
)
df_20.to_csv(csv_20, index=False)

with open(os.path.join(
    EVAL_PARAMS["save_dir"], "block20_consistency_ttt.json"
), "w") as f:
    json.dump({
        "method":          "self_supervised_consistency",
        "n_images":        len(image_files),
        "corruptions":     [c[1] for c in EVAL_CORRUPTIONS_20],
        "ttt_steps":       TTT_STEPS_20,
        "lr":              TTT_LR_20,
        "overall_gain":    round(float(overall_gain_20), 4),
        "positive_images": int(positive_imgs),
        "conclusion":      conclusion,
        "ablation_summary": {
            "block_16": round(float(gain_16), 4)  if gain_16  is not None else None,
            "block_16b": round(float(gain_16b), 4) if gain_16b is not None else None,
            "block_16c": round(float(gain_16c), 4) if gain_16c is not None else None,
            "block_17": round(float(gain_17), 4)  if gain_17  is not None else None,
            "block_18": round(float(gain_18), 4)  if gain_18  is not None else None,
            "block_19": round(float(gain_19), 4)  if gain_19  is not None else None,
            "block_19c": round(float(gain_19c), 4) if gain_19c is not None else None,
            "block_20": round(float(overall_gain_20), 4)
        }
    }, f, indent=2)

print(f"\n✅ Results saved: {csv_20}")
print("\n" + "="*50)
print("BLOCK 20 COMPLETE")
print("="*50)
