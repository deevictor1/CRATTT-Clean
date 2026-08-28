# ============================================================
# BLOCK 21: Trained Cross-Modal Projection + Encoder Gate TTT
# Option B — properly implemented with pre-trained projection.
#
# Block 18 failed because the DINO-to-CLIP projection was
# randomly initialised (orthogonal init), producing near-zero
# Soracle scores. This block fixes that by:
#
# Step 1: Pre-train the projection on clean COCO images
#         (~30 mins, free on Kaggle T4)
#         Maps DINO encoder features → CLIP text space
#         so Soracle scores are in the correct 0.2-0.4 range
#
# Step 2: Freeze the trained projection
#         Inject LoRA into DINO encoder neck (rank=4)
#         Run TTT with encoder-feature Soracle gate
#
# Hypothesis: With a trained projection, the encoder-based
# Soracle will produce meaningful scores. LoRA updates to
# the DINO encoder will shift these scores, making the
# TTRV gate sensitive to TTT for the first time.
#
# NOTE: this is config 8 of 12 in Table 4.8 — Blocks 22, 22b,
# 23, 23b, 23c still follow this one. Whatever this prints is
# not yet the final word on Section A's TTT exploration.
#
# Scope: 20 images, 4 corruptions, severity 5, 10 steps
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

print("="*55)
print("BLOCK 21: TRAINED PROJECTION + ENCODER GATE TTT")
print("="*55)
print()

# --- 21.1 Define the Cross-Modal Projection ---
class CrossModalProjection(nn.Module):
    """
    Learns to map DINO encoder hidden states to CLIP text space.
    Architecture: Linear → LayerNorm → ReLU → Linear
    Input:  [batch, dino_hidden_dim]  e.g. [1, 256]
    Output: [batch, clip_text_dim]    e.g. [1, 512]

    Trained on clean COCO images using CLIP text embeddings
    as supervision targets.
    """
    def __init__(self, dino_dim, clip_dim, hidden_dim=384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dino_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, clip_dim)
        )

    def forward(self, x):
        return self.net(x)


# --- 21.2 Get DINO hidden dimension ---
print("Step 0: Detecting DINO encoder hidden dimension...")
test_img = loaded_images[image_files[0]]
with torch.no_grad():
    tv = dino_processor(
        images=test_img, text=DINO_TEXT_PROMPT,
        return_tensors="pt"
    ).to(device)
    ov = dino_model(**tv)

if hasattr(ov, 'encoder_last_hidden_state') and \
   ov.encoder_last_hidden_state is not None:
    dino_hidden_dim = ov.encoder_last_hidden_state.shape[-1]
elif hasattr(ov, 'last_hidden_state') and \
     ov.last_hidden_state is not None:
    dino_hidden_dim = ov.last_hidden_state.shape[-1]
else:
    dino_hidden_dim = 256  # fallback only — every other block using this
                            # same hasattr pattern has hit the primary
                            # branch successfully, so this is not expected
                            # to execute; verify the actual value here if
                            # it ever does

clip_text_dim = clip_text_features.shape[-1]  # 512

print(f"   DINO encoder dim : {dino_hidden_dim}")
print(f"   CLIP text dim    : {clip_text_dim}")

projection = CrossModalProjection(
    dino_dim=dino_hidden_dim,
    clip_dim=clip_text_dim,
    hidden_dim=384
).to(device)

proj_params = sum(p.numel() for p in projection.parameters())
print(f"   Projection params: {proj_params:,}")
print(f"✅ Projection module created")


# --- 21.3 Pre-Training Loop ---
print()
print("="*55)
print("Step 1: Pre-training cross-modal projection")
print("="*55)
print(f"Training on {len(image_files)} clean COCO images")
print(f"Supervision: CLIP text embeddings for detected classes")
print(f"Epochs: 5 passes over the image set")
print()

PRE_TRAIN_EPOCHS = 5
PRE_TRAIN_LR     = 1e-3

proj_optimizer = torch.optim.AdamW(
    projection.parameters(),
    lr=PRE_TRAIN_LR,
    weight_decay=1e-4
)
proj_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    proj_optimizer,
    T_max=PRE_TRAIN_EPOCHS * len(image_files)
)

projection.train()
dino_model.eval()  # DINO frozen during pre-training

all_epoch_losses = []

for epoch in range(PRE_TRAIN_EPOCHS):
    epoch_losses = []
    pbar = tqdm(
        image_files, desc=f"Epoch {epoch+1}/{PRE_TRAIN_EPOCHS}",
        leave=False
    )

    for img_path in pbar:
        raw_img = loaded_images[img_path]

        with torch.no_grad():
            inputs = dino_processor(
                images=raw_img, text=DINO_TEXT_PROMPT,
                return_tensors="pt"
            ).to(device)
            outputs = dino_model(**inputs)

            if hasattr(outputs, 'encoder_last_hidden_state') and \
               outputs.encoder_last_hidden_state is not None:
                enc_hidden = outputs.encoder_last_hidden_state
            else:
                enc_hidden = outputs.last_hidden_state

            dino_feat = enc_hidden.mean(dim=1)

            res = dino_processor\
                .post_process_grounded_object_detection(
                    outputs, inputs.input_ids,
                    target_sizes=[raw_img.shape[:2]],
                    text_threshold=CRATTT_PARAMS["dino_text_thr"]
                )[0]

            labels = res.get("text_labels", res.get("labels", []))
            target_embs = []
            for lbl in labels:
                if isinstance(lbl, str):
                    clean = lbl.lower().replace(".", "").strip()
                    if clean in COCO_CLASSES:
                        idx = COCO_CLASSES.index(clean)
                        target_embs.append(clip_text_features[idx])

        if not target_embs:
            continue

        target = torch.stack(target_embs).mean(dim=0, keepdim=True)
        target = F.normalize(target, p=2, dim=-1)

        proj_optimizer.zero_grad()
        projected = projection(dino_feat)
        proj_norm  = F.normalize(projected, p=2, dim=-1)

        similarity = (proj_norm * target).sum(dim=-1)
        loss       = (1.0 - similarity).mean()

        loss.backward()
        proj_optimizer.step()
        proj_scheduler.step()
        epoch_losses.append(loss.item())
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    mean_epoch_loss = np.mean(epoch_losses) if epoch_losses else 0
    all_epoch_losses.append(mean_epoch_loss)
    print(f"  Epoch {epoch+1}: mean loss = {mean_epoch_loss:.6f}")

print(f"\n✅ Pre-training complete")
print(f"   Loss trajectory: {[round(l,4) for l in all_epoch_losses]}")

for param in projection.parameters():
    param.requires_grad = False
print(f"✅ Projection frozen")

# --- 21.4 Validate Projection Soracle Scores ---
print()
print("Validating trained projection Soracle scores...")
projection.eval()

with torch.no_grad():
    inputs_val = dino_processor(
        images=test_img, text=DINO_TEXT_PROMPT,
        return_tensors="pt"
    ).to(device)
    outputs_val = dino_model(**inputs_val)

    if hasattr(outputs_val, 'encoder_last_hidden_state') and \
       outputs_val.encoder_last_hidden_state is not None:
        enc_val = outputs_val.encoder_last_hidden_state
    else:
        enc_val = outputs_val.last_hidden_state

    dino_feat_val = enc_val.mean(dim=1)
    proj_feat     = projection(dino_feat_val)
    proj_norm_val = F.normalize(proj_feat, p=2, dim=-1)

    sample_labels = ['person', 'car', 'chair', 'tv', 'dog']
    print(f"\n   Trained projection Soracle scores:")
    for lbl in sample_labels:
        if lbl in COCO_CLASSES:
            idx       = COCO_CLASSES.index(lbl)
            text_feat = F.normalize(
                clip_text_features[idx:idx+1], p=2, dim=-1
            )
            sim = (proj_norm_val * text_feat).sum().item()
            print(f"   {lbl:<20}: {sim:.4f}")

print()
print(f"   (Block 18 random-projection scores were ~0.01-0.05)")
print(f"   (Target range for meaningful gate: 0.20-0.40)")


# --- 21.5 Confirm clean state, then inject LoRA for TTT ---
print()
print("Step 2: Checking DINO's current LoRA state before TTT injection...")

class LoRALinear21(nn.Module):
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


def inject_lora_b21(model, rank=4, lora_alpha=8):
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
        lora_layer = LoRALinear21(
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


# FIX: dino_model still has Block 20's rank=4 LoRALinear wrappers
# (query/value, same scope this block targets). Calling
# inject_lora_b21 directly on it would match 0 layers and silently
# reuse Block 20's drifted lora_A — functionally harmless here
# because the main loop re-zeros lora_B per corruption anyway, but
# not a clean start. Reloading guarantees a genuine fresh init.
existing_lora_b = sum(
    1 for n, _ in dino_model.named_parameters() if 'lora_B' in n
)
existing_trainable = sum(
    p.numel() for p in dino_model.parameters() if p.requires_grad
)
if existing_lora_b == 36:
    print(f"   dino_model already has {existing_lora_b} LoRA layers "
          f"({existing_trainable:,} trainable params) from a prior "
          f"block. Reloading fresh for a genuine Kaiming-init start.")

del dino_model
gc.collect()
torch.cuda.empty_cache()
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
    DINO_MODEL_ID, token=hf_token
).to(device)
dino_model.eval()

n_inj_21, _ = inject_lora_b21(dino_model, rank=4, lora_alpha=8)
assert n_inj_21 == 36, f"Expected 36 LoRA layers, got {n_inj_21}"

trainable_21 = sum(
    p.numel() for p in dino_model.parameters()
    if p.requires_grad
)
assert trainable_21 == 73_728, (
    f"Expected 73,728 trainable params, got {trainable_21:,}"
)
print(f"✅ LoRA injected fresh: {n_inj_21} layers, "
      f"{trainable_21:,} trainable params")

# Forward pass check
with torch.no_grad():
    tv2 = dino_processor(
        images=test_img, text=DINO_TEXT_PROMPT,
        return_tensors="pt"
    ).to(device)
    ov2 = dino_model(**tv2)
rv2 = dino_processor.post_process_grounded_object_detection(
    ov2, tv2.input_ids,
    target_sizes=[test_img.shape[:2]],
    text_threshold=CRATTT_PARAMS["dino_text_thr"]
)[0]
print(f"✅ Forward pass OK — {len(rv2['boxes'])} detections")

vram_free = (
    torch.cuda.get_device_properties(0).total_memory -
    torch.cuda.memory_allocated()
) / 1e9
print(f"   VRAM free: {vram_free:.2f} GB")


# --- 21.6 Encoder-Feature CRATTT with Trained Projection ---
def run_crattt_trained_proj(image_np, dino_outputs=None):
    """
    CRATTT inference using trained projection Soracle.
    Computes Soracle from DINO encoder features projected
    into CLIP text space via the trained CrossModalProjection.
    """
    inputs_ref = dino_processor(
        images=image_np, text=DINO_TEXT_PROMPT,
        return_tensors="pt"
    ).to(device)

    if dino_outputs is None:
        with torch.no_grad():
            dino_outputs = dino_model(**inputs_ref)

    res = dino_processor.post_process_grounded_object_detection(
        dino_outputs, inputs_ref.input_ids,
        target_sizes=[image_np.shape[:2]],
        text_threshold=CRATTT_PARAMS["dino_text_thr"]
    )[0]

    boxes  = res["boxes"]
    scores = res["scores"]
    labels = res.get("text_labels", res.get("labels", []))

    if len(boxes) == 0:
        return [], {"n_proposals": 0, "n_verified": 0,
                    "n_rejected": 0}

    if hasattr(dino_outputs, 'encoder_last_hidden_state') and \
       dino_outputs.encoder_last_hidden_state is not None:
        enc = dino_outputs.encoder_last_hidden_state
    else:
        enc = dino_outputs.last_hidden_state

    dino_feat  = enc.mean(dim=1)

    with torch.no_grad():
        proj_feat  = projection(dino_feat)
        proj_norm  = F.normalize(proj_feat, p=2, dim=-1)

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

        if clean_label in COCO_CLASSES:
            idx       = COCO_CLASSES.index(clean_label)
            text_feat = F.normalize(
                clip_text_features[idx:idx+1], p=2, dim=-1
            )
            soracle = (proj_norm * text_feat).sum().item()
        else:
            soracle = 0.0

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


# --- 21.7 Validate Trained Gate on Clean Image ---
print("\nValidating trained projection gate on clean image...")
with torch.no_grad():
    clean_in  = dino_processor(
        images=test_img, text=DINO_TEXT_PROMPT,
        return_tensors="pt"
    ).to(device)
    clean_out = dino_model(**clean_in)

verified_gate, stats_gate = run_crattt_trained_proj(
    test_img, clean_out
)

# FIX: compute the original crop-gate's count live, on this run's
# clean image, instead of the hardcoded "10" left over from a
# different run on a different backbone.
crop_gate_clean_preds, crop_gate_clean_stats = run_crattt_inference(test_img)
print(f"✅ Trained gate: {stats_gate['n_verified']} verified, "
      f"{stats_gate['n_rejected']} rejected")
print(f"   (Original BARON crop gate, this run: "
      f"{crop_gate_clean_stats['n_verified']} verified)")
print(f"   (Block 18 random-projection gate gave 0 verified)")

if stats_gate['n_verified'] == 0:
    print()
    print("⚠️  Gate still passing 0 detections.")
    print("   Checking Soracle scores vs tau...")
    if verified_gate == [] and len(rv2['boxes']) > 0:
        labels_check = rv2.get(
            "text_labels", rv2.get("labels", [])
        )
        with torch.no_grad():
            enc_c = clean_out.encoder_last_hidden_state \
                if hasattr(clean_out, 'encoder_last_hidden_state') \
                and clean_out.encoder_last_hidden_state is not None \
                else clean_out.last_hidden_state
            df_c   = enc_c.mean(dim=1)
            pf_c   = F.normalize(projection(df_c), p=2, dim=-1)

        print(f"   Sample Rjoint values (tau={CRATTT_PARAMS['tau']}):")
        for ii in range(min(5, len(rv2['boxes']))):
            ds = rv2['scores'][ii].item()
            lbl = labels_check[ii] if ii < len(labels_check) else "?"
            if isinstance(lbl, str):
                cl = lbl.lower().replace(".", "").strip()
                if cl in COCO_CLASSES:
                    idx = COCO_CLASSES.index(cl)
                    tf  = F.normalize(
                        clip_text_features[idx:idx+1], p=2, dim=-1
                    )
                    so  = (pf_c * tf).sum().item()
                    rj  = (ds**CRATTT_PARAMS["alpha"]) * \
                          (max(so,0)**CRATTT_PARAMS["beta"]) if so > 0 else 0
                    print(f"   {cl:<20} dino={ds:.3f} "
                          f"soracle={so:.4f} rjoint={rj:.4f}")


# --- 21.8 TTT Loss ---
def compute_proj_ttt_loss(dino_model, dino_processor,
                           projection, image_np,
                           verified_preds, device):
    """
    TTT loss using trained projection.
    Maximises cosine alignment between projected DINO features
    and CLIP text embeddings for verified classes.
    Same projection the live gate uses — gradients flow through
    the same feature space the gate scores, unlike Block 17.
    """
    if not verified_preds:
        return None

    inputs = dino_processor(
        images=image_np, text=DINO_TEXT_PROMPT,
        return_tensors="pt"
    ).to(device)
    outputs = dino_model(**inputs)

    if hasattr(outputs, 'encoder_last_hidden_state') and \
       outputs.encoder_last_hidden_state is not None:
        enc = outputs.encoder_last_hidden_state
    else:
        enc = outputs.last_hidden_state

    dino_feat = enc.mean(dim=1)
    proj_feat = projection(dino_feat)
    proj_norm = F.normalize(proj_feat, p=2, dim=-1)

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
    similarity  = (proj_norm * target_norm).sum(dim=-1)
    loss        = (1.0 - similarity).mean()
    return loss


# --- 21.9 Main TTT Loop ---
EVAL_CORRUPTIONS_21 = [
    ("Noise",   "gaussian_noise", 5),
    ("Blur",    "motion_blur",    5),
    ("Weather", "snow",           5),
    ("Digital", "contrast",       5),
]
TTT_STEPS_21 = 10
TTT_LR_21    = 1e-4

print(f"\n--- Block 21 TTT Configuration ---")
print(f"Gate type   : Trained projection encoder gate")
print(f"Images      : {len(image_files)}")
print(f"Corruptions : {len(EVAL_CORRUPTIONS_21)}")
print(f"TTT steps   : {TTT_STEPS_21}")
print(f"TTT lr      : {TTT_LR_21}")
print(f"Tau         : {CRATTT_PARAMS['tau']}")

b21_ckpt_dir = os.path.join(EVAL_PARAMS["ckpt_dir"], "block21")
os.makedirs(b21_ckpt_dir, exist_ok=True)

all_rows_21 = []

for cat_name, corruption, severity in EVAL_CORRUPTIONS_21:

    ckpt_path = os.path.join(b21_ckpt_dir, f"{corruption}.json")
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            rows = json.load(f)
        print(f"↩️  {corruption}: from checkpoint")
        all_rows_21.extend(rows)
        continue

    print(f"\n▶  {cat_name}: {corruption} sev{severity}")

    for name, param in dino_model.named_parameters():
        if 'lora_B' in name:
            param.data.zero_()

    optimizer_21 = torch.optim.AdamW(
        [p for p in dino_model.parameters() if p.requires_grad],
        lr=TTT_LR_21,
        weight_decay=1e-4
    )

    corruption_rows = []
    pbar = tqdm(image_files, desc=f"  {corruption}", leave=False)

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

        with torch.no_grad():
            inp_b = dino_processor(
                images=c_img, text=DINO_TEXT_PROMPT,
                return_tensors="pt"
            ).to(device)
            out_b = dino_model(**inp_b)
        crattt_b, stats_b = run_crattt_trained_proj(c_img, out_b)
        for p in crattt_b:
            p["image_id"] = img_id
        coco_b = [{
            "image_id": p["image_id"],
            "category_id": p["category_id"],
            "bbox": p["bbox"], "score": p["score"]
        } for p in crattt_b]
        cmap_b, _ = compute_map(coco_b, coco_gt, [img_id])

        loss_vals = []
        dino_model.train()
        for step in range(TTT_STEPS_21):
            optimizer_21.zero_grad()
            loss = compute_proj_ttt_loss(
                dino_model, dino_processor,
                projection, c_img, crattt_b, device
            )
            if loss is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in dino_model.parameters()
                     if p.requires_grad],
                    max_norm=1.0
                )
                optimizer_21.step()
                loss_vals.append(round(loss.item(), 6))
        dino_model.eval()

        with torch.no_grad():
            inp_c = dino_processor(
                images=c_img, text=DINO_TEXT_PROMPT,
                return_tensors="pt"
            ).to(device)
            out_c = dino_model(**inp_c)
        crattt_c, stats_c = run_crattt_trained_proj(c_img, out_c)
        for p in crattt_c:
            p["image_id"] = img_id
        coco_c = [{
            "image_id": p["image_id"],
            "category_id": p["category_id"],
            "bbox": p["bbox"], "score": p["score"]
        } for p in crattt_c]
        cmap_c, _ = compute_map(coco_c, coco_gt, [img_id])

        row = {
            "category":            cat_name,
            "corruption":          corruption,
            "image":               fname,
            "baseline_mAP":        round(float(bmap),   4),
            "proj_crattt_mAP":     round(float(cmap_b), 4),
            "proj_ttt_mAP":        round(float(cmap_c), 4),
            "delta_ttt_vs_crattt": round(float(cmap_c - cmap_b), 4),
            "n_baseline":          len(base_res['boxes']),
            "n_proj_crattt":       stats_b['n_verified'],
            "n_proj_ttt":          stats_c['n_verified'],
            "loss_mean":           round(float(np.mean(loss_vals))
                                         if loss_vals else 0, 4)
        }
        corruption_rows.append(row)
        pbar.set_postfix({
            "B": f"{bmap:.3f}",
            "C": f"{cmap_b:.3f}",
            "T": f"{cmap_c:.3f}"
        })

    with open(ckpt_path, "w") as f:
        json.dump(corruption_rows, f, indent=2)
    all_rows_21.extend(corruption_rows)

    c_df = pd.DataFrame(corruption_rows)
    print(f"  Baseline mAP          : {c_df['baseline_mAP'].mean():.4f}")
    print(f"  Proj CRATTT mAP       : {c_df['proj_crattt_mAP'].mean():.4f}")
    print(f"  Proj TTT mAP          : {c_df['proj_ttt_mAP'].mean():.4f}")
    print(f"  TTT gain vs CRATTT    : "
          f"{c_df['delta_ttt_vs_crattt'].mean():+.4f}")

# --- 21.10 Results ---
df_21 = pd.DataFrame(all_rows_21)
overall_21 = df_21['delta_ttt_vs_crattt'].mean()
positive_21 = (df_21['delta_ttt_vs_crattt'] > 0.001).sum()

print("\n" + "="*70)
print("TABLE: BLOCK 21 TRAINED PROJECTION TTT RESULTS")
print("="*70)
summary_21 = df_21.groupby(["category", "corruption"]).agg(
    Baseline_mAP    =("baseline_mAP",        "mean"),
    Proj_CRATTT_mAP =("proj_crattt_mAP",      "mean"),
    Proj_TTT_mAP    =("proj_ttt_mAP",         "mean"),
    TTT_gain        =("delta_ttt_vs_crattt",  "mean"),
).round(4).reset_index()
display(summary_21)

print(f"\nOverall TTT gain   : {overall_21:+.4f}")
print(f"Images with gain   : {positive_21}/{len(df_21)}")

# FIXED: reads every prior block's gain live from memory if
# available, falling back to its saved CSV on disk, instead of
# hardcoded literals from an earlier run.
def _get_gain(varname, csv_path, csv_col="delta_ttt_vs_crattt"):
    if varname in globals():
        return globals()[varname][csv_col].mean(), "live"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)[csv_col].mean(), "from disk"
    return None, "unavailable"

table_dir = EVAL_PARAMS["table_dir"]
gain_16, _  = _get_gain("df_final", os.path.join(table_dir, "table_4_7_end_to_end.csv"))
gain_16b, _ = _get_gain("df_b",     os.path.join(table_dir, "block16b_alignment_ablation.csv"))
gain_16c, _ = _get_gain("df_16c",   os.path.join(table_dir, "block16c_rank16_ttt.csv"))
gain_17, _  = _get_gain("df_17",    os.path.join(table_dir, "table_block17_coadapt_pilot.csv"),
                         csv_col="delta_coadapt_vs_crattt")
gain_18, _  = _get_gain("df_18",    os.path.join(table_dir, "table_block18_encoder_gate.csv"))
gain_19, _  = _get_gain("df_19",    os.path.join(table_dir, "table_block19_dethead_lora.csv"))
gain_19c, _ = _get_gain("df_19c",   os.path.join(table_dir, "table_block19c_gaussian_noise_n20.csv"))
gain_20, _  = _get_gain("df_20",    os.path.join(table_dir, "table_block20_consistency_ttt.csv"))

print(f"\n{'='*72}")
print("ABLATION SO FAR: 8 of 12 Table 4.8 Configurations Tested (Swin-B)")
print("(Blocks 22, 22b, 23, 23b, 23c still remain)")
print(f"{'='*72}")
print(f"{'Configuration':<50} {'TTT gain':>10}")
print("-"*62)

def _fmt(label, gain, marker=""):
    g = f"{gain:+.4f}" if gain is not None else "N/A"
    print(f"  {label:<48} {g:>10}{marker}")

_fmt("Blk 16  encoder LoRA conf-loss  frozen oracle",    gain_16)
_fmt("Blk 16b encoder LoRA align-loss frozen oracle",    gain_16b)
_fmt("Blk 16c encoder LoRA rank=16    frozen oracle",    gain_16c)
_fmt("Blk 17  encoder LoRA align-loss co-adapt oracle",  gain_17)
_fmt("Blk 18  encoder gate untrained projection",        gain_18)
_fmt("Blk 19/19c dethead LoRA (19c: N=20, 1 image drove it)", gain_19c)
_fmt("Blk 20  self-supervised consistency",              gain_20)
_fmt("Blk 21  trained projection encoder gate",          overall_21, " ◄")

print()
if overall_21 > 0.010:
    print("✅ POSITIVE GAIN — Trained projection TTT works!")
    print("   Worth deeper investigation before drawing conclusions —")
    print("   but Blocks 22-23c still need running either way.")
    conclusion = "positive_investigate_further"
elif overall_21 > 0.003:
    print("✅ MARGINAL POSITIVE GAIN")
    print("   Small improvement. 4 more configs remain in Table 4.8.")
    conclusion = "marginal_positive"
elif overall_21 > -0.003:
    print("⚠️  NEAR-ZERO — No reliable improvement from this config")
    print("   This is config 8 of 12 — Blocks 22, 22b, 23, 23b, 23c remain.")
    conclusion = "near_zero_continue_ablation"
else:
    print("❌ NEGATIVE for this configuration")
    print("   This is config 8 of 12 — Blocks 22, 22b, 23, 23b, 23c remain.")
    conclusion = "negative_continue_ablation"

# Save
csv_21 = os.path.join(
    EVAL_PARAMS["table_dir"], "table_block21_trained_proj.csv"
)
df_21.to_csv(csv_21, index=False)
with open(os.path.join(
    EVAL_PARAMS["save_dir"], "block21_results.json"
), "w") as f:
    json.dump({
        "method":          "trained_projection_encoder_gate",
        "n_images":        len(image_files),
        "pre_train_epochs": PRE_TRAIN_EPOCHS,
        "loss_trajectory": [round(l,4) for l in all_epoch_losses],
        "overall_gain":    round(float(overall_21), 4),
        "positive_images": int(positive_21),
        "conclusion":      conclusion,
        "ablation_so_far": {
            "block_16": round(float(gain_16), 4)  if gain_16  is not None else None,
            "block_16b": round(float(gain_16b), 4) if gain_16b is not None else None,
            "block_16c": round(float(gain_16c), 4) if gain_16c is not None else None,
            "block_17": round(float(gain_17), 4)  if gain_17  is not None else None,
            "block_18": round(float(gain_18), 4)  if gain_18  is not None else None,
            "block_19c": round(float(gain_19c), 4) if gain_19c is not None else None,
            "block_20": round(float(gain_20), 4)  if gain_20  is not None else None,
            "block_21": round(float(overall_21), 4)
        }
    }, f, indent=2)

print(f"\n✅ Saved: {csv_21}")
print("\n" + "="*50)
print("BLOCK 21 COMPLETE — 8 of 12 Table 4.8 configs done")
print("="*50)
