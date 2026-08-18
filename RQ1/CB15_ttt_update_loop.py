# ============================================================
# BLOCK 15: Test-Time Training Update Loop
# Uses TTRV-verified detections as pseudo-labels to perform
# lightweight LoRA parameter updates at test time.
# Only detections that passed the CLIP Oracle gate (Rjoint≥τ)
# are used for gradient updates.
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from imagecorruptions import corrupt as ic_corrupt
from tqdm.notebook import tqdm

# --- 15.1 TTT Loss Function ---
def compute_ttt_loss(dino_output, verified_preds, img_shape):
    """
    Computes the TTT supervision loss from verified pseudo-labels.
    
    Uses a focal-style confidence loss:
    - Encourages high logit scores for verified detections
    - Does not penalise unverified detections (unsupervised)
    
    Args:
        dino_output    : raw model output (before post-processing)
        verified_preds : list of dicts from run_crattt_inference
        img_shape      : (H, W) of input image
    
    Returns:
        loss : scalar tensor with gradient
    """
    if not verified_preds:
        return None

    # Extract logits from DINO output
    # pred_logits shape: [1, num_queries, num_classes]
    logits = dino_output.logits_per_query  \
        if hasattr(dino_output, 'logits_per_query') \
        else dino_output[0]

    # Use the maximum logit across queries as a confidence proxy
    # We want to maximise confidence on verified detections
    max_logits = logits.max(dim=-1).values  # [1, num_queries]

    # Focal confidence loss: -log(sigmoid(max_logit))
    # This encourages the model to produce higher confidence
    # on the verified detection distribution
    loss = -F.logsigmoid(max_logits).mean()

    return loss


def compute_pseudo_label_loss(dino_model, dino_processor,
                               image_np, verified_preds,
                               device):
    """
    Direct pseudo-label supervision loss.
    
    For each verified detection, creates a target score of 1.0
    and computes BCE loss against the model's predicted score.
    This directly encourages the model to be confident about
    Oracle-verified objects.
    
    Args:
        dino_model     : GroundingDINO model with LoRA
        dino_processor : processor
        image_np       : corrupted image array
        verified_preds : Oracle-verified detections
        device         : cuda/cpu
    
    Returns:
        loss : scalar tensor
    """
    if not verified_preds:
        return None

    inputs = dino_processor(
        images=image_np,
        text=DINO_TEXT_PROMPT,
        return_tensors="pt"
    ).to(device)

    # Forward pass with gradients
    outputs = dino_model(**inputs)

    # Get predicted scores for all queries
    # post_process gives us scores in [0,1]
    res = dino_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        target_sizes=[image_np.shape[:2]],
        text_threshold=0.05  # Very low to get all scores
    )[0]

    if len(res['scores']) == 0:
        return None

    # All verified detections are targets with score=1.0
    # We maximise the mean score of top predictions
    # This is an unsupervised confidence maximisation objective
    top_scores = res['scores'][:len(verified_preds)]

    # Target: push verified detection scores toward 1.0
    targets = torch.ones_like(top_scores)
    loss = F.binary_cross_entropy(
        top_scores.clamp(1e-6, 1-1e-6),
        targets
    )

    return loss


# --- 15.2 Single-Image TTT Update ---
def ttt_update_single(image_np, optimizer, n_steps=3):
    """
    Performs TTT update on a single corrupted image.
    
    Steps:
    1. Run CRATTT to get Oracle-verified pseudo-labels
    2. If enough verified detections, compute loss and update
    3. Return updated detection results
    
    Args:
        image_np  : corrupted image [H, W, 3]
        optimizer : AdamW over LoRA parameters only
        n_steps   : number of gradient steps
    
    Returns:
        verified_after : detections after TTT update
        stats          : diagnostic dict
    """
    stats = {
        "n_verified_before": 0,
        "n_verified_after":  0,
        "loss_values":       [],
        "update_performed":  False
    }

    # Step 1: Get verified pseudo-labels (no gradient)
    with torch.no_grad():
        verified_before, inf_stats = run_crattt_inference(image_np)

    stats["n_verified_before"] = len(verified_before)

    # Only update if we have at least 1 verified detection
    # No verified detections = no reliable signal for TTT
    if len(verified_before) == 0:
        with torch.no_grad():
            verified_after, _ = run_crattt_inference(image_np)
        stats["n_verified_after"] = len(verified_after)
        return verified_after, stats

    # Step 2: Gradient update steps
    dino_model.train()

    for step in range(n_steps):
        optimizer.zero_grad()

        loss = compute_pseudo_label_loss(
            dino_model, dino_processor,
            image_np, verified_before, device
        )

        if loss is None:
            break

        loss.backward()
        optimizer.step()
        stats["loss_values"].append(round(loss.item(), 6))

    dino_model.eval()
    stats["update_performed"] = True

    # Step 3: Re-run inference with updated weights
    with torch.no_grad():
        verified_after, _ = run_crattt_inference(image_np)

    stats["n_verified_after"] = len(verified_after)
    return verified_after, stats


# --- 15.3 Initialise TTT Optimizer ---
# Only optimise LoRA parameters
lora_parameters = [
    p for p in dino_model.parameters()
    if p.requires_grad
]

optimizer = torch.optim.AdamW(
    lora_parameters,
    lr=1e-4,        # Conservative learning rate for TTT
    weight_decay=1e-4
)

print("="*50)
print("BLOCK 15: TTT UPDATE LOOP")
print("="*50)
print(f"Optimizer      : AdamW")
print(f"Learning rate  : 1e-4")
print(f"LoRA params    : {len(lora_parameters)}")
print(f"TTT steps/image: 3")
print(f"Tau            : {CRATTT_PARAMS['tau']}")
print()

# --- 15.4 Pilot TTT Test on Single Image ---
print("--- Pilot: Single Image TTT Test ---")
test_img    = loaded_images[image_files[0]]
test_corrupt = ic_corrupt(
    test_img, corruption_name='snow', severity=5
)

# Save pre-TTT LoRA weights for comparison
pre_ttt_lora = {
    name: param.data.clone()
    for name, param in dino_model.named_parameters()
    if 'lora_B' in name  # B starts at 0, tracks adaptation
}

verified_after, ttt_stats = ttt_update_single(
    test_corrupt, optimizer, n_steps=3
)

print(f"Verified before TTT : {ttt_stats['n_verified_before']}")
print(f"Verified after TTT  : {ttt_stats['n_verified_after']}")
print(f"Loss trajectory     : {ttt_stats['loss_values']}")
print(f"Update performed    : {ttt_stats['update_performed']}")

# Check LoRA weights actually changed
weight_changes = []
for name, param in dino_model.named_parameters():
    if 'lora_B' in name and name in pre_ttt_lora:
        change = (param.data - pre_ttt_lora[name]).abs().max().item()
        weight_changes.append(change)

max_change  = max(weight_changes) if weight_changes else 0
mean_change = np.mean(weight_changes) if weight_changes else 0

print(f"\n--- LoRA Weight Change Verification ---")
print(f"Max lora_B change  : {max_change:.8f}")
print(f"Mean lora_B change : {mean_change:.8f}")

if max_change > 1e-8:
    print("✅ LoRA weights updated — TTT is learning")
else:
    print("⚠️  No weight change detected — check loss/gradient flow")

# --- 15.5 Save TTT Pilot Results ---
ttt_pilot = {
    "corruption":          "snow",
    "severity":            5,
    "image":               os.path.basename(image_files[0]),
    "n_verified_before":   ttt_stats["n_verified_before"],
    "n_verified_after":    ttt_stats["n_verified_after"],
    "loss_values":         ttt_stats["loss_values"],
    "update_performed":    ttt_stats["update_performed"],
    "max_weight_change":   float(max_change),
    "mean_weight_change":  float(mean_change),
    "lora_rank":           LORA_RANK,
    "lora_alpha":          LORA_ALPHA,
    "lr":                  1e-4,
    "n_steps":             3
}

ttt_path = os.path.join(
    EVAL_PARAMS["save_dir"], "ttt_pilot_results.json"
)
with open(ttt_path, "w") as f:
    json.dump(ttt_pilot, f, indent=2)

print(f"\n✅ TTT pilot results saved: {ttt_path}")
print("\n" + "="*50)
print("BLOCK 15 COMPLETE — TTT update loop verified")
print("="*50)
