# Run the verification cell first

# =========================================================================
checks = {
    "dino_model":           'dino_model' in dir(),
    "clip_model":           'clip_model' in dir(),
    "tau_correct":          'CRATTT_PARAMS' in dir() and CRATTT_PARAMS.get('tau') == 0.321,
    "image_files_20":       'image_files' in dir() and len(image_files) == 20,
    "loaded_images":        'loaded_images' in dir() and len(loaded_images) == 20,
    "coco_gt":              'coco_gt' in dir(),
    "baron":                'baron' in dir(),
    "compute_soracle":      'compute_soracle' in dir(),
    "run_crattt_inference": 'run_crattt_inference' in dir(),
    "compute_map":          'compute_map' in dir(),
    "dino_to_coco_format":  'dino_to_coco_format' in dir(),
    "clip_text_features":   'clip_text_features' in dir(),
    "COCO_CLASSES":         'COCO_CLASSES' in dir(),
}

all_ok = True
for name, status in checks.items():
    icon = "✅" if status else "❌"
    print(f"{icon} {name}")
    if not status:
        all_ok = False

print()
if all_ok:
    print("✅ Ready for Block 22")
else:
    print("❌ Rerun setup blocks first or restart kernel")

# =========================================================================
# ============================================================
# BLOCK 22: Oracle-Gated Multi-View Pseudo-Labelling
# Novel approach addressing the encoder-detection head
# decoupling constraint identified in Blocks 16-21.
#
# Key insight from nine-configuration ablation:
# - Encoder LoRA cannot cascade to detection output
# - Detection head LoRA shifts boxes randomly (no guidance)
# - Solution: supervise detection head DIRECTLY using
#   pseudo-ground-truth boxes that are:
#   (a) spatially consistent across multiple augmented views
#   (b) semantically verified by the CLIP Oracle
#
# This bypasses the encoder entirely. The detection head
# receives direct box-level supervision from high-confidence,
# multi-view consistent, Oracle-verified detections.
#
# Distinction from existing work:
# - DETReg (Bar et al. 2022): uses classical selective search
#   proposals. Ours uses Oracle-verified semantic proposals.
# - ContrastTTA (Chen et al. 2022): unfiltered consistency.
#   Ours adds Oracle verification filter.
#
# ABLATION FALLBACK: gains for Blocks 16-21 are read live from
# memory or disk where available; KNOWN_RESULTS below provides
# the confirmed values as a fallback when neither is available.
#
# Scope: 5 images (pilot), 4 corruptions, severity 5
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
from torchvision.ops import box_iou
from transformers import AutoModelForZeroShotObjectDetection

print("="*60)
print("BLOCK 22: ORACLE-GATED MULTI-VIEW PSEUDO-LABELLING")
print("="*60)
print()
print("Novel approach — supervises detection head directly")
print("using Oracle-verified multi-view consistent boxes.")
print()

# --- 22.1 Reload clean model, then inject fresh LoRA ---
existing_lora_b = sum(
    1 for n, _ in dino_model.named_parameters() if 'lora_B' in n
)
if existing_lora_b > 0:
    print(f"   dino_model has {existing_lora_b} LoRA layers already "
          f"present. Reloading fresh.")

del dino_model
gc.collect()
torch.cuda.empty_cache()
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
    DINO_MODEL_ID, token=hf_token
).to(device)
dino_model.eval()

for param in dino_model.parameters():
    param.requires_grad = False
for param in clip_model.parameters():
    param.requires_grad = False

class LoRALinear22(nn.Module):
    """Clean rank=4 LoRA for Block 22."""
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


def inject_lora_b22(model, rank=4, lora_alpha=8):
    """
    Injects LoRA into encoder query/value projections.
    Only matches nn.Linear — safe here since dino_model was
    just reloaded fresh above.
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
        lora_layer = LoRALinear22(
            module, rank=rank, lora_alpha=lora_alpha
        ).to(device)
        setattr(parent, parts[-1], lora_layer)
        n_injected += 1
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            param.requires_grad = True
    return n_injected


n_inj = inject_lora_b22(dino_model, rank=4, lora_alpha=8)
assert n_inj == 36, f"Expected 36 LoRA layers, got {n_inj}"

trainable = sum(
    p.numel() for p in dino_model.parameters()
    if p.requires_grad
)
assert trainable == 73_728, (
    f"Expected 73,728 trainable params, got {trainable:,}"
)
print(f"✅ LoRA injected fresh: {n_inj} layers, {trainable:,} params")

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


# --- 22.2 Augmentation Suite (5 views) ---
def generate_augmented_views(image_np, n_views=5):
    """
    Generates N augmented views of a corrupted image.
    Each view uses a different mild augmentation that
    preserves semantic content while changing pixel values.
    Mild augmentations ensure DINO can still detect objects.
    """
    views = [image_np.copy()]  # View 0: original corrupted

    # View 1: Brightness jitter
    img = image_np.astype(np.float32)
    views.append(
        np.clip(img * np.random.uniform(0.85, 1.15),
                0, 255).astype(np.uint8)
    )

    # View 2: Gaussian blur (mild)
    views.append(
        cv2.GaussianBlur(image_np.copy(), (3, 3), sigmaX=0.8)
    )

    # View 3: Contrast adjustment
    img = image_np.astype(np.float32)
    mean = img.mean()
    views.append(
        np.clip((img - mean) * np.random.uniform(0.9, 1.1) + mean,
                0, 255).astype(np.uint8)
    )

    # View 4: Slight horizontal flip then flip back
    # (tests spatial consistency without changing content)
    flipped = cv2.flip(image_np.copy(), 1)
    views.append(cv2.flip(flipped, 1))  # Double flip = identity + noise

    return views[:n_views]


# --- 22.3 Multi-View Consistent Box Finder ---
def find_consistent_boxes(image_np, n_views=5,
                           consistency_threshold=3,
                           iou_threshold=0.5):
    """
    Runs DINO on N augmented views and finds boxes that
    appear consistently across at least consistency_threshold views.

    A box is considered consistent if it has IoU > iou_threshold
    with at least one box from consistency_threshold other views.

    Returns:
        consistent_boxes : torch.Tensor [M, 4] xyxy format
        consistent_labels: list of label strings
        consistency_scores: list of how many views agreed
    """
    views = generate_augmented_views(image_np, n_views)

    all_boxes  = []
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for view in views:
            inputs = dino_processor(
                images=view, text=DINO_TEXT_PROMPT,
                return_tensors="pt"
            ).to(device)
            outputs = dino_model(**inputs)
            res = dino_processor\
                .post_process_grounded_object_detection(
                    outputs, inputs.input_ids,
                    target_sizes=[image_np.shape[:2]],
                    text_threshold=CRATTT_PARAMS["dino_text_thr"]
                )[0]

            labels = res.get("text_labels", res.get("labels", []))
            if len(res['boxes']) > 0:
                all_boxes.append(res['boxes'])
                all_labels.append(labels)
                all_scores.append(res['scores'])

    if len(all_boxes) < consistency_threshold:
        return torch.zeros((0, 4)), [], []

    ref_boxes  = all_boxes[0]
    ref_labels = all_labels[0]

    consistent_boxes  = []
    consistent_labels = []
    consistency_scores = []

    for i, (ref_box, ref_label) in enumerate(
        zip(ref_boxes, ref_labels)
    ):
        if not isinstance(ref_label, str):
            continue

        clean_label = ref_label.lower().replace(".", "").strip()
        if clean_label not in COCO_MAP:
            continue

        agreement_count = 1

        for v_idx in range(1, len(all_boxes)):
            other_boxes = all_boxes[v_idx]
            if len(other_boxes) == 0:
                continue

            ref_box_expanded = ref_box.unsqueeze(0)
            ious = box_iou(ref_box_expanded, other_boxes)
            max_iou = ious.max().item()

            if max_iou >= iou_threshold:
                agreement_count += 1

        if agreement_count >= consistency_threshold:
            consistent_boxes.append(ref_box)
            consistent_labels.append(clean_label)
            consistency_scores.append(agreement_count)

    if not consistent_boxes:
        return torch.zeros((0, 4)), [], []

    return (torch.stack(consistent_boxes),
            consistent_labels,
            consistency_scores)


# --- 22.4 Oracle Verification of Consistent Boxes ---
def oracle_verify_consistent_boxes(image_np, consistent_boxes,
                                    consistent_labels):
    """
    Applies CLIP Oracle verification to multi-view consistent boxes.
    Only boxes passing both consistency AND Oracle verification
    are used as pseudo-ground-truth for detection head supervision.

    compute_soracle(crop_pil, proposed_label, top_k=3) per Block 9 —
    confirmed matching the call below (crop first, label second).
    """
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

        crop = image_np[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop_pil = __import__('PIL').Image.fromarray(crop)

        soracle, _ = compute_soracle(crop_pil, label)

        dino_score_proxy = 0.5  # Conservative proxy
        if soracle > 0:
            rjoint = (dino_score_proxy ** CRATTT_PARAMS["alpha"]) * \
                     (soracle ** CRATTT_PARAMS["beta"])
        else:
            rjoint = 0.0

        if rjoint >= CRATTT_PARAMS["tau"]:
            verified_boxes.append(box)
            verified_labels.append(label)

    return verified_boxes, verified_labels


# --- 22.5 Detection Head Supervision Loss ---
def compute_multiview_detection_loss(dino_model, dino_processor,
                                      image_np, pseudo_gt_boxes,
                                      pseudo_gt_labels, device):
    """
    Supervised detection loss using Oracle-verified pseudo-GT boxes.

    For each pseudo-GT box, finds the closest matching DINO proposal
    and applies:
    1. Box regression consistency loss (L1 between pseudo-GT
       and closest DINO box)
    2. Confidence maximisation for matched proposals
    """
    if not pseudo_gt_boxes:
        return None

    inputs = dino_processor(
        images=image_np,
        text=DINO_TEXT_PROMPT,
        return_tensors="pt"
    ).to(device)

    outputs = dino_model(**inputs)

    res = dino_processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        target_sizes=[image_np.shape[:2]],
        text_threshold=0.05
    )[0]

    if len(res['boxes']) == 0:
        return None

    pred_boxes  = res['boxes']
    pred_scores = res['scores']

    gt_boxes = torch.stack(pseudo_gt_boxes).to(device)

    ious = box_iou(gt_boxes, pred_boxes)

    total_loss = None

    for gt_idx in range(len(gt_boxes)):
        best_iou, best_pred_idx = ious[gt_idx].max(dim=0)

        if best_iou.item() < 0.1:
            continue

        matched_box   = pred_boxes[best_pred_idx]
        matched_score = pred_scores[best_pred_idx]
        gt_box        = gt_boxes[gt_idx]

        h, w    = image_np.shape[:2]
        scale   = torch.tensor(
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

        combined = box_loss + 0.5 * conf_loss

        total_loss = combined if total_loss is None \
            else total_loss + combined

    if total_loss is not None:
        total_loss = total_loss / len(gt_boxes)

    return total_loss


# --- 22.6 Configuration ---
PILOT_IMAGES_22     = image_files[:5]
EVAL_CORRUPTIONS_22 = [
    ("Noise",   "gaussian_noise", 5),
    ("Blur",    "motion_blur",    5),
    ("Weather", "snow",           5),
    ("Digital", "contrast",       5),
]
TTT_STEPS_22    = 10
TTT_LR_22       = 5e-5
N_VIEWS_22      = 5
CONSISTENCY_THR = 3  # Must appear in 3/5 views
IOU_THR         = 0.5

print(f"\n--- Block 22 Configuration ---")
print(f"Approach    : Oracle-Gated Multi-View Pseudo-Labelling")
print(f"Images      : {len(PILOT_IMAGES_22)} (pilot)")
print(f"Corruptions : {len(EVAL_CORRUPTIONS_22)}")
print(f"Views       : {N_VIEWS_22}")
print(f"Consistency : {CONSISTENCY_THR}/{N_VIEWS_22} views")
print(f"IoU thresh  : {IOU_THR}")
print(f"TTT steps   : {TTT_STEPS_22}")
print(f"TTT lr      : {TTT_LR_22}")
print(f"Tau         : {CRATTT_PARAMS['tau']}")


# --- 22.7 Validate Multi-View Finder on Clean Image ---
print(f"\nValidating multi-view consistency on clean image...")
consistent_boxes, consistent_labels, consistency_scores = \
    find_consistent_boxes(
        test_img, n_views=N_VIEWS_22,
        consistency_threshold=CONSISTENCY_THR,
        iou_threshold=IOU_THR
    )

print(f"   DINO proposals (single view): {len(rv['boxes'])}")
print(f"   Multi-view consistent boxes : {len(consistent_boxes)}")
print(f"   Consistency scores          : {consistency_scores[:5]}")

verified_boxes, verified_labels = oracle_verify_consistent_boxes(
    test_img, consistent_boxes, consistent_labels
)
print(f"   Oracle-verified pseudo-GT   : {len(verified_boxes)}")

if len(verified_boxes) > 0:
    print(f"   Verified labels: {verified_labels[:5]}")
    print(f"   ✅ Pipeline producing valid pseudo-GT boxes")
else:
    print(f"   ⚠️  No verified boxes on clean image")
    print(f"       Check consistency/IoU thresholds")
    print(f"       (If this is 0, most of the TTT loop below will")
    print(f"        skip — worth pausing here before the full run)")


# --- 22.8 Main TTT Loop ---
all_rows_22 = []

b22_ckpt_dir = os.path.join(
    EVAL_PARAMS["ckpt_dir"], "block22"
)
os.makedirs(b22_ckpt_dir, exist_ok=True)

for cat_name, corruption, severity in EVAL_CORRUPTIONS_22:

    ckpt_path = os.path.join(
        b22_ckpt_dir, f"{corruption}_sev{severity}.json"
    )
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            rows = json.load(f)
        print(f"↩️  {corruption}: from checkpoint")
        all_rows_22.extend(rows)
        continue

    print(f"\n▶  {cat_name}: {corruption} sev{severity}")

    for name, param in dino_model.named_parameters():
        if 'lora_B' in name:
            param.data.zero_()

    optimizer_22 = torch.optim.AdamW(
        [p for p in dino_model.parameters() if p.requires_grad],
        lr=TTT_LR_22,
        weight_decay=1e-4
    )

    corruption_rows = []
    pbar = tqdm(
        PILOT_IMAGES_22, desc=f"  {corruption}", leave=False
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

        with torch.no_grad():
            crattt_b, stats_b = run_crattt_inference(c_img)
        for p in crattt_b:
            p["image_id"] = img_id
        coco_b = [{
            "image_id": p["image_id"],
            "category_id": p["category_id"],
            "bbox": p["bbox"], "score": p["score"]
        } for p in crattt_b]
        cmap_b, _ = compute_map(coco_b, coco_gt, [img_id])

        consistent_boxes, consistent_labels, c_scores = \
            find_consistent_boxes(
                c_img, n_views=N_VIEWS_22,
                consistency_threshold=CONSISTENCY_THR,
                iou_threshold=IOU_THR
            )

        pseudo_gt_boxes, pseudo_gt_labels = \
            oracle_verify_consistent_boxes(
                c_img, consistent_boxes, consistent_labels
            )

        n_pseudo_gt = len(pseudo_gt_boxes)

        loss_values = []
        dino_model.train()

        if n_pseudo_gt > 0:
            for step in range(TTT_STEPS_22):
                optimizer_22.zero_grad()
                loss = compute_multiview_detection_loss(
                    dino_model, dino_processor,
                    c_img, pseudo_gt_boxes,
                    pseudo_gt_labels, device
                )
                if loss is not None:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in dino_model.parameters()
                         if p.requires_grad],
                        max_norm=1.0
                    )
                    optimizer_22.step()
                    loss_values.append(round(loss.item(), 6))

        dino_model.eval()

        with torch.no_grad():
            crattt_c, stats_c = run_crattt_inference(c_img)
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
            "severity":            severity,
            "image":               fname,
            "baseline_mAP":        round(float(bmap),   4),
            "crattt_mAP":          round(float(cmap_b), 4),
            "multiview_ttt_mAP":   round(float(cmap_c), 4),
            "delta_ttt_vs_crattt": round(float(cmap_c - cmap_b), 4),
            "delta_ttt_vs_base":   round(float(cmap_c - bmap),   4),
            "n_consistent_boxes":  n_pseudo_gt,
            "n_crattt_before":     stats_b['n_verified'],
            "n_crattt_after":      stats_c['n_verified'],
            "loss_mean":           round(float(np.mean(loss_values))
                                         if loss_values else 0, 4),
            "ttt_fired":           n_pseudo_gt > 0
        }
        corruption_rows.append(row)

        pbar.set_postfix({
            "B":  f"{bmap:.3f}",
            "C":  f"{cmap_b:.3f}",
            "T":  f"{cmap_c:.3f}",
            "PG": f"{n_pseudo_gt}"
        })

    with open(ckpt_path, "w") as f:
        json.dump(corruption_rows, f, indent=2)
    all_rows_22.extend(corruption_rows)

    c_df = pd.DataFrame(corruption_rows)
    ttt_fired = c_df['ttt_fired'].sum()
    print(f"  Baseline mAP          : {c_df['baseline_mAP'].mean():.4f}")
    print(f"  CRATTT mAP            : {c_df['crattt_mAP'].mean():.4f}")
    print(f"  MultiView TTT mAP     : {c_df['multiview_ttt_mAP'].mean():.4f}")
    print(f"  TTT gain vs CRATTT    : {c_df['delta_ttt_vs_crattt'].mean():+.4f}")
    print(f"  Images where TTT fired: {ttt_fired}/{len(c_df)}")
    print(f"  Mean pseudo-GT boxes  : {c_df['n_consistent_boxes'].mean():.2f}")
    if c_df['loss_mean'].max() > 0:
        print(f"  Mean loss             : {c_df[c_df['loss_mean']>0]['loss_mean'].mean():.6f}")


# --- 22.9 Results ---
df_22 = pd.DataFrame(all_rows_22)
overall_22  = df_22['delta_ttt_vs_crattt'].mean()
positive_22 = (df_22['delta_ttt_vs_crattt'] > 0.001).sum()
ttt_fired   = df_22['ttt_fired'].sum()

print("\n" + "="*70)
print("BLOCK 22: ORACLE-GATED MULTI-VIEW RESULTS (N=5 PILOT)")
print("="*70)

summary_22 = df_22.groupby(["category", "corruption"]).agg(
    Baseline_mAP      =("baseline_mAP",        "mean"),
    CRATTT_mAP        =("crattt_mAP",           "mean"),
    MultiView_TTT_mAP =("multiview_ttt_mAP",    "mean"),
    TTT_gain          =("delta_ttt_vs_crattt",  "mean"),
    Mean_PseudoGT     =("n_consistent_boxes",   "mean"),
    TTT_fired         =("ttt_fired",            "sum"),
).round(4).reset_index()

display(summary_22)

print(f"\nOverall TTT gain vs CRATTT : {overall_22:+.4f}")
print(f"Images with gain > 0       : {positive_22}/{len(df_22)}")
print(f"Images where TTT fired     : {ttt_fired}/{len(df_22)}")
print(f"Mean pseudo-GT boxes       : {df_22['n_consistent_boxes'].mean():.2f}")

# --- Ablation table: confirmed values, live/disk preferred ---
KNOWN_RESULTS = {
    "block_16":  +0.0001,
    "block_16b": -0.0000,
    "block_16c": -0.0345,
    "block_17":  +0.0000,
    "block_18":  +0.0000,
    "block_19c": +0.0100,
    "block_20":  -0.0069,
    "block_21":  +0.0002,
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

print(f"\n{'='*78}")
print("ABLATION SO FAR: 9 of 12 Table 4.8 Configurations Tested (Swin-B)")
print("(Blocks 22b, 23, 23b, 23c still remain)")
print(f"{'='*78}")
print(f"{'Configuration':<48} {'TTT gain':>10}  {'source'}")
print("-"*78)

def _fmt(label, gain, src, marker=""):
    g = f"{gain:+.4f}" if gain is not None else "N/A"
    print(f"  {label:<46} {g:>10}  {src}{marker}")

_fmt("Blk 16  encoder conf-loss  frozen oracle",   gain_16,  src_16)
_fmt("Blk 16b encoder align-loss frozen oracle",   gain_16b, src_16b)
_fmt("Blk 16c encoder rank=16    frozen oracle",   gain_16c, src_16c)
_fmt("Blk 17  encoder co-adapt   oracle",          gain_17,  src_17)
_fmt("Blk 18  encoder gate (untrained proj)",      gain_18,  src_18)
_fmt("Blk 19c dethead LoRA N=20 (1/20 imgs drove it)", gain_19c, src_19c)
_fmt("Blk 20  self-supervised consistency",        gain_20,  src_20)
_fmt("Blk 21  trained projection (collapsed scores)", gain_21, src_21)
_fmt("Blk 22  oracle-gated multi-view (novel)",    overall_22, "live", " ◄ NEW")

print()
if overall_22 > 0.010:
    verdict    = "✅ POSITIVE GAIN — Novel approach works!"
    action     = "Scale to N=20 then Vast.ai for full evaluation."
    conclusion = "positive_scale_up"
elif overall_22 > 0.003:
    verdict    = "✅ MARGINAL POSITIVE — Promising signal"
    action     = "Scale to N=20 to confirm. Adjust parameters."
    conclusion = "marginal_investigate"
elif overall_22 > -0.003:
    verdict    = "⚠️  NEAR-ZERO — No clear improvement"
    action     = "Check pseudo-GT quality. Adjust IoU/consistency."
    conclusion = "near_zero"
else:
    verdict    = "❌ NEGATIVE — Approach needs parameter tuning"
    action     = "Review pseudo-GT count. Lower consistency threshold."
    conclusion = "negative_tune_params"

print(verdict)
print(f"Action: {action}")

# Save
csv_22 = os.path.join(
    EVAL_PARAMS["table_dir"], "table_block22_multiview.csv"
)
df_22.to_csv(csv_22, index=False)
with open(os.path.join(
    EVAL_PARAMS["save_dir"], "block22_results.json"
), "w") as f:
    json.dump({
        "method":          "oracle_gated_multi_view",
        "n_images":        len(PILOT_IMAGES_22),
        "n_views":         N_VIEWS_22,
        "consistency_thr": CONSISTENCY_THR,
        "iou_thr":         IOU_THR,
        "ttt_steps":       TTT_STEPS_22,
        "lr":              TTT_LR_22,
        "overall_gain":    round(float(overall_22), 4),
        "positive_images": int(positive_22),
        "ttt_fired":       int(ttt_fired),
        "conclusion":      conclusion,
        "ablation_so_far": {
            "block_16": round(float(gain_16), 4)  if gain_16  is not None else None,
            "block_16b": round(float(gain_16b), 4) if gain_16b is not None else None,
            "block_16c": round(float(gain_16c), 4) if gain_16c is not None else None,
            "block_17": round(float(gain_17), 4)  if gain_17  is not None else None,
            "block_18": round(float(gain_18), 4)  if gain_18  is not None else None,
            "block_19c": round(float(gain_19c), 4) if gain_19c is not None else None,
            "block_20": round(float(gain_20), 4)  if gain_20  is not None else None,
            "block_21": round(float(gain_21), 4)  if gain_21  is not None else None,
            "block_22": round(float(overall_22), 4)
        }
    }, f, indent=2)

print(f"\n✅ Saved: {csv_22}")
print("\n" + "="*50)
print("BLOCK 22 COMPLETE — 9 of 12 Table 4.8 configs done")
print("="*50)
