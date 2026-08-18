# ============================================================
# BLOCK 10: External-Oracle CRATTT Inference Function
# Implements the external-oracle-gated CRATTT approach (BARON +
# CLIP Oracle + harmonic Rjoint gate). Used throughout Phase 2
# (Tables 4.3-4.6) and most of Phase 3's twelve-configuration
# ablation (Blocks 15-23c). Superseded from Phase 4 onward by the
# internal TTRV approach (b24_run_dino and related functions),
# which does not call this function.
# ============================================================

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

# --- 10.1 The Canonical run_crattt_inference ---
def run_crattt_inference(
    image_np,
    alpha=None,
    beta=None,
    tau=None
):
    """
    Full CRATTT inference with BARON + CLIP Oracle TTRV gate.

    Stage 1: GroundingDINO generates proposals at low threshold
    Stage 2: BARON extracts region crops for each proposal
    Stage 3: CLIP Oracle computes Soracle per region
    Stage 4: Harmonic product Joint Reward gates each detection

    Args:
        image_np : np.ndarray [H, W, 3] uint8
        alpha    : float — DINO exponent in Rjoint (default from CRATTT_PARAMS)
        beta     : float — Oracle exponent in Rjoint (default from CRATTT_PARAMS)
        tau      : float — verification threshold (default from CRATTT_PARAMS)

    Returns:
        verified : list of dicts with keys:
                   image_id, category_id, bbox, score,
                   label, dino_score, soracle, rjoint
        stats    : dict with diagnostic counts
    """
    # Use global defaults unless overridden
    alpha = alpha if alpha is not None else CRATTT_PARAMS["alpha"]
    beta  = beta  if beta  is not None else CRATTT_PARAMS["beta"]
    tau   = tau   if tau   is not None else CRATTT_PARAMS["tau"]

    # --- Stage 1: GroundingDINO Proposals ---
    inputs = dino_processor(
        images=image_np,
        text=DINO_TEXT_PROMPT,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = dino_model(**inputs)

    res = dino_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        target_sizes=[image_np.shape[:2]],
        text_threshold=CRATTT_PARAMS["dino_text_thr"]
    )[0]

    boxes  = res["boxes"]
    scores = res["scores"]
    labels = res.get("text_labels", res.get("labels", []))

    if len(boxes) == 0:
        return [], {
            "n_proposals": 0,
            "n_verified": 0,
            "n_rejected": 0
        }

    # --- Stage 2: BARON Region Extraction ---
    crops, valid_boxes, kept_idx = baron.extract(
        image_np, boxes, scores,
        threshold=CRATTT_PARAMS["dino_text_thr"]
    )

    if len(crops) == 0:
        return [], {
            "n_proposals": len(boxes),
            "n_verified": 0,
            "n_rejected": len(boxes)
        }

    # --- Stage 3 & 4: Oracle Verification + Rjoint Gate ---
    verified  = []
    n_rejected = 0

    for j, (crop_pil, box) in enumerate(
        zip(crops, valid_boxes)
    ):
        orig_idx = kept_idx[j]

        # Get DINO score and label for this region
        dino_score = scores[orig_idx].item()

        if orig_idx < len(labels):
            label = labels[orig_idx]
        else:
            continue

        # Clean label for COCO lookup
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

        # Compute Soracle
        soracle, _ = compute_soracle(crop_pil, clean_label)

        # Harmonic Product Joint Reward
        # Both scores must be positive for Rjoint to be non-zero
        if dino_score > 0 and soracle > 0:
            rjoint = (dino_score ** alpha) * (soracle ** beta)
        else:
            rjoint = 0.0

        # TTRV Verification Gate
        if rjoint >= tau:
            b = box.tolist()
            verified.append({
                "image_id":   0,       # placeholder; set by caller
                "category_id": category_id,
                "bbox": [
                    b[0], b[1],
                    b[2] - b[0],       # width
                    b[3] - b[1]        # height
                ],
                "score":      rjoint,
                "label":      clean_label,
                "dino_score": dino_score,
                "soracle":    soracle,
                "rjoint":     rjoint
            })
        else:
            n_rejected += 1

    stats = {
        "n_proposals": len(boxes),
        "n_regions":   len(crops),
        "n_verified":  len(verified),
        "n_rejected":  n_rejected
    }

    return verified, stats


# --- 10.2 Single Image Validation ---
print("Validating canonical CRATTT on clean test image...")
print(f"Alpha={CRATTT_PARAMS['alpha']}, "
      f"Beta={CRATTT_PARAMS['beta']}, "
      f"Tau={CRATTT_PARAMS['tau']}")
print()

test_img = loaded_images[image_files[0]]
verified, stats = run_crattt_inference(test_img)

print(f"--- Inference Stats ---")
print(f"DINO proposals : {stats['n_proposals']}")
print(f"BARON regions  : {stats['n_regions']}")
print(f"Verified (pass): {stats['n_verified']}")
print(f"Rejected (fail): {stats['n_rejected']}")
print()

if verified:
    print(f"--- Verified Detections ---")
    for v in verified[:5]:
        print(f"  {v['label']:<20} "
              f"DINO={v['dino_score']:.3f}  "
              f"Soracle={v['soracle']:.3f}  "
              f"Rjoint={v['rjoint']:.4f}")

# --- 10.3 Corrupted Image Validation ---
print(f"\nValidating on severity-5 snow corruption...")
from imagecorruptions import corrupt as ic_corrupt

corrupted = ic_corrupt(test_img, corruption_name='snow', severity=5)
verified_c, stats_c = run_crattt_inference(corrupted)

print(f"--- Corrupted Inference Stats ---")
print(f"DINO proposals : {stats_c['n_proposals']}")
print(f"BARON regions  : {stats_c['n_regions']}")
print(f"Verified (pass): {stats_c['n_verified']}")
print(f"Rejected (fail): {stats_c['n_rejected']}")

# --- 10.4 Clean vs Corrupted Comparison ---
print(f"\n--- Clean vs Corrupted ---")
print(f"Verified clean    : {stats['n_verified']}")
print(f"Verified corrupted: {stats_c['n_verified']}")
delta = stats_c['n_verified'] - stats['n_verified']
print(f"Delta             : {delta:+d}")
print(f"('Negative delta under corruption is expected')")

# --- 10.5 Rjoint Distribution Check ---
if verified:
    rjoints = [v['rjoint'] for v in verified]
    print(f"\n--- Rjoint Distribution (clean) ---")
    print(f"Min : {min(rjoints):.4f}")
    print(f"Max : {max(rjoints):.4f}")
    print(f"Mean: {np.mean(rjoints):.4f}")

print("\n" + "="*50)
print("BLOCK 10 COMPLETE — Canonical CRATTT function ready")
print("="*50)
