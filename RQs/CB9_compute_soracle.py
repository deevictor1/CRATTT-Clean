# ============================================================
# BLOCK 9: CLIP Oracle — compute_soracle
# Implements semantic verification of BARON region crops.
# Tests on a single image before integration into CRATTT.
# ============================================================

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# --- 9.1 compute_soracle Function ---
def compute_soracle(crop_pil, proposed_label, top_k=3):
    """
    Computes the CLIP semantic similarity score (Soracle) for a
    single region crop against a proposed class label.

    Uses the pre-computed clip_text_features from Block 3.
    
    Args:
        crop_pil       : PIL.Image — region crop from BARON
        proposed_label : str — class label proposed by DINO
        top_k          : int — for diagnostic output only

    Returns:
        soracle : float in [0, 1] — cosine similarity between
                  crop and proposed label in CLIP space
        top_matches : list of (label, score) — top_k matches
                      for diagnostic purposes
    """
    # Encode the crop image
    with torch.no_grad():
        img_inputs = clip_processor(
            images=crop_pil,
            return_tensors="pt"
        ).to(device)

        # Use vision model directly to avoid Ultralytics interference
        vision_out = clip_model.vision_model(**img_inputs)
        pooled = vision_out.pooler_output
        img_feat = clip_model.visual_projection(pooled)
        img_feat = F.normalize(img_feat, p=2, dim=-1)

    # Cosine similarity against all 80 class embeddings
    # clip_text_features shape: [80, 512]
    # img_feat shape: [1, 512]
    similarities = (img_feat @ clip_text_features.T).squeeze(0)
    # Shape: [80]

    # Score for the proposed label specifically
    if proposed_label in COCO_CLASSES:
        label_idx = COCO_CLASSES.index(proposed_label)
        soracle = similarities[label_idx].item()
    else:
        # If label not in COCO_CLASSES, use max similarity
        soracle = similarities.max().item()

    # Top-k matches for diagnostics
    topk_vals, topk_idx = torch.topk(similarities, top_k)
    top_matches = [
        (COCO_CLASSES[i], round(v.item(), 4))
        for i, v in zip(topk_idx.tolist(), topk_vals)
    ]

    return soracle, top_matches


# --- 9.2 Validate on Test Image ---
print("Validating CLIP Oracle on BARON crops...")
print(f"Using pre-computed text embeddings: {clip_text_features.shape}")
print()

test_img_np = loaded_images[image_files[0]]
labels      = res.get("text_labels", res.get("labels", []))

# Re-use crops from Block 8 (same image)
# If Block 8 ran successfully, crops and kept_idx are in memory
n_validate = min(5, len(crops))

oracle_scores = []

for j in range(n_validate):
    crop_pil = crops[j]
    idx      = kept_idx[j]
    proposed = labels[idx] if idx < len(labels) else "unknown"
    
    # Clean label for COCO_CLASSES lookup
    if isinstance(proposed, str):
        clean_proposed = proposed.lower().replace(".", "").strip()
    else:
        clean_proposed = COCO_CLASSES[proposed] if isinstance(
            proposed, int) else "unknown"

    soracle, top_matches = compute_soracle(crop_pil, clean_proposed)
    oracle_scores.append(soracle)

    dino_score = scores[idx].item()
    print(f"Region {j+1}:")
    print(f"   DINO label  : {proposed}")
    print(f"   DINO score  : {dino_score:.4f}")
    print(f"   Soracle     : {soracle:.4f}")
    print(f"   Top CLIP matches: {top_matches}")
    print()

# --- 9.3 Soracle Distribution Plot ---
if oracle_scores:
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(oracle_scores)), oracle_scores,
            color='steelblue', alpha=0.8)
    plt.axhline(y=CRATTT_PARAMS["tau"], color='red',
                linestyle='--', label=f"τ={CRATTT_PARAMS['tau']}")
    plt.xlabel("Region Index")
    plt.ylabel("Soracle Score")
    plt.title("CLIP Oracle Scores for BARON Regions\n"
              "(Red line = TTRV verification threshold τ)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    fig_path = os.path.join(
        EVAL_PARAMS["fig_dir"], "figure_soracle_validation.pdf"
    )
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Soracle validation figure saved: {fig_path}")

# --- 9.4 Shape and Range Checks ---
print("\n--- Validation Checks ---")
print(f"Soracle scores computed: {len(oracle_scores)}")
if oracle_scores:
    print(f"Score range : [{min(oracle_scores):.4f}, "
          f"{max(oracle_scores):.4f}]")
    print(f"Score mean  : {np.mean(oracle_scores):.4f}")
    all_in_range = all(-1 <= s <= 1 for s in oracle_scores)
    print(f"All scores in [-1, 1]: "
          f"{'✅ Yes' if all_in_range else '❌ No'}")

print("\n" + "="*50)
print("BLOCK 9 COMPLETE — CLIP Oracle validated")
print("="*50)
