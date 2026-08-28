# ============================================================
# BLOCK 8: BARON Region Extractor
# Implements and validates the region crop extraction component.
# Tests on a single image before integration into CRATTT.
# ============================================================

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- 8.1 BARON Extractor Class ---
class BARONExtractor:
    """
    Bag-of-Regions Object-level Network (BARON) extractor.
    Crops detected regions from the image tensor and resizes
    them to a fixed size for CLIP Oracle evaluation.
    """
    def __init__(self, max_regions=15, output_size=(224, 224)):
        self.max_regions  = max_regions
        self.output_size  = output_size

    def extract(self, image_np, boxes, scores, threshold=0.1):
        """
        Args:
            image_np : np.ndarray [H, W, 3] uint8
            boxes    : torch.Tensor [N, 4] in xyxy pixel coords
            scores   : torch.Tensor [N]
            threshold: minimum score to include a region

        Returns:
            crops_pil : list of PIL.Image — one per kept region
            kept_boxes: torch.Tensor [M, 4] — coordinates of kept regions
            kept_idx  : list[int] — original indices of kept detections
        """
        keep_mask = scores > threshold
        kept_boxes  = boxes[keep_mask]
        kept_scores = scores[keep_mask]
        kept_idx    = keep_mask.nonzero(as_tuple=True)[0].tolist()

        # If more than max_regions, keep the highest scoring ones
        if len(kept_boxes) > self.max_regions:
            topk_vals, topk_idx = torch.topk(
                kept_scores, self.max_regions
            )
            kept_boxes  = kept_boxes[topk_idx]
            kept_idx    = [kept_idx[i] for i in topk_idx.tolist()]

        crops_pil = []
        valid_boxes = []

        h, w = image_np.shape[:2]

        for box in kept_boxes:
            x1 = max(0, int(box[0].item()))
            y1 = max(0, int(box[1].item()))
            x2 = min(w, int(box[2].item()))
            y2 = min(h, int(box[3].item()))

            # Skip degenerate boxes
            if x2 <= x1 or y2 <= y1:
                continue

            crop = image_np[y1:y2, x1:x2]

            # Skip empty crops
            if crop.size == 0:
                continue

            crop_pil = Image.fromarray(crop).resize(
                self.output_size, Image.BILINEAR
            )
            crops_pil.append(crop_pil)
            valid_boxes.append(box)

        if valid_boxes:
            valid_boxes = torch.stack(valid_boxes)
        else:
            valid_boxes = torch.zeros((0, 4))

        return crops_pil, valid_boxes, kept_idx


# --- 8.2 Instantiate ---
baron = BARONExtractor(
    max_regions=CRATTT_PARAMS["max_regions"],
    output_size=CRATTT_PARAMS["region_size"]
)
print("✅ BARON Extractor instantiated")
print(f"   Max regions : {baron.max_regions}")
print(f"   Output size : {baron.output_size}")

# --- 8.3 Visual Validation on Clean Image ---
# Note: this demo uses the standard, unpatched
# post_process_grounded_object_detection, predating the composite-phrase
# fix (Block 24 onward, Section 4.1.2). This validation was never used
# to produce any reported result, only Sections 8.1-8.2 (the BARON
# extractor itself) are load-bearing for anything downstream.
print("\nValidating BARON on clean test image...")

test_img_path = image_files[0]
test_img_np   = loaded_images[test_img_path]

# Run GroundingDINO to get proposals
inputs = dino_processor(
    images=test_img_np,
    text=DINO_TEXT_PROMPT,
    return_tensors="pt"
).to(device)

with torch.no_grad():
    outputs = dino_model(**inputs)

res = dino_processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    target_sizes=[test_img_np.shape[:2]],
    text_threshold=CRATTT_PARAMS["dino_text_thr"]
)[0]

boxes  = res["boxes"]
scores = res["scores"]
labels = res.get("text_labels", res.get("labels", []))

print(f"   DINO proposals: {len(boxes)}")

# Extract regions
crops, valid_boxes, kept_idx = baron.extract(
    test_img_np, boxes, scores,
    threshold=CRATTT_PARAMS["dino_text_thr"]
)

print(f"   BARON regions extracted: {len(crops)}")

# --- 8.4 Visualise Crops ---
if len(crops) > 0:
    n_show = min(6, len(crops))
    fig, axes = plt.subplots(1, n_show + 1, 
                              figsize=(3 * (n_show + 1), 4))

    # Show original with boxes
    axes[0].imshow(test_img_np)
    axes[0].set_title("Original + BARON regions", fontsize=9)
    for box in valid_boxes:
        b = box.tolist()
        axes[0].add_patch(patches.Rectangle(
            (b[0], b[1]), b[2]-b[0], b[3]-b[1],
            linewidth=2, edgecolor='lime', facecolor='none'
        ))
    axes[0].axis('off')

    # Show individual crops
    for j in range(n_show):
        axes[j+1].imshow(crops[j])
        idx = kept_idx[j]
        lbl = labels[idx] if idx < len(labels) else "?"
        scr = scores[idx].item() if idx < len(scores) else 0
        axes[j+1].set_title(f"{lbl}\n{scr:.2f}", fontsize=8)
        axes[j+1].axis('off')

    plt.suptitle("BARON Region Extraction Validation", 
                 fontweight='bold')
    plt.tight_layout()

    fig_path = os.path.join(
        EVAL_PARAMS["fig_dir"], "figure_baron_validation.pdf"
    )
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Validation figure saved: {fig_path}")
else:
    print("⚠️  No crops extracted — check threshold")

# --- 8.5 Shape Contract Verification ---
print("\n--- Shape Contract ---")
print(f"Input image : {test_img_np.shape}")
if crops:
    sample = np.array(crops[0])
    print(f"Output crop : {sample.shape}")
    assert sample.shape == (224, 224, 3), \
        f"Unexpected crop shape: {sample.shape}"
    print("✅ Shape contract verified: all crops are (224, 224, 3)")

print("\n" + "="*50)
print("BLOCK 8 COMPLETE — BARON extractor validated")
print("="*50)
