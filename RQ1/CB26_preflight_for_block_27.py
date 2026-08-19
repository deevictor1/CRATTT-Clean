# ==== PHASE 6 ====

# ─────────────────────────────────────────────────────────────
# Pre-flight check for the patch below
# ─────────────────────────────────────────────────────────────
required_p6_patch = ["category_order", "category_token_spans", "get_category_distribution"]
missing = [f for f in required_p6_patch if f not in globals()]
if missing:
    print(f"❌ Missing: {missing} — re-run the Section C prerequisite cell before the patch")
else:
    print("✅ All three present — safe to run the patch cell directly")

# ============================================================
# PATCHED b24_run_dino — fixes composite-phrase silent-drop bug
# Replaces the library's post_process_grounded_object_detection
# (which merges adjacent categories into composite phrases like
# "chair couch") with clean per-category argmax extraction.
# Same signature as before, so every downstream block that calls
# b24_run_dino inherits the fix automatically.
#
# CONFIRMED IMPACT (N=100, Swin-B): mean mAP 0.6120 -> 0.6711
# (+0.0591, Wilcoxon p=0.000004), 28/100 images improved, 0 worse.
#
# Idempotent redefinition — this is the same patch already applied
# before Block 24 (Section A). Reapplying it here is harmless and
# keeps Phase 6 self-contained if run independently.
# ============================================================

BOX_THRESHOLD = 0.25  # confirmed correct box-confidence gate

def b24_run_dino(image_np: np.ndarray):
    dino_model.eval()
    with torch.no_grad():
        inputs = dino_processor(images=image_np, text=DINO_TEXT_PROMPT, return_tensors="pt").to(device)
        outputs = dino_model(**inputs)

    cat_scores_raw, _ = get_category_distribution(outputs, category_token_spans, category_order)
    max_raw_scores, max_cat_idx = cat_scores_raw.max(dim=-1)
    keep_mask = max_raw_scores >= BOX_THRESHOLD
    keep_indices = keep_mask.nonzero(as_tuple=True)[0]

    if len(keep_indices) == 0:
        return torch.empty((0, 4)), torch.empty((0,)), []

    img_h, img_w = image_np.shape[:2]
    pred_cxcywh = outputs.pred_boxes[0]
    cx = pred_cxcywh[:, 0] * img_w
    cy = pred_cxcywh[:, 1] * img_h
    pw = pred_cxcywh[:, 2] * img_w
    ph = pred_cxcywh[:, 3] * img_h
    pred_xyxy = torch.stack([cx - pw/2, cy - ph/2, cx + pw/2, cy + ph/2], dim=-1)

    boxes  = pred_xyxy[keep_indices]
    scores = max_raw_scores[keep_indices]
    labels = [category_order[i] for i in max_cat_idx[keep_indices].tolist()]

    return boxes, scores, labels

print("✅ b24_run_dino patched -- clean argmax extraction, box_threshold=0.25")
print("   Every downstream block (Block 5, 24-24f, b24_compute_snr/b24_apply_gate)")
print("   now inherits the fix automatically.")

# ─────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────
import inspect
_src = inspect.getsource(b24_run_dino)
if "post_process_grounded_object_detection" in _src:
    print("❌ WARNING: still unpatched")
elif "BOX_THRESHOLD" in _src:
    print("✅ Confirmed: patched version active")
