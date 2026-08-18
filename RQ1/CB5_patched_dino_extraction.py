# ============================================================
# BLOCK 5: Patched DINO Extraction + Shared Clean Baseline
# ============================================================

BOX_THRESHOLD = 0.25

category_order = list(COCO_MAP.keys())
tokenizer = dino_processor.tokenizer
tok_with_offsets = tokenizer(
    DINO_TEXT_PROMPT, return_offsets_mapping=True,
    add_special_tokens=True, return_tensors=None
)
offsets = tok_with_offsets["offset_mapping"]
token_ids_ref = tok_with_offsets["input_ids"]

_sanity_inputs = dino_processor(
    images=loaded_images[image_files[0]], text=DINO_TEXT_PROMPT,
    return_tensors="pt"
).to(device)
if token_ids_ref != _sanity_inputs.input_ids[0].cpu().tolist():
    raise RuntimeError("Tokenization mismatch — stop and investigate.")
print(f"✅ Tokenization confirmed identical to model inputs ({len(token_ids_ref)} tokens)")

def build_category_token_spans(text_prompt, category_names, offsets):
    spans, not_found, search_start = {}, [], 0
    for cat in category_names:
        idx = text_prompt.find(cat, search_start)
        if idx == -1:
            idx = text_prompt.lower().find(cat.lower(), search_start)
        if idx == -1:
            spans[cat] = []; not_found.append(cat); continue
        start_char, end_char = idx, idx + len(cat)
        search_start = end_char
        spans[cat] = [i for i, (s, e) in enumerate(offsets)
                       if not (s == 0 and e == 0) and s < end_char and e > start_char]
    return spans, not_found

category_token_spans, not_found = build_category_token_spans(
    DINO_TEXT_PROMPT, category_order, offsets
)
print(f"✅ Token spans built for {len(category_order) - len(not_found)}/{len(category_order)} categories")

def get_category_distribution(outputs, category_token_spans, category_order):
    logits = outputs.logits[0]
    scores_sigmoid = logits.sigmoid()
    cols = []
    for cat in category_order:
        tids = category_token_spans.get(cat, [])
        cols.append(torch.zeros(scores_sigmoid.shape[0], device=scores_sigmoid.device)
                     if not tids else scores_sigmoid[:, tids].max(dim=-1).values)
    return torch.stack(cols, dim=-1), None

def run_dino_with_categories(image_np, box_threshold=None):
    if box_threshold is None:
        box_threshold = BOX_THRESHOLD
    dino_model.eval()
    with torch.no_grad():
        inputs = dino_processor(images=image_np, text=DINO_TEXT_PROMPT, return_tensors="pt").to(device)
        outputs = dino_model(**inputs)
    cat_scores_raw, _ = get_category_distribution(outputs, category_token_spans, category_order)
    max_raw_scores, max_cat_idx = cat_scores_raw.max(dim=-1)
    keep_indices = (max_raw_scores >= box_threshold).nonzero(as_tuple=True)[0]
    if len(keep_indices) == 0:
        return [], [], []
    img_h, img_w = image_np.shape[:2]
    pred_cxcywh = outputs.pred_boxes[0]
    cx, cy = pred_cxcywh[:,0]*img_w, pred_cxcywh[:,1]*img_h
    pw, ph = pred_cxcywh[:,2]*img_w, pred_cxcywh[:,3]*img_h
    pred_xyxy = torch.stack([cx-pw/2, cy-ph/2, cx+pw/2, cy+ph/2], dim=-1)
    boxes  = pred_xyxy[keep_indices]
    scores = max_raw_scores[keep_indices]
    labels = [category_order[i] for i in max_cat_idx[keep_indices].tolist()]
    return boxes, scores, labels

def dino_to_coco_format_patched(boxes, scores, labels, img_id):
    preds = []
    for box, score, label in zip(boxes, scores, labels):
        cat_id = COCO_MAP.get(label)
        if cat_id is None:
            continue
        b = box.tolist()
        preds.append({"image_id": img_id, "category_id": cat_id,
                      "bbox": [b[0], b[1], b[2]-b[0], b[3]-b[1]], "score": float(score)})
    return preds

print("✅ run_dino_with_categories ready (patched, box_threshold=0.25)")

# --- yolo_to_coco_format and compute_map were originally defined inside
# the skipped Block 5 (Section 5.1); reproduced here since Block 6
# depends on both.

from pycocotools.cocoeval import COCOeval

def yolo_to_coco_format(results, img_id):
    """
    Converts YOLO-World output to COCO evaluation format.
    """
    coco_preds = []
    names = yolo_model.names

    for box in results.boxes:
        raw_name  = names[int(box.cls[0])]
        clean_name = raw_name.lower().replace("_", " ").strip()
        category_id = COCO_MAP.get(clean_name)

        if category_id is None:
            continue

        b = box.xyxy[0].tolist()
        coco_preds.append({
            "image_id":    img_id,
            "category_id": category_id,
            "bbox": [
                b[0],
                b[1],
                b[2] - b[0],
                b[3] - b[1]
            ],
            "score": float(box.conf)
        })

    return coco_preds


def compute_map(predictions, coco_gt_obj, img_ids):
    """
    Computes mAP@0.50:0.95 using pycocotools.
    Returns (mAP, status_string).
    """
    if not predictions:
        return 0.0, "no_predictions"

    try:
        import sys
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

        dt  = coco_gt_obj.loadRes(predictions)
        ev  = COCOeval(coco_gt_obj, dt, 'bbox')
        ev.params.imgIds = img_ids
        ev.evaluate()
        ev.accumulate()
        ev.summarize()

        sys.stdout.close()
        sys.stdout = old_stdout

        return ev.stats[0], "ok"

    except Exception as e:
        try:
            sys.stdout.close()
        except:
            pass
        sys.stdout = old_stdout
        return 0.0, f"error: {e}"

print("✅ yolo_to_coco_format and compute_map ready (from original Block 5)")

# Shared clean baseline — established from the N=5000 rerun, not
# recomputed here. Both 4.2a and 4.2b now normalise against this
# same pair of values.
map_clean_dino = 0.5281
map_clean_yolo = 0.4327
print(f"✅ Clean baseline set (shared, N=5000): DINO={map_clean_dino}  YOLO={map_clean_yolo}")
