"""
LVIS PILOT ROBUSTNESS EVALUATION

Extends the ImageNet-C robustness sweep (Table 4.2a/4.2b, MS-COCO) to
LVIS, a larger, long-tailed dataset with 1,203 categories, to check
whether the DINO-vs-YOLO robustness pattern generalises beyond COCO.
Produces Table 4.2c and the LVIS Pilot decay figure.

Uses "federated" querying, the correct LVIS evaluation protocol, which
queries only the categories actually annotated as present for a given
image, rather than the full 1,203-category vocabulary. A chunked
full-vocabulary alternative was tested and found non-viable at scale
(36+ hours at N=100 alone), confirmed via a small N=5 smoke test before
committing to the N=100 pilot run.
"""

# ======================================================================
# CELL 0
# ======================================================================
# CELL 1 — installs
# -----------------------------------------------------------------------------
# lvis-api is the LVIS-specific counterpart to pycocotools; it handles the
# federated annotation logic (rare/common/frequent categories, per-image
# negative category sets) that pycocotools does not.

!pip install lvis imagecorruptions ultralytics --quiet

# ======================================================================
# CELL 1
# ======================================================================
# CELL 2 — imports
# -----------------------------------------------------------------------------
import os
import time
import json
import random
import zipfile
import urllib.request
 
import numpy as np
from PIL import Image
from lvis import LVIS
from imagecorruptions import corrupt
 
# lvis-api (last updated ~2020) uses np.float, np.int, np.bool, np.object —
# all deprecated in NumPy 1.20, removed entirely in recent versions. These
# are pure aliases for Python builtins, so restoring them is a no-op
# semantically, not a behaviour change. Needed before any LVISEval call
# (Cells 10b, 12), patching here in Cell 2 covers both.
# Comprehensive patch for legacy numpy aliases removed in NumPy 2.0.
# imagecorruptions (~2019) and lvis-api (~2020) both predate this removal
# and use several of these across different functions — we've already hit
# np.float (lvis-api's eval.py) and np.float_ (imagecorruptions' fog via
# plasma_fractal), so patching the full known set now rather than
# discovering the rest one crash at a time through the remaining corruptions.
_np_legacy_aliases = {
    "float": float, "int": int, "bool": bool, "object": object,
    "str": str, "long": int, "unicode": str,
    "float_": np.float64, "complex_": np.complex128,
    "longfloat": np.longdouble, "singlecomplex": np.complex64,
    "cfloat": np.complex128, "longcomplex": np.clongdouble,
    "clongfloat": np.clongdouble, "string_": np.bytes_,
    "unicode_": np.str_, "Inf": np.inf, "Infinity": np.inf,
    "NAN": np.nan, "NaN": np.nan, "infty": np.inf,
    "NINF": -np.inf, "NZERO": -0.0, "PZERO": 0.0,
    "bool8": np.bool_,
}
for _name, _replacement in _np_legacy_aliases.items():
    if not hasattr(np, _name):
        setattr(np, _name, _replacement)
 
# imagecorruptions' glass_blur calls skimage.filters.gaussian(multichannel=True),
# a parameter scikit-image removed in favour of channel_axis. This is likely
# why Table 4.2b footnotes glass_blur as a special case — the pipeline that
# produced it likely hit this same break. Patching the reference
# imagecorruptions.corruptions actually calls (not skimage.filters.gaussian
# itself — corruptions.py already bound the old name at import time, patching
# skimage's copy afterward wouldn't reach it).
import imagecorruptions.corruptions as _ic_corruptions
from skimage.filters import gaussian as _skimage_gaussian
 
def _gaussian_compat(image, sigma, multichannel=None, **kwargs):
    if multichannel is not None:
        kwargs["channel_axis"] = -1 if multichannel else None
    return _skimage_gaussian(image, sigma=sigma, **kwargs)
 
_ic_corruptions.gaussian = _gaussian_compat
 
random.seed(42)  # reproducible image selection
 

# ======================================================================
# CELL 2
# ======================================================================
# CELL 3 — download LVIS v1 val annotations (skip if already cached
# from a Kaggle dataset — check Kaggle Datasets search for
# "lvis" first, it's often faster than pulling from fbaipublicfiles fresh
# each session)
# -----------------------------------------------------------------------------
LVIS_ANN_URL = "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip"
ANN_DIR = "/kaggle/working/lvis_annotations"
ANN_ZIP = os.path.join(ANN_DIR, "lvis_v1_val.json.zip")
ANN_JSON = os.path.join(ANN_DIR, "lvis_v1_val.json")
 
os.makedirs(ANN_DIR, exist_ok=True)
 
if not os.path.exists(ANN_JSON):
    print("Downloading LVIS v1 val annotations (~190MB zipped)...")
    urllib.request.urlretrieve(LVIS_ANN_URL, ANN_ZIP)
    with zipfile.ZipFile(ANN_ZIP, "r") as z:
        z.extractall(ANN_DIR)
    print("Done.")
else:
    print("Annotations already present, skipping download.")
 
lvis = LVIS(ANN_JSON)
print(f"Loaded LVIS val: {len(lvis.get_img_ids())} images, "
      f"{len(lvis.get_cat_ids())} categories")

# ======================================================================
# CELL 3
# ======================================================================
# CELL 4 — select N=5 images, biased toward including at least one
# rare-category instance (this is the failure mode most likely to expose
# federated-evaluation bugs, per the earlier discussion — a random sample
# risks missing tail categories entirely at this small N)
# -----------------------------------------------------------------------------
N_SMOKE = 5
 
cats = lvis.load_cats(lvis.get_cat_ids())
rare_cat_ids = [c["id"] for c in cats if c["frequency"] == "r"]
common_cat_ids = [c["id"] for c in cats if c["frequency"] in ("c", "f")]
 
print(f"{len(rare_cat_ids)} rare categories, {len(common_cat_ids)} common/frequent")
 
# grab image ids containing at least one rare-category annotation
rare_img_ids = set()
for cat_id in random.sample(rare_cat_ids, min(20, len(rare_cat_ids))):
    ann_ids = lvis.get_ann_ids(cat_ids=[cat_id])
    for ann_id in ann_ids[:5]:
        ann = lvis.load_anns([ann_id])[0]
        rare_img_ids.add(ann["image_id"])
    if len(rare_img_ids) >= 2:
        break
 
rare_img_ids = list(rare_img_ids)[:2]
all_img_ids = lvis.get_img_ids()
remaining_needed = N_SMOKE - len(rare_img_ids)
other_img_ids = random.sample(
    [i for i in all_img_ids if i not in rare_img_ids], remaining_needed
)
 
smoke_img_ids = rare_img_ids + other_img_ids
smoke_images = lvis.load_imgs(smoke_img_ids)
 
print(f"Selected {len(smoke_images)} images "
      f"({len(rare_img_ids)} deliberately include a rare category)")
 

# ======================================================================
# CELL 4
# ======================================================================
# CELL 5 — download the actual image files
# -----------------------------------------------------------------------------
IMG_DIR = "/kaggle/working/lvis_smoke_images"
os.makedirs(IMG_DIR, exist_ok=True)
 
local_paths = []
for img in smoke_images:
    url = img["coco_url"]  # LVIS reuses COCO's image files directly
    fname = os.path.join(IMG_DIR, os.path.basename(url))
    if not os.path.exists(fname):
        urllib.request.urlretrieve(url, fname)
    local_paths.append(fname)
    print(f"  {fname}")
 
print(f"\n{len(local_paths)} images ready.")
 

# ======================================================================
# CELL 5
# ======================================================================
# CELL 6 — apply brightness corruption, severity 1
# -----------------------------------------------------------------------------
def load_and_corrupt(path, corruption_name="brightness", severity=1):
    img = np.array(Image.open(path).convert("RGB"))
    corrupted = corrupt(img, corruption_name=corruption_name, severity=severity)
    return Image.fromarray(corrupted.astype(np.uint8))
 
corrupted_images = [load_and_corrupt(p) for p in local_paths]
print(f"Corrupted {len(corrupted_images)} images at brightness severity 1.")

# ======================================================================
# CELL 6
# ======================================================================
# CELL 7 — model loading + composite-phrase patch, generalised for LVIS
# -----------------------------------------------------------------------------
# Three changes from the original BLOCK V5 / V7b code:
#   1. grounding-dino-tiny -> grounding-dino-base (now using Swin-B)
#   2. DINO_TEXT_PROMPT / COCO_MAP were hardcoded to 80 COCO categories;
#      here they're built per-call so the same patch works for both the
#      federated and chunked-full-vocabulary LVIS query modes
#   3. run_dino_with_categories now takes category_names as an argument
#      instead of relying on fixed globals
 
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
 
device = "cuda" if torch.cuda.is_available() else "cpu"
 
DINO_MODEL_ID = "IDEA-Research/grounding-dino-base"  # Swin-B, not -tiny
BOX_THRESHOLD = 0.25  # matches the Vast.ai pipeline default
 
print(f"Loading GroundingDINO (Swin-B) from {DINO_MODEL_ID}...")
dino_processor = AutoProcessor.from_pretrained(DINO_MODEL_ID)
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL_ID).to(device)
dino_model.eval()
for param in dino_model.parameters():
    param.requires_grad = False
print(f"✅ GroundingDINO (Swin-B): {sum(p.numel() for p in dino_model.parameters()):,} params")
 
 
def build_dino_prompt(category_names):
    """Same format as the Vast.ai DINO_TEXT_PROMPT: lowercase, ' . '-joined."""
    return " . ".join(c.lower() for c in category_names) + " ."
 
 
def build_category_token_spans(text_prompt, category_names_lower, offsets):
    """Unchanged patch logic from BLOCK V7b (crattt-clean-2.ipynb,
    cells 9/12) — per-category argmax over token spans, not the buggy
    default post-processing that merges adjacent phrases."""
    spans, not_found, search_start = {}, [], 0
    for cat in category_names_lower:
        idx = text_prompt.find(cat, search_start)
        if idx == -1:
            spans[cat] = []
            not_found.append(cat)
            continue
        start_char, end_char = idx, idx + len(cat)
        search_start = end_char
        spans[cat] = [i for i, (s, e) in enumerate(offsets)
                       if not (s == 0 and e == 0) and s < end_char and e > start_char]
    return spans, not_found
 
 
def get_category_distribution(outputs, category_token_spans, category_names_lower):
    logits = outputs.logits[0]
    scores_sigmoid = logits.sigmoid()
    cols = []
    for cat in category_names_lower:
        tids = category_token_spans.get(cat, [])
        cols.append(torch.zeros(scores_sigmoid.shape[0], device=scores_sigmoid.device)
                     if not tids else scores_sigmoid[:, tids].max(dim=-1).values)
    cat_scores_raw = torch.stack(cols, dim=-1)
    return cat_scores_raw, torch.softmax(cat_scores_raw, dim=-1)
 
 
def run_dino_with_categories(image_np, category_names, box_threshold=None):
    """Same patched detection logic as the Vast.ai run_dino_with_categories,
    built to take a category list per call rather than a fixed global, so
    the smoke test can swap between federated and chunked-full-vocab modes."""
    if box_threshold is None:
        box_threshold = BOX_THRESHOLD
 
    cat_lower = [c.lower() for c in category_names]
    text_prompt = build_dino_prompt(category_names)
 
    tok = dino_processor.tokenizer(text_prompt, return_offsets_mapping=True,
                                     add_special_tokens=True, return_tensors=None)
    if len(tok["input_ids"]) > 256:
        print(f"  ⚠️  prompt is {len(tok['input_ids'])} tokens — GroundingDINO's "
              f"max_text_len is 256, this call will crash, reduce chunk size")
    offsets = tok["offset_mapping"]
 
    dino_model.eval()
    with torch.no_grad():
        inputs = dino_processor(images=image_np, text=text_prompt, return_tensors="pt").to(device)
        outputs = dino_model(**inputs)
 
    category_token_spans, not_found = build_category_token_spans(text_prompt, cat_lower, offsets)
    if not_found:
        print(f"  ⚠️  {len(not_found)}/{len(cat_lower)} categories not found in prompt")
 
    cat_scores_raw, _ = get_category_distribution(outputs, category_token_spans, cat_lower)
    max_raw_scores, max_cat_idx = cat_scores_raw.max(dim=-1)
    keep_indices = (max_raw_scores >= box_threshold).nonzero(as_tuple=True)[0]
    if len(keep_indices) == 0:
        return [], [], []
 
    img_h, img_w = image_np.shape[:2]
    pred_cxcywh = outputs.pred_boxes[0]
    cx, cy = pred_cxcywh[:, 0]*img_w, pred_cxcywh[:, 1]*img_h
    pw, ph = pred_cxcywh[:, 2]*img_w, pred_cxcywh[:, 3]*img_h
    pred_xyxy = torch.stack([cx-pw/2, cy-ph/2, cx+pw/2, cy+ph/2], dim=-1)
    boxes = pred_xyxy[keep_indices]
    scores = max_raw_scores[keep_indices]
    labels = [category_names[i] for i in max_cat_idx[keep_indices].tolist()]
    return boxes, scores, labels
 
print("✅ run_dino_with_categories ready (LVIS-generalised, Swin-B, box_threshold=0.25)")
 
 
# YOLO-World — unchanged from BLOCK V5, but NOTE: the actual
# prediction-calling wrapper used elsewhere in the pipeline was not
# available when this was written, only the loading call was. The
# predict line below is standard ultralytics usage, not verified
# against the real inference code — check this before trusting its
# timing number.
print("\nLoading YOLO-World-large...")
from ultralytics import YOLOWorld
yolo_model = YOLOWorld("yolov8x-worldv2.pt")  # corrected — matches the actual Table 4.2a/4.2b/N=5000 methodology
print("✅ YOLO-World-large loaded (classes set per-call in Cell 8)")
 

# ======================================================================
# CELL 7
# ======================================================================
# CELL 8 — timed inference, federated vs. chunked-full-vocabulary
# -----------------------------------------------------------------------------
# "Full vocabulary" here means chunked: 1,203 categories split into batches
# that each fit under GroundingDINO's 512-token limit, one forward pass per
# chunk. A single prompt with all 1,203 names isn't a valid comparison,
# it would silently truncate — see chat for why.
 
MAX_TEXT_LEN = 256  # GroundingDINO's actual internal limit (not the 512 I
                     # quoted earlier — that's the underlying BERT's raw
                     # capacity, but GroundingDINO's own contrastive
                     # embedding layer imposes this tighter cap, and the HF
                     # processor does not auto-truncate to it, it errors)
SAFE_MARGIN = 220    # stay under MAX_TEXT_LEN with room for special tokens
 
def build_token_aware_chunks(category_names, max_tokens=SAFE_MARGIN):
    """Groups categories into chunks by actual tokenized length rather than
    a fixed category count, since name length varies enough across LVIS
    (short: 'chair', long: 'identification_card') that a fixed count isn't
    a reliable safety margin. Runs once over the full vocabulary, not
    per-image, so the extra tokenization cost here is negligible."""
    chunks, current_chunk = [], []
    for cat in category_names:
        candidate = current_chunk + [cat]
        n_tokens = len(dino_processor.tokenizer(
            build_dino_prompt(candidate), add_special_tokens=True
        )["input_ids"])
        if n_tokens > max_tokens and current_chunk:
            chunks.append(current_chunk)
            current_chunk = [cat]
        else:
            current_chunk = candidate
    if current_chunk:
        chunks.append(current_chunk)
    return chunks
 
def get_federated_categories(img_id, lvis_obj):
    """Categories LVIS actually annotates as present for this specific
    image — the narrow, correct-for-LVIS query mode. Per-image category
    counts here are small enough (typically well under 30) that this mode
    doesn't need chunk-awareness, it stays far below MAX_TEXT_LEN on its own."""
    ann_ids = lvis_obj.get_ann_ids(img_ids=[img_id])
    anns = lvis_obj.load_anns(ann_ids)
    pos_cat_ids = set(a["category_id"] for a in anns)
    cat_id_to_name = {c["id"]: c["name"] for c in cats}
    return [cat_id_to_name[cid] for cid in pos_cat_ids]
 
all_category_names = [c["name"] for c in cats]
print("Building token-aware chunks (one-time pass over full vocabulary)...")
category_chunks = build_token_aware_chunks(all_category_names)
chunk_sizes = [len(c) for c in category_chunks]
print(f"Full vocabulary split into {len(category_chunks)} chunks, "
      f"sizes ranging {min(chunk_sizes)}-{max(chunk_sizes)} categories "
      f"(vs. the fixed 150 that just crashed)\n")
 
results = {"chunked_full_vocab": [], "federated": []}
 
for img_pil, img_meta in zip(corrupted_images, smoke_images):
    img_np = np.array(img_pil)
 
    federated_cats = get_federated_categories(img_meta["id"], lvis)
    if not federated_cats:
        federated_cats = all_category_names[:5]  # shouldn't normally trigger, given Cell 4's rare-category selection
 
    # --- chunked full-vocabulary: one forward pass per chunk, summed ---
    t0 = time.perf_counter()
    all_labels = []
    for chunk in category_chunks:
        _, _, labels = run_dino_with_categories(img_np, chunk)
        all_labels.extend(labels)
    t1 = time.perf_counter()
    results["chunked_full_vocab"].append(t1 - t0)
 
    # --- federated: one forward pass, only this image's relevant categories ---
    t0 = time.perf_counter()
    _, _, fed_labels = run_dino_with_categories(img_np, federated_cats)
    t1 = time.perf_counter()
    results["federated"].append(t1 - t0)
 
    print(f"image {img_meta['id']}: {len(federated_cats)} federated cats, "
          f"{len(category_chunks)} chunks for full-vocab | "
          f"chunked_full={results['chunked_full_vocab'][-1]:.2f}s  "
          f"federated={results['federated'][-1]:.2f}s | "
          f"detections: federated={len(fed_labels)}, chunked_full={len(all_labels)}")

# ======================================================================
# CELL 8
# ======================================================================
# CELL 9 — summarise and extrapolate to the full N=100 / 75-config pilot
# -----------------------------------------------------------------------------
def summarise(times, label):
    if not times or all(t == 0 for t in times):
        print(f"{label}: no timing data yet")
        return None
    mean_t = np.mean(times)
    print(f"{label}: mean {mean_t:.3f}s/image (GroundingDINO only — add YOLO-World "
          f"timing separately once its predict call is confirmed)")
    return mean_t
 
mean_chunked = summarise(results["chunked_full_vocab"], "Chunked full-vocabulary query")
mean_fed = summarise(results["federated"], "Federated query")
 
if mean_chunked and mean_fed:
    ratio = mean_chunked / mean_fed
    print(f"\nChunked full-vocabulary query is {ratio:.1f}x the cost of federated query.")
    print(f"({len(category_chunks)} forward passes vs. 1, so this ratio is expected "
          f"to roughly track the chunk count — a useful sanity check on the numbers.)")
 
    for label, mean_t in [("chunked full-vocab", mean_chunked), ("federated", mean_fed)]:
        est_hours = (mean_t * 100 * 75) / 3600
        print(f"\nExtrapolated N=100, 75 configs, {label} mode: {est_hours:.1f} hours "
              f"(GroundingDINO only)")
 
# checkpoint the raw numbers in case this session ends
with open("/kaggle/working/lvis_smoke_test_results.json", "w") as f:
    json.dump({
        "image_ids": [img["id"] for img in smoke_images],
        "rare_category_images_included": len(rare_img_ids),
        "timings": results,
    }, f, indent=2)
 
print("\nSaved results to /kaggle/working/lvis_smoke_test_results.json")

# ======================================================================
# CELL 9
# ======================================================================
# =============================================================================
# REAL PILOT — N=100, all 75 configs, federated mode only
#
# Chunked full-vocab is excluded here entirely — confirmed non-viable at
# scale (36+ hours at N=100 alone). Federated mode is both faster AND the
# methodologically correct LVIS evaluation protocol, so there's no
# tradeoff being made by dropping the other mode for this run.
#
# Checkpoints per corruption/severity as it goes. If this session dies
# partway through, re-running Cell 11 resumes from the last completed
# config rather than starting over.
# =============================================================================
 
 
# CELL 10 — select N=100 pilot images (separate from the 5 smoke-test images)
# -----------------------------------------------------------------------------
N_PILOT = 100
 
rare_pilot_target = 15  # roughly mirrors LVIS's own rare-category proportion
rare_pilot_img_ids = set()
for cat_id in random.sample(rare_cat_ids, min(40, len(rare_cat_ids))):
    ann_ids = lvis.get_ann_ids(cat_ids=[cat_id])
    for ann_id in ann_ids[:3]:
        ann = lvis.load_anns([ann_id])[0]
        rare_pilot_img_ids.add(ann["image_id"])
    if len(rare_pilot_img_ids) >= rare_pilot_target:
        break
 
rare_pilot_img_ids = list(rare_pilot_img_ids)[:rare_pilot_target]
remaining_needed = N_PILOT - len(rare_pilot_img_ids)
other_pilot_img_ids = random.sample(
    [i for i in all_img_ids if i not in rare_pilot_img_ids], remaining_needed
)
pilot_img_ids = rare_pilot_img_ids + other_pilot_img_ids
pilot_images = lvis.load_imgs(pilot_img_ids)
print(f"Selected {len(pilot_images)} pilot images "
      f"({len(rare_pilot_img_ids)} include rare categories)")
 
pilot_local_paths = []
for img in pilot_images:
    url = img["coco_url"]
    fname = os.path.join(IMG_DIR, os.path.basename(url))
    if not os.path.exists(fname):
        urllib.request.urlretrieve(url, fname)
    pilot_local_paths.append(fname)
print(f"{len(pilot_local_paths)} pilot images on disk")
 
cat_name_to_id = {c["name"]: c["id"] for c in cats}
pilot_np_images = {p: np.array(Image.open(p).convert("RGB")) for p in pilot_local_paths}
img_path_to_id = {p: img["id"] for p, img in zip(pilot_local_paths, pilot_images)}

# ======================================================================
# CELL 10
# ======================================================================
# CELL 10b — clean (uncorrupted) baseline pass
# -----------------------------------------------------------------------------
# Required by the CE formula verified against Table 4.2a/4.2b:
# CE_c = (1 - mean_mAP_c) / (1 - clean_mAP), where clean_mAP is a single
# fixed reference value, the same role played by the COCO clean mAP
# (0.5281 for GroundingDINO), just re-measured on LVIS since it's a
# different dataset.
#
# No corruption applied here — same 100 images, same federated categories,
# straight through the detector as-is.
 
CLEAN_BASELINE_PATH = "/kaggle/working/lvis_clean_baseline.json"
 
if os.path.exists(CLEAN_BASELINE_PATH):
    with open(CLEAN_BASELINE_PATH) as f:
        clean_detections = json.load(f)
    print(f"Loaded {len(clean_detections)} clean-baseline detections from checkpoint")
else:
    clean_detections = []
    t0 = time.perf_counter()
    for path in pilot_local_paths:
        img_id = img_path_to_id[path]
        federated_cats = get_federated_categories(img_id, lvis)
        if not federated_cats:
            continue
        boxes, scores, labels = run_dino_with_categories(pilot_np_images[path], federated_cats)
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box.tolist()
            clean_detections.append({
                "image_id": img_id,
                "category_id": cat_name_to_id[label],
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score),
            })
    print(f"Clean pass: {len(clean_detections)} detections, "
          f"{time.perf_counter()-t0:.1f}s")
    with open(CLEAN_BASELINE_PATH, "w") as f:
        json.dump(clean_detections, f)
 
# score against ground truth to get clean_mAP
from lvis import LVISResults, LVISEval
 
clean_det_path = "/kaggle/working/dets_clean.json"
with open(clean_det_path, "w") as f:
    json.dump(clean_detections, f)
 
lvis_results = LVISResults(lvis, clean_det_path)
lvis_eval = LVISEval(lvis, lvis_results, iou_type="bbox")
lvis_eval.params.img_ids = pilot_img_ids
lvis_eval.run()
clean_mAP = lvis_eval.results["AP"]
 
print(f"\n✅ Clean LVIS mAP (GroundingDINO, N=100, federated): {clean_mAP:.4f}")
print(f"   (compare against the COCO clean mAP of 0.5281 — a similar "
      f"ballpark is a reasonable sanity check, though LVIS's larger, "
      f"long-tailed category set makes some divergence expected)")

# ======================================================================
# CELL 11
# ======================================================================
# CELL 10d — YOLO-World inference, matching the actual Block 2/5 pattern
# -----------------------------------------------------------------------------
# The original loading code moves the model to CPU before set_classes,
# then back to GPU — a workaround for a CUDA text-encoding bug in
# ultralytics' YOLO-World, documented in Block 2. The COCO pipeline
# only pays this cost once (fixed 80-class list). LVIS federated mode
# needs a different category set per image, so this round-trip happens
# on every call here — expect this to run slower per-image than
# GroundingDINO's federated queries.
 
YOLO_CONF = 0.12

def run_yolo_with_categories(image_np, category_names, conf=YOLO_CONF):
    import cv2
    yolo_model.to('cpu')
    yolo_model.set_classes(category_names)
    yolo_model.to(device)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    results = yolo_model.predict(image_bgr, conf=conf, verbose=False)[0]

    boxes, scores, labels = [], [], []
    names = yolo_model.names
    for box in results.boxes:
        boxes.append(box.xyxy[0].tolist())
        scores.append(float(box.conf))
        labels.append(names[int(box.cls[0])])
    return boxes, scores, labels

print("✅ run_yolo_with_categories ready (yolov8x-worldv2, BGR, conf=0.12)")

# ======================================================================
# CELL 12
# ======================================================================
# CELL 10e — YOLO-World clean baseline
# -----------------------------------------------------------------------------
YOLO_CLEAN_BASELINE_PATH = "/kaggle/working/lvis_clean_baseline_yolo.json"
 
if os.path.exists(YOLO_CLEAN_BASELINE_PATH):
    with open(YOLO_CLEAN_BASELINE_PATH) as f:
        yolo_clean_detections = json.load(f)
    print(f"Loaded {len(yolo_clean_detections)} YOLO clean-baseline detections from checkpoint")
else:
    yolo_clean_detections = []
    t0 = time.perf_counter()
    for path in pilot_local_paths:
        img_id = img_path_to_id[path]
        federated_cats = get_federated_categories(img_id, lvis)
        if not federated_cats:
            continue
        boxes, scores, labels = run_yolo_with_categories(pilot_np_images[path], federated_cats)
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            cat_id = cat_name_to_id.get(label)
            if cat_id is None:  # YOLO's class name may not exactly match an LVIS name
                continue
            yolo_clean_detections.append({
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score),
            })
    print(f"YOLO clean pass: {len(yolo_clean_detections)} detections, "
          f"{time.perf_counter()-t0:.1f}s")
    with open(YOLO_CLEAN_BASELINE_PATH, "w") as f:
        json.dump(yolo_clean_detections, f)
 
yolo_clean_det_path = "/kaggle/working/dets_clean_yolo.json"
with open(yolo_clean_det_path, "w") as f:
    json.dump(yolo_clean_detections, f)
 
yolo_lvis_results = LVISResults(lvis, yolo_clean_det_path)
yolo_lvis_eval = LVISEval(lvis, yolo_lvis_results, iou_type="bbox")
yolo_lvis_eval.params.img_ids = pilot_img_ids
yolo_lvis_eval.run()
yolo_clean_mAP = yolo_lvis_eval.results["AP"]
 
print(f"\n✅ Clean LVIS mAP (YOLO-World, N=100, federated): {yolo_clean_mAP:.4f}")
print(f"   (compare against the COCO clean mAP of 0.4327)")

# ======================================================================
# CELL 13
# ======================================================================
# CELL 11 — run detection across all 75 configs, checkpointed
# -----------------------------------------------------------------------------
CORRUPTION_TYPES = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]
SEVERITIES = [1, 2, 3, 4, 5]
CHECKPOINT_PATH = "/kaggle/working/lvis_pilot_checkpoint.json"
 
def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return {}
 
def save_checkpoint(data):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(data, f)
 
pilot_results = load_checkpoint()
t_start = time.perf_counter()
configs_done = sum(len(v) for v in pilot_results.values())
 
for corruption in CORRUPTION_TYPES:
    if corruption not in pilot_results:
        pilot_results[corruption] = {}
    for severity in SEVERITIES:
        if str(severity) in pilot_results[corruption]:
            continue  # resuming from checkpoint
 
        config_detections = []
        for path in pilot_local_paths:
            img_id = img_path_to_id[path]
            corrupted = corrupt(pilot_np_images[path], corruption_name=corruption, severity=severity)
            federated_cats = get_federated_categories(img_id, lvis)
            if not federated_cats:
                continue
            boxes, scores, labels = run_dino_with_categories(corrupted, federated_cats)
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box.tolist()
                config_detections.append({
                    "image_id": img_id,
                    "category_id": cat_name_to_id[label],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score),
                })
 
        pilot_results[corruption][str(severity)] = config_detections
        save_checkpoint(pilot_results)
        configs_done += 1
        elapsed = (time.perf_counter() - t_start) / 60
        print(f"[{configs_done}/75] {corruption} sev {severity}: "
              f"{len(config_detections)} detections, {elapsed:.1f} min elapsed")
 
print(f"\n✅ Detection complete across all 75 configs, "
      f"{(time.perf_counter()-t_start)/60:.1f} min total")
 

# ======================================================================
# CELL 14
# ======================================================================
# CELL 11b — YOLO-World corrupted sweep, separate checkpoint from DINO's
# -----------------------------------------------------------------------------
# Re-applies corruption per config rather than caching corrupted images,
# same as Cell 11 — keeps memory flat, cost is regenerating corruption
# twice total (once for DINO, once here) rather than storing 7,500 images.
 
YOLO_CHECKPOINT_PATH = "/kaggle/working/lvis_pilot_checkpoint_yolo.json"
 
def load_yolo_checkpoint():
    if os.path.exists(YOLO_CHECKPOINT_PATH):
        with open(YOLO_CHECKPOINT_PATH) as f:
            return json.load(f)
    return {}
 
def save_yolo_checkpoint(data):
    with open(YOLO_CHECKPOINT_PATH, "w") as f:
        json.dump(data, f)
 
yolo_pilot_results = load_yolo_checkpoint()
t_start_yolo = time.perf_counter()
yolo_configs_done = sum(len(v) for v in yolo_pilot_results.values())
 
for corruption in CORRUPTION_TYPES:
    if corruption not in yolo_pilot_results:
        yolo_pilot_results[corruption] = {}
    for severity in SEVERITIES:
        if str(severity) in yolo_pilot_results[corruption]:
            continue  # resuming from checkpoint
 
        config_detections = []
        for path in pilot_local_paths:
            img_id = img_path_to_id[path]
            corrupted = corrupt(pilot_np_images[path], corruption_name=corruption, severity=severity)
            federated_cats = get_federated_categories(img_id, lvis)
            if not federated_cats:
                continue
            boxes, scores, labels = run_yolo_with_categories(corrupted, federated_cats)
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box
                cat_id = cat_name_to_id.get(label)
                if cat_id is None:
                    continue
                config_detections.append({
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score),
                })
 
        yolo_pilot_results[corruption][str(severity)] = config_detections
        save_yolo_checkpoint(yolo_pilot_results)
        yolo_configs_done += 1
        elapsed = (time.perf_counter() - t_start_yolo) / 60
        print(f"[{yolo_configs_done}/75] {corruption} sev {severity}: "
              f"{len(config_detections)} detections, {elapsed:.1f} min elapsed")
 
print(f"\n✅ YOLO-World detection complete across all 75 configs, "
      f"{(time.perf_counter()-t_start_yolo)/60:.1f} min total")
 

# ======================================================================
# CELL 15
# ======================================================================
# CELL 12 — score against LVIS ground truth
# -----------------------------------------------------------------------------
from lvis import LVISResults, LVISEval
 
pilot_ap_table = []
 
for corruption in CORRUPTION_TYPES:
    for severity in SEVERITIES:
        dets = pilot_results[corruption][str(severity)]
        if not dets:
            pilot_ap_table.append({"corruption": corruption, "severity": severity, "AP": 0.0})
            print(f"{corruption} sev {severity}: no detections, AP=0.0")
            continue
        det_path = f"/kaggle/working/dets_{corruption}_{severity}.json"
        with open(det_path, "w") as f:
            json.dump(dets, f)
        lvis_results = LVISResults(lvis, det_path)
        lvis_eval = LVISEval(lvis, lvis_results, iou_type="bbox")
        lvis_eval.params.img_ids = pilot_img_ids  # score only the pilot subset
        lvis_eval.run()
        ap = lvis_eval.results["AP"]
        pilot_ap_table.append({"corruption": corruption, "severity": severity, "AP": ap})
        print(f"{corruption} sev {severity}: AP={ap:.4f}")
 
with open("/kaggle/working/lvis_pilot_ap_table.json", "w") as f:
    json.dump(pilot_ap_table, f, indent=2)
 
print("\n✅ Saved raw AP table to /kaggle/working/lvis_pilot_ap_table.json")

# ======================================================================
# CELL 16
# ======================================================================
# CELL 12b — score YOLO-World against LVIS ground truth
# -----------------------------------------------------------------------------
yolo_pilot_ap_table = []
 
for corruption in CORRUPTION_TYPES:
    for severity in SEVERITIES:
        dets = yolo_pilot_results[corruption][str(severity)]
        if not dets:
            yolo_pilot_ap_table.append({"corruption": corruption, "severity": severity, "AP": 0.0})
            print(f"{corruption} sev {severity}: no detections, AP=0.0")
            continue
        det_path = f"/kaggle/working/dets_yolo_{corruption}_{severity}.json"
        with open(det_path, "w") as f:
            json.dump(dets, f)
        lvis_results = LVISResults(lvis, det_path)
        lvis_eval = LVISEval(lvis, lvis_results, iou_type="bbox")
        lvis_eval.params.img_ids = pilot_img_ids
        lvis_eval.run()
        ap = lvis_eval.results["AP"]
        yolo_pilot_ap_table.append({"corruption": corruption, "severity": severity, "AP": ap})
        print(f"{corruption} sev {severity}: AP={ap:.4f}")
 
with open("/kaggle/working/lvis_pilot_ap_table_yolo.json", "w") as f:
    json.dump(yolo_pilot_ap_table, f, indent=2)
 
print("\n✅ Saved YOLO AP table to /kaggle/working/lvis_pilot_ap_table_yolo.json")

# ======================================================================
# CELL 17
# ======================================================================
# CELL 13 — convert both models to CE / mCE, matching Table 4.2a/4.2b's
# exact formula and layout: CE_c = (1 - mean_mAP_c) / (1 - clean_mAP)
# -----------------------------------------------------------------------------
from collections import defaultdict
 
dino_ap_by_corruption = defaultdict(list)
for row in pilot_ap_table:
    dino_ap_by_corruption[row["corruption"]].append(row["AP"])
 
yolo_ap_by_corruption = defaultdict(list)
for row in yolo_pilot_ap_table:
    yolo_ap_by_corruption[row["corruption"]].append(row["AP"])
 
print(f"{'Corruption':<20} {'DINO mAP':>10} {'YOLO mAP':>10} {'DINO CE':>10} {'YOLO CE':>10}")
print("-" * 64)
 
dino_ce_values, yolo_ce_values = [], []
final_table = []
for corruption in CORRUPTION_TYPES:
    dino_mean_ap = np.mean(dino_ap_by_corruption[corruption])
    yolo_mean_ap = np.mean(yolo_ap_by_corruption[corruption])
    dino_ce = (1 - dino_mean_ap) / (1 - clean_mAP)
    yolo_ce = (1 - yolo_mean_ap) / (1 - yolo_clean_mAP)
    dino_ce_values.append(dino_ce)
    yolo_ce_values.append(yolo_ce)
    final_table.append({
        "corruption": corruption,
        "dino_mean_AP": dino_mean_ap, "yolo_mean_AP": yolo_mean_ap,
        "dino_CE": dino_ce, "yolo_CE": yolo_ce,
    })
    print(f"{corruption:<20} {dino_mean_ap:>10.4f} {yolo_mean_ap:>10.4f} "
          f"{dino_ce:>10.4f} {yolo_ce:>10.4f}")
 
dino_mCE = np.mean(dino_ce_values)
yolo_mCE = np.mean(yolo_ce_values)
print("-" * 64)
print(f"{'Overall mCE (N=100)':<20} {'':>10} {'':>10} {dino_mCE:>10.4f} {yolo_mCE:>10.4f}")
 
with open("/kaggle/working/lvis_pilot_CE_table_final.json", "w") as f:
    json.dump({
        "dino_clean_mAP": clean_mAP, "yolo_clean_mAP": yolo_clean_mAP,
        "per_corruption": final_table,
        "dino_overall_mCE": dino_mCE, "yolo_overall_mCE": yolo_mCE,
        "N": N_PILOT,
    }, f, indent=2)
 
print(f"\n✅ Saved final Table 4.2c to /kaggle/working/lvis_pilot_CE_table_final.json")
print(f"\nCompare against the COCO baseline:")
print(f"   {'':20} {'DINO':>10} {'YOLO':>10}")
print(f"   {'Table 4.2a (N=20)':20} {1.1583:>10.4f} {1.1593:>10.4f}")
print(f"   {'Table 4.2b (N=5000)':20} {1.3140:>10.4f} {1.3230:>10.4f}")
print(f"   {'Table 4.2c (LVIS, N=100)':20} {dino_mCE:>10.4f} {yolo_mCE:>10.4f}")
print(f"\n   Reminder: 4.2c uses federated (ground-truth-informed) category")
print(f"   querying for both models — not directly comparable to 4.2a/4.2b's")
print(f"   full-80-category querying, but DINO-vs-YOLO within 4.2c is a fair")
print(f"   comparison since both models were queried the same way.")

# ======================================================================
# CELL 18
# ======================================================================
import json
with open("/kaggle/working/lvis_pilot_ap_table.json") as f:
    pilot_ap_table = json.load(f)
with open("/kaggle/working/lvis_pilot_ap_table_yolo.json") as f:
    yolo_pilot_ap_table = json.load(f)

# ======================================================================
# CELL 19
# ======================================================================
# ============================================================
# FIGURE: LVIS Pilot — mAP Decay under ImageNet-C Corruptions
# Same style as Figure 4.1, built from live in-memory
# pilot_ap_table / yolo_pilot_ap_table (Cells 12/12b), not reloaded
# from disk.
# ============================================================

import matplotlib.pyplot as plt
from collections import defaultdict

# --- group per-severity AP by corruption, for both models ---
dino_by_corr = defaultdict(dict)
for row in pilot_ap_table:
    dino_by_corr[row["corruption"]][row["severity"]] = row["AP"]

yolo_by_corr = defaultdict(dict)
for row in yolo_pilot_ap_table:
    yolo_by_corr[row["corruption"]][row["severity"]] = row["AP"]

# same 15-type / 4-category grouping as the COCO Figure 4.1
LVIS_CORRUPTION_CATEGORIES = {
    "Noise":   ["gaussian_noise", "shot_noise", "impulse_noise"],
    "Blur":    ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"],
    "Weather": ["snow", "frost", "fog", "brightness"],
    "Digital": ["contrast", "elastic_transform", "pixelate", "jpeg_compression"],
}

SEVERITIES = [1, 2, 3, 4, 5]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(
    "LVIS Pilot Robustness — mAP Decay under ImageNet-C Corruptions\n"
    f"(N=100 images, federated querying, GroundingDINO clean={clean_mAP:.3f}, "
    f"YOLO-World clean={yolo_clean_mAP:.3f})",
    fontsize=13, fontweight='bold'
)
axes_flat = axes.flatten()

for idx, (cat_name, corruptions) in enumerate(LVIS_CORRUPTION_CATEGORIES.items()):
    ax = axes_flat[idx]

    for corruption in corruptions:
        label_name = corruption.replace("_", " ").title()
        dino_vals = [dino_by_corr[corruption][s] for s in SEVERITIES]
        yolo_vals = [yolo_by_corr[corruption][s] for s in SEVERITIES]

        ax.plot(SEVERITIES, dino_vals,
                marker='o', linewidth=1.5, alpha=0.8,
                label=f"DINO {label_name}")
        ax.plot(SEVERITIES, yolo_vals,
                marker='s', linewidth=1.5, alpha=0.8,
                linestyle='--',
                label=f"YOLO {label_name}")

    # clean baseline reference lines — same shared baseline used in Table 4.2c's CE
    ax.axhline(y=clean_mAP, color='green', linestyle=':', alpha=0.5, label='DINO clean')
    ax.axhline(y=yolo_clean_mAP, color='blue', linestyle=':', alpha=0.5, label='YOLO clean')

    ax.set_title(f"{cat_name} Corruptions", fontweight='bold')
    ax.set_xlabel("Severity Level")
    ax.set_ylabel("AP (LVIS, federated)")
    ax.set_ylim(0, max(clean_mAP, yolo_clean_mAP) + 0.1)
    ax.set_xticks(SEVERITIES)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)

plt.tight_layout()

fig_path_pdf = "/kaggle/working/figure_lvis_pilot_decay.pdf"
fig_path_png = "/kaggle/working/figure_lvis_pilot_decay.png"
plt.savefig(fig_path_pdf, format='pdf', dpi=300, bbox_inches='tight')
plt.savefig(fig_path_png, format='png', dpi=200, bbox_inches='tight')
plt.show()

print(f"✅ Figure saved: {fig_path_pdf}")
print(f"✅ Figure saved: {fig_path_png}")
