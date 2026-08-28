"""
TTRV GATE TROUBLESHOOTING AND VALIDATION

Diagnostic and validation session that preceded and justified the
standalone TTRV gate demo (see ../ttrv-gate-demo). Confirms the gate's
calibration (snr_baseline=32.842, tau_internal=0.9796, from
cratt-rq1-new.ipynb Block 24c) actually discriminates correct from
incorrect detections at the box level (IoU >= 0.5 against COCO ground
truth), not just the coarser category-presence check used earlier in
the pipeline.

Structure: each section below was originally a separate notebook cell,
run in order in a single continuous session. Two genuine bugs were
found and fixed along the way (documented inline, not edited out):
a zero-ground-truth image contaminating early precision estimates
(Cell 9/10), and an RNG isolation bug where per-image corruption
seeding was silently affecting which images got sampled for the next
corruption type (Cell 13/14).
"""

# ======================================================================
# CELL 0
# ======================================================================
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

!pip install imagecorruptions pycocotools -q

import random
import numpy as np
import torch
from torchvision.ops import box_iou
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from PIL import Image as PILImage
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Device: {device}")

DINO_MODEL_ID = "IDEA-Research/grounding-dino-base"
dino_processor = AutoProcessor.from_pretrained(DINO_MODEL_ID)
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL_ID).to(device)
dino_model.eval()
dino_params = sum(p.numel() for p in dino_model.parameters())
print(f"✅ GroundingDINO-base (Swin-B) loaded -- {dino_params:,} params")

# ======================================================================
# CELL 1
# ======================================================================
COCO_MAP = {
    'person': 1, 'bicycle': 2, 'car': 3, 'motorcycle': 4,
    'airplane': 5, 'bus': 6, 'train': 7, 'truck': 8, 'boat': 9,
    'traffic light': 10, 'fire hydrant': 11, 'stop sign': 13,
    'parking meter': 14, 'bench': 15, 'bird': 16, 'cat': 17,
    'dog': 18, 'horse': 19, 'sheep': 20, 'cow': 21,
    'elephant': 22, 'bear': 23, 'zebra': 24, 'giraffe': 25,
    'backpack': 27, 'umbrella': 28, 'handbag': 31, 'tie': 32,
    'suitcase': 33, 'frisbee': 34, 'skis': 35, 'snowboard': 36,
    'sports ball': 37, 'kite': 38, 'baseball bat': 39,
    'baseball glove': 40, 'skateboard': 41, 'surfboard': 42,
    'tennis racket': 43, 'bottle': 44, 'wine glass': 46,
    'cup': 47, 'fork': 48, 'knife': 49, 'spoon': 50, 'bowl': 51,
    'banana': 52, 'apple': 53, 'sandwich': 54, 'orange': 55,
    'broccoli': 56, 'carrot': 57, 'hot dog': 58, 'pizza': 59,
    'donut': 60, 'cake': 61, 'chair': 62, 'couch': 63,
    'potted plant': 64, 'bed': 65, 'dining table': 67,
    'toilet': 70, 'tv': 72, 'laptop': 73, 'mouse': 74,
    'remote': 75, 'keyboard': 76, 'cell phone': 77,
    'microwave': 78, 'oven': 79, 'toaster': 80, 'sink': 81,
    'refrigerator': 82, 'book': 84, 'clock': 85, 'vase': 86,
    'scissors': 87, 'teddy bear': 88, 'hair drier': 89,
    'toothbrush': 90
}
COCO_CLASSES = list(COCO_MAP.keys())
DINO_TEXT_PROMPT = " . ".join(COCO_CLASSES) + " ."
category_order = list(COCO_MAP.keys())

tokenizer = dino_processor.tokenizer
tok_with_offsets = tokenizer(
    DINO_TEXT_PROMPT, return_offsets_mapping=True,
    add_special_tokens=True, return_tensors=None,
)
offsets = tok_with_offsets["offset_mapping"]

def build_category_token_spans(text_prompt, category_names, offsets):
    spans = {}
    search_start = 0
    for cat in category_names:
        idx = text_prompt.find(cat, search_start)
        if idx == -1:
            idx = text_prompt.lower().find(cat.lower(), search_start)
        if idx == -1:
            spans[cat] = []
            continue
        start_char, end_char = idx, idx + len(cat)
        search_start = end_char
        token_ids = [i for i, (s, e) in enumerate(offsets)
                     if not (s == 0 and e == 0) and s < end_char and e > start_char]
        spans[cat] = token_ids
    return spans

category_token_spans = build_category_token_spans(DINO_TEXT_PROMPT, category_order, offsets)
n_mapped = sum(1 for c in category_order if category_token_spans[c])
print(f"✅ {len(COCO_MAP)} COCO categories, {n_mapped}/{len(category_order)} mapped")

# ======================================================================
# CELL 2
# ======================================================================
def get_category_distribution(outputs, category_token_spans, category_order):
    logits = outputs.logits[0]
    scores_sigmoid = logits.sigmoid()
    cat_score_cols = []
    for cat in category_order:
        tids = category_token_spans.get(cat, [])
        if len(tids) == 0:
            cat_score_cols.append(torch.zeros(scores_sigmoid.shape[0], device=scores_sigmoid.device))
        else:
            cat_score_cols.append(scores_sigmoid[:, tids].max(dim=-1).values)
    cat_scores_raw = torch.stack(cat_score_cols, dim=-1)
    return cat_scores_raw

BOX_THRESHOLD = 0.25

def b24_run_dino(image_np: np.ndarray):
    dino_model.eval()
    with torch.no_grad():
        inputs = dino_processor(images=image_np, text=DINO_TEXT_PROMPT, return_tensors="pt").to(device)
        outputs = dino_model(**inputs)
    cat_scores_raw = get_category_distribution(outputs, category_token_spans, category_order)
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
    boxes = pred_xyxy[keep_indices]
    scores = max_raw_scores[keep_indices]
    labels = [category_order[i] for i in max_cat_idx[keep_indices].tolist()]
    return boxes, scores, labels

print("✅ b24_run_dino ready -- clean argmax extraction, box_threshold=0.25")

# ======================================================================
# CELL 3
# ======================================================================
B24 = {
    "n_views": 5, "iou_thr": 0.40, "alpha": 0.40, "beta": 0.60,
    "snr_eps": 1e-4,
    "snr_baseline": 32.842,   # confirmed calibration behind reported N=2,400 result
    "tau_internal": 0.9796,
}
print(f"Using confirmed calibration: snr_baseline={B24['snr_baseline']}, tau_internal={B24['tau_internal']}")

def b24_augment(image_np: np.ndarray, seed: int) -> np.ndarray:
    random.seed(seed * 31 + 7)
    img = PILImage.fromarray(image_np.astype(np.uint8))
    brightness = 0.85 + random.random() * 0.30
    contrast   = 0.85 + random.random() * 0.30
    img = TF.adjust_brightness(img, brightness)
    img = TF.adjust_contrast(img, contrast)
    if seed % 3 == 0:
        img = TF.adjust_saturation(img, 0.80 + random.random() * 0.40)
    elif seed % 3 == 1:
        img = T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.4))(img)
    else:
        img = TF.adjust_hue(img, (-0.05 + random.random() * 0.10))
    return np.array(img)

def b24_compute_snr(image_np: np.ndarray) -> list:
    ref_boxes, ref_scores, ref_labels = b24_run_dino(image_np)
    if len(ref_boxes) == 0:
        return []
    view_outputs = []
    for v in range(1, B24["n_views"]):
        aug = b24_augment(image_np, seed=v)
        vb, vs, vl = b24_run_dino(aug)
        view_outputs.append((vb, vs, vl))
    detections = []
    for ref_box, ref_score, ref_label in zip(ref_boxes, ref_scores, ref_labels):
        if ref_label not in COCO_MAP:
            continue
        scores_seen = [ref_score.item()]
        match_count = 1
        rb_exp = ref_box.unsqueeze(0)
        for (vb, vs, _) in view_outputs:
            if len(vb) == 0:
                continue
            ious = box_iou(rb_exp, vb)
            max_iou, best_j = ious[0].max(0)
            if max_iou.item() >= B24["iou_thr"]:
                scores_seen.append(vs[best_j].item())
                match_count += 1
        arr = np.array(scores_seen)
        s_snr = float(arr.mean()) / (float(arr.std()) + B24["snr_eps"])
        iou_cons = match_count / B24["n_views"]
        detections.append({
            "box": ref_box, "label": ref_label,
            "s_snr": s_snr, "iou_consensus": iou_cons,
        })
    return detections

print("✅ b24_compute_snr ready")

# ======================================================================
# CELL 4
# ======================================================================
import glob
IMAGE_DIR = "/kaggle/input//datasets/awsaf49/coco-2017-dataset/coco2017/val2017"

all_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))
print(f"✅ Found {len(all_paths)} images in {IMAGE_DIR}")
if len(all_paths) == 0:
    print("⚠️  No images found. Check that awsaf49/coco-2017-dataset is added as an input, "
          "and confirm the exact path via: !ls /kaggle/input")

# ======================================================================
# CELL 5
# ======================================================================
# ============================================================
# DIAGNOSTIC: SNR saturation check
# Logs raw s_snr and the resulting snr_norm for every detection
# across a mixed sample, to determine whether the gate's SNR
# term is discriminating or saturating at the 2.0 cap.
# ============================================================
import os, random, json, time
import numpy as np
import pandas as pd
from imagecorruptions import corrupt as ic_corrupt

# --- Use the CONFIRMED calibration behind the reported N=2,400 result ---
B24["snr_baseline"] = 32.842
B24["tau_internal"] = 0.9796
print(f"Using confirmed calibration: snr_baseline={B24['snr_baseline']}, "
      f"tau_internal={B24['tau_internal']}")

# --- Sample images and mix corruptions/severities ---
N_IMAGES = 40
CORRUPTIONS = ["gaussian_noise", "motion_blur", "snow", "contrast"]
SEVERITIES = [1, 3, 5]

random.seed(123)
sample_paths = random.sample(all_paths, N_IMAGES)

records = []
start_time = time.time()
image_times = []

for i, path in enumerate(sample_paths):
    img_start = time.time()

    corruption = CORRUPTIONS[i % len(CORRUPTIONS)]
    severity = SEVERITIES[i % len(SEVERITIES)]

    img = np.array(__import__("PIL").Image.open(path).convert("RGB"))
    seed_val = (hash((os.path.basename(path), corruption, severity)) % (2**31))
    np.random.seed(seed_val)
    random.seed(seed_val)
    c_img = ic_corrupt(img, corruption_name=corruption, severity=severity)

    dets = b24_compute_snr(c_img)
    for d in dets:
        snr_norm = min(d["s_snr"] / max(B24["snr_baseline"], 1e-6), 2.0)
        records.append({
            "image": os.path.basename(path),
            "corruption": corruption,
            "severity": severity,
            "label": d["label"],
            "s_snr_raw": d["s_snr"],
            "snr_norm": snr_norm,
            "capped": snr_norm >= 1.999,
            "iou_consensus": d["iou_consensus"],
        })

    img_elapsed = time.time() - img_start
    image_times.append(img_elapsed)

    if (i + 1) == 1:
        print(f"  [1/{N_IMAGES}] first image took {img_elapsed:.1f}s "
              f"(includes CUDA warm-up, later images should be faster)")
    elif (i + 1) % 10 == 0:
        recent_avg = np.mean(image_times[-10:])
        elapsed_total = time.time() - start_time
        remaining = (N_IMAGES - (i + 1)) * recent_avg
        print(f"  [{i+1}/{N_IMAGES}] recent avg: {recent_avg:.1f}s/image | "
              f"elapsed: {elapsed_total/60:.1f} min | "
              f"est. remaining: {remaining/60:.1f} min")

total_elapsed = time.time() - start_time
print(f"\n✅ Done. Total time: {total_elapsed/60:.1f} minutes "
      f"({total_elapsed/N_IMAGES:.1f}s/image average)")

df = pd.DataFrame(records)
print(f"Total detections logged: {len(df)}")

# --- Key diagnostic numbers ---
pct_capped = df["capped"].mean() * 100
print(f"\n% of detections hitting the snr_norm cap (2.0): {pct_capped:.1f}%")
print(f"\nRaw s_snr distribution (percentiles):")
print(df["s_snr_raw"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]))

print(f"\nBreakdown by corruption type:")
print(df.groupby("corruption")["capped"].mean() * 100)

print(f"\nBreakdown by severity:")
print(df.groupby("severity")["capped"].mean() * 100)

# --- Save for the record ---
df.to_csv("/kaggle/working/snr_saturation_diagnostic.csv", index=False)
summary = {
    "n_images": N_IMAGES,
    "n_detections": len(df),
    "pct_capped": float(pct_capped),
    "snr_baseline_used": B24["snr_baseline"],
    "tau_internal_used": B24["tau_internal"],
    "total_runtime_minutes": total_elapsed / 60,
}
with open("/kaggle/working/snr_saturation_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n✅ Saved snr_saturation_diagnostic.csv and snr_saturation_summary.json")

# ======================================================================
# CELL 6
# ======================================================================
# ============================================================
# SPOT-CHECK: gate decisions against COCO ground truth
# Reads snr_saturation_diagnostic.csv, recomputes rjoint and
# the verify/reject decision for each detection, then checks
# whether the predicted category actually appears in COCO's
# ground truth for that image.
#
# NOTE ON WHAT THIS CHECKS: this is a category-presence check
# (does the predicted label exist anywhere in this image's GT
# annotations), not a full box-level IoU match. It's a coarser
# signal than mAP-style correctness, but it's a fast, honest
# first read: a detection whose predicted category isn't even
# present in the image is a clear miss regardless of box
# placement; a detection whose category IS present still needs
# the box to be roughly in the right place to be truly correct,
# which this check can't confirm on its own.
# ============================================================
import pandas as pd
import numpy as np
from pycocotools.coco import COCO

# --- Load ground truth annotations (fresh in this session) ---
ANN_PATH = ("/kaggle/input/datasets/awsaf49/"
            "coco-2017-dataset/coco2017/annotations/"
            "instances_val2017.json")
coco_gt = COCO(ANN_PATH)
official_cats = {cat['name']: cat['id'] for cat in coco_gt.loadCats(coco_gt.getCatIds())}

# --- Load the saved diagnostic data ---
df = pd.read_csv("/kaggle/working/snr_saturation_diagnostic.csv")

# --- Recompute the actual gate decision for each detection ---
ALPHA, BETA = 0.40, 0.60
TAU = 0.9796

df["rjoint"] = (df["iou_consensus"] ** ALPHA) * (df["snr_norm"] ** BETA)
df["verified"] = df["rjoint"] >= TAU
df["dist_from_tau"] = (df["rjoint"] - TAU).abs()

def category_in_gt(image_filename, label):
    """Check if `label` appears anywhere in this image's COCO ground truth."""
    img_id = int(image_filename.split(".")[0])
    cat_id = official_cats.get(label)
    if cat_id is None:
        return None  # label not a known COCO category (shouldn't happen here)
    ann_ids = coco_gt.getAnnIds(imgIds=img_id, catIds=[cat_id])
    return len(ann_ids) > 0

# --- Pick a representative, mixed sample ---
np.random.seed(7)
selections = []

# 3 clearly verified, high confidence
pool = df[df["verified"] & (df["rjoint"] > TAU + 0.02)]
if len(pool) >= 3:
    selections.append(pool.sample(min(3, len(pool)), random_state=1))

# 3 clearly flagged, low confidence
pool = df[~df["verified"] & (df["rjoint"] < TAU - 0.05)]
if len(pool) >= 3:
    selections.append(pool.sample(min(3, len(pool)), random_state=2))

# 2 near-threshold (within 0.01 of tau either side) -- the most informative cases
pool = df[df["dist_from_tau"] < 0.01]
if len(pool) >= 2:
    selections.append(pool.sample(min(2, len(pool)), random_state=3))

# 2 capped SNR cases specifically
pool = df[df["capped"]]
if len(pool) >= 2:
    selections.append(pool.sample(min(2, len(pool)), random_state=4))

sample = pd.concat(selections).drop_duplicates()

# --- Check each selected detection against ground truth ---
print(f"{'Image':<20} {'Label':<15} {'Corr/Sev':<15} {'rjoint':>7} {'Decision':<10} {'In GT?':<8}")
print("-" * 85)
for _, row in sample.iterrows():
    in_gt = category_in_gt(row["image"], row["label"])
    in_gt_str = "YES" if in_gt else ("NO" if in_gt is False else "??")
    decision = "VERIFIED" if row["verified"] else "FLAGGED"
    corr_sev = f"{row['corruption']}/{row['severity']}"
    print(f"{row['image']:<20} {row['label']:<15} {corr_sev:<15} "
          f"{row['rjoint']:>7.4f} {decision:<10} {in_gt_str:<8}")

# --- Summary: does verified correlate with being in GT here? ---
sample["in_gt"] = sample.apply(lambda r: category_in_gt(r["image"], r["label"]), axis=1)
print(f"\nVerified detections in GT:  {sample[sample['verified']]['in_gt'].mean()*100:.0f}% "
      f"(n={sample['verified'].sum()})")
print(f"Flagged detections in GT:   {sample[~sample['verified']]['in_gt'].mean()*100:.0f}% "
      f"(n={(~sample['verified']).sum()})")

# ======================================================================
# CELL 7
# ======================================================================
# ============================================================
# BOX-LEVEL CORRECTNESS CHECK
# Re-runs detection on the same 10 selected images/corruptions,
# recovers the actual predicted box, and checks IoU against the
# real COCO ground-truth box for that category. This is the
# same correctness standard (IoU >= 0.5) the dissertation's precision
# figures are built on, unlike the category-presence check.
# ============================================================
import os
import numpy as np
from PIL import Image as PILImage
from torchvision.ops import box_iou
import torch
from imagecorruptions import corrupt as ic_corrupt

IOU_THRESHOLD = 0.5

def get_gt_boxes_xyxy(image_filename, label):
    """Return all GT boxes (xyxy) for `label` in this image."""
    img_id = int(image_filename.split(".")[0])
    cat_id = official_cats.get(label)
    if cat_id is None:
        return []
    ann_ids = coco_gt.getAnnIds(imgIds=img_id, catIds=[cat_id])
    anns = coco_gt.loadAnns(ann_ids)
    boxes = []
    for a in anns:
        x, y, w, h = a["bbox"]  # COCO format: x, y, width, height
        boxes.append([x, y, x + w, y + h])
    return boxes

results = []
for _, row in sample.iterrows():
    fname = row["image"]
    label = row["label"]
    corruption = row["corruption"]
    severity = int(row["severity"])

    img_path = os.path.join(IMAGE_DIR, fname)
    img = np.array(PILImage.open(img_path).convert("RGB"))

    # Reproduce the exact same corruption used in the original diagnostic
    seed_val = (hash((fname, corruption, severity)) % (2**31))
    np.random.seed(seed_val)
    import random as _random
    _random.seed(seed_val)
    c_img = ic_corrupt(img, corruption_name=corruption, severity=severity)

    # Re-run detection and find the box matching this specific label
    boxes, scores, labels = b24_run_dino(c_img)
    matching_idx = [i for i, l in enumerate(labels) if l == label]

    if not matching_idx:
        results.append({**row.to_dict(), "best_iou": None, "correct": None,
                         "note": "label not re-detected on rerun"})
        continue

    # Take the highest-scoring box for this label (matches b24_compute_snr's
    # reference-view selection logic, which processes each detected box once)
    best_local_idx = matching_idx[int(scores[matching_idx].argmax())]
    pred_box = boxes[best_local_idx].unsqueeze(0)

    gt_boxes = get_gt_boxes_xyxy(fname, label)
    if not gt_boxes:
        results.append({**row.to_dict(), "best_iou": None, "correct": None,
                         "note": "no GT box for this category (shouldn't happen)"})
        continue

    gt_tensor = torch.tensor(gt_boxes, dtype=torch.float32).to(pred_box.device)  # <-- fixed
    ious = box_iou(pred_box, gt_tensor)
    best_iou = float(ious.max())
    correct = best_iou >= IOU_THRESHOLD

    results.append({**row.to_dict(), "best_iou": best_iou, "correct": correct, "note": ""})

results_df = pd.DataFrame(results)

print(f"{'Image':<20} {'Label':<15} {'Decision':<10} {'Best IoU':>9} {'Correct?':<9} {'Note'}")
print("-" * 90)
for _, r in results_df.iterrows():
    decision = "VERIFIED" if r["verified"] else "FLAGGED"
    iou_str = f"{r['best_iou']:.3f}" if r["best_iou"] is not None else "  n/a"
    correct_str = ("YES" if r["correct"] else "NO") if r["correct"] is not None else "??"
    print(f"{r['image']:<20} {r['label']:<15} {decision:<10} {iou_str:>9} {correct_str:<9} {r['note']}")

# --- The number that actually matters ---
valid = results_df.dropna(subset=["correct"])
print(f"\nVerified detections correct (IoU>=0.5): "
      f"{valid[valid['verified']]['correct'].mean()*100:.0f}% "
      f"(n={valid['verified'].sum()})")
print(f"Flagged detections correct (IoU>=0.5):  "
      f"{valid[~valid['verified']]['correct'].mean()*100:.0f}% "
      f"(n={(~valid['verified']).sum()})")

# ======================================================================
# CELL 8
# ======================================================================
# ============================================================
# SCALED, RANDOMIZED box-level correctness check
# Randomly samples detections across all four corruption types
# and multiple severities, computes IoU-based correctness against
# COCO ground truth, and runs the same statistical test the dissertation uses (Fisher's exact, odds ratio) on verified vs flagged.
#
# Also logs pairwise same-category box overlap, to check whether
# the duplicate/overlapping-box pattern seen in the demo is common.
# ============================================================
import os, random, time, json
import numpy as np
import pandas as pd
import torch
from PIL import Image as PILImage
from torchvision.ops import box_iou
from imagecorruptions import corrupt as ic_corrupt
from pycocotools.coco import COCO
from scipy.stats import fisher_exact

# --- Ground truth (idempotent if already loaded) ---
if "coco_gt" not in dir():
    ANN_PATH = ("/kaggle/input/datasets/awsaf49/"
                "coco-2017-dataset/coco2017/annotations/"
                "instances_val2017.json")
    coco_gt = COCO(ANN_PATH)
    official_cats = {cat['name']: cat['id'] for cat in coco_gt.loadCats(coco_gt.getCatIds())}
    print("✅ Ground truth loaded")

# --- Confirmed calibration behind the reported N=2,400 result ---
B24["snr_baseline"] = 32.842
B24["tau_internal"] = 0.9796
ALPHA, BETA = 0.40, 0.60
IOU_THRESHOLD = 0.5

# --- Random sample: proper mix, not cherry-picked ---
N_IMAGES = 160
CORRUPTIONS = ["gaussian_noise", "motion_blur", "snow", "contrast"]
SEVERITIES = [1, 2, 3, 4, 5]

random.seed(2026)
sample_paths = random.sample(all_paths, N_IMAGES)

def get_gt_boxes_xyxy(image_filename, label):
    img_id = int(image_filename.split(".")[0])
    cat_id = official_cats.get(label)
    if cat_id is None:
        return []
    ann_ids = coco_gt.getAnnIds(imgIds=img_id, catIds=[cat_id])
    anns = coco_gt.loadAnns(ann_ids)
    return [[a["bbox"][0], a["bbox"][1], a["bbox"][0]+a["bbox"][2], a["bbox"][1]+a["bbox"][3]]
            for a in anns]

records = []
dup_records = []
start_time = time.time()

for i, path in enumerate(sample_paths):
    fname = os.path.basename(path)
    corruption = CORRUPTIONS[i % len(CORRUPTIONS)]
    severity = SEVERITIES[i % len(SEVERITIES)]

    img = np.array(PILImage.open(path).convert("RGB"))
    seed_val = hash((fname, corruption, severity)) % (2**31)
    np.random.seed(seed_val)
    random.seed(seed_val)
    c_img = ic_corrupt(img, corruption_name=corruption, severity=severity)

    dets = b24_compute_snr(c_img)

    # --- duplicate/overlapping box check within this image ---
    by_label = {}
    for d in dets:
        by_label.setdefault(d["label"], []).append(d["box"])
    for label, boxes in by_label.items():
        if len(boxes) > 1:
            stacked = torch.stack(boxes)
            pair_ious = box_iou(stacked, stacked)
            n = len(boxes)
            for a in range(n):
                for b in range(a+1, n):
                    dup_records.append({
                        "image": fname, "label": label,
                        "pair_iou": float(pair_ious[a, b])
                    })

    # --- correctness check for each detection ---
    for d in dets:
        label = d["label"]
        snr_norm = min(d["s_snr"] / max(B24["snr_baseline"], 1e-6), 2.0)
        rjoint = (d["iou_consensus"] ** ALPHA) * (snr_norm ** BETA)
        verified = rjoint >= B24["tau_internal"]

        gt_boxes = get_gt_boxes_xyxy(fname, label)
        if not gt_boxes:
            correct = False  # predicted category not present in GT at all -> wrong
        else:
            gt_tensor = torch.tensor(gt_boxes, dtype=torch.float32).to(d["box"].device)
            ious = box_iou(d["box"].unsqueeze(0), gt_tensor)
            correct = float(ious.max()) >= IOU_THRESHOLD

        records.append({
            "image": fname, "corruption": corruption, "severity": severity,
            "label": label, "rjoint": rjoint, "verified": verified, "correct": correct,
        })

    if (i + 1) % 20 == 0:
        elapsed = time.time() - start_time
        rate = elapsed / (i + 1)
        remaining = (N_IMAGES - (i + 1)) * rate
        print(f"  [{i+1}/{N_IMAGES}] elapsed: {elapsed/60:.1f} min | "
              f"est. remaining: {remaining/60:.1f} min")

total_elapsed = time.time() - start_time
print(f"\n✅ Done in {total_elapsed/60:.1f} minutes")

df = pd.DataFrame(records)
dup_df = pd.DataFrame(dup_records)

df.to_csv("/kaggle/working/scaled_correctness_check.csv", index=False)
dup_df.to_csv("/kaggle/working/duplicate_box_check.csv", index=False)

# --- Contingency table + Fisher's exact test (mirrors paper's own method) ---
verified_correct = ((df["verified"]) & (df["correct"])).sum()
verified_incorrect = ((df["verified"]) & (~df["correct"])).sum()
rejected_correct = ((~df["verified"]) & (df["correct"])).sum()
rejected_incorrect = ((~df["verified"]) & (~df["correct"])).sum()

table = [[verified_correct, verified_incorrect],
         [rejected_correct, rejected_incorrect]]
odds_ratio, p_value = fisher_exact(table)

verified_precision = verified_correct / (verified_correct + verified_incorrect)
rejected_precision = rejected_correct / (rejected_correct + rejected_incorrect)

print(f"\n{'':20}{'Correct':>10}{'Incorrect':>12}{'Precision':>12}")
print(f"{'Verified':20}{verified_correct:>10}{verified_incorrect:>12}{verified_precision:>12.3f}")
print(f"{'Rejected':20}{rejected_correct:>10}{rejected_incorrect:>12}{rejected_precision:>12.3f}")
print(f"\nOdds ratio: {odds_ratio:.3f}")
print(f"Fisher's exact p-value: {p_value:.6f}")
print(f"Total detections: {len(df)}")

# --- Duplicate box summary ---
if len(dup_df) > 0:
    high_overlap = (dup_df["pair_iou"] > 0.5).mean() * 100
    print(f"\nSame-category box pairs found: {len(dup_df)}")
    print(f"% of those pairs with IoU > 0.5 (likely duplicates): {high_overlap:.1f}%")
else:
    print("\nNo same-category duplicate box pairs found in this sample.")

summary = {
    "n_images": N_IMAGES, "n_detections": len(df),
    "verified_precision": float(verified_precision),
    "rejected_precision": float(rejected_precision),
    "odds_ratio": float(odds_ratio), "p_value": float(p_value),
    "runtime_minutes": total_elapsed / 60,
    "calibration": {"snr_baseline": B24["snr_baseline"], "tau_internal": B24["tau_internal"]},
}
with open("/kaggle/working/scaled_correctness_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\n✅ Saved scaled_correctness_check.csv, duplicate_box_check.csv, scaled_correctness_summary.json")

# ======================================================================
# CELL 9
# ======================================================================
# Quick check: how many detections had zero GT boxes for their category?
zero_gt = df.apply(lambda r: len(get_gt_boxes_xyxy(r["image"], r["label"])) == 0, axis=1)
print(f"Detections with category absent from GT entirely: {zero_gt.sum()} / {len(df)} ({zero_gt.mean()*100:.1f}%)")

# Recompute the contingency table EXCLUDING those (matches a stricter methodology)
df_present = df[~zero_gt]
vc = ((df_present["verified"]) & (df_present["correct"])).sum()
vi = ((df_present["verified"]) & (~df_present["correct"])).sum()
rc = ((~df_present["verified"]) & (df_present["correct"])).sum()
ri = ((~df_present["verified"]) & (~df_present["correct"])).sum()
from scipy.stats import fisher_exact
odds_ratio2, p2 = fisher_exact([[vc, vi],[rc, ri]])
print(f"Verified precision: {vc/(vc+vi):.3f} | Rejected precision: {rc/(rc+ri):.3f}")
print(f"Odds ratio: {odds_ratio2:.3f} | p={p2:.6f}")

# ======================================================================
# CELL 10
# ======================================================================
# ============================================================
# SCALED, RANDOMIZED box-level correctness check (CORRECTED)
# Fixed to match the original protocol's image-selection rule:
# only images with >=1 COCO ground-truth annotation are eligible,
# matching how image_files was built in the original notebook.
# Everything else (random sampling across all 4 corruptions and
# 5 severities, real IoU-based correctness, Fisher's exact test,
# duplicate-box logging) is unchanged from the broader check.
# ============================================================
import os, random, time, json
import numpy as np
import pandas as pd
import torch
from PIL import Image as PILImage
from torchvision.ops import box_iou
from imagecorruptions import corrupt as ic_corrupt
from pycocotools.coco import COCO
from scipy.stats import fisher_exact

# --- Ground truth (idempotent if already loaded) ---
if "coco_gt" not in dir():
    ANN_PATH = ("/kaggle/input/datasets/awsaf49/"
                "coco-2017-dataset/coco2017/annotations/"
                "instances_val2017.json")
    coco_gt = COCO(ANN_PATH)
    official_cats = {cat['name']: cat['id'] for cat in coco_gt.loadCats(coco_gt.getCatIds())}
    print("✅ Ground truth loaded")

# --- Confirmed calibration behind the reported N=2,400 / N=2,624 results ---
B24["snr_baseline"] = 32.842
B24["tau_internal"] = 0.9796
ALPHA, BETA = 0.40, 0.60
IOU_THRESHOLD = 0.5

# --- Build a properly filtered random sample (fair test, no GT-less images) ---
N_IMAGES = 160
CORRUPTIONS = ["gaussian_noise", "motion_blur", "snow", "contrast"]
SEVERITIES = [1, 2, 3, 4, 5]

random.seed(2026)
candidates = random.sample(all_paths, min(len(all_paths), N_IMAGES * 3))  # oversample, then filter
sample_paths = []
for p in candidates:
    img_id = int(os.path.basename(p).split(".")[0])
    if len(coco_gt.getAnnIds(imgIds=[img_id])) > 0:
        sample_paths.append(p)
    if len(sample_paths) >= N_IMAGES:
        break

print(f"✅ Filtered sample: {len(sample_paths)} images, all with >=1 GT annotation")
if len(sample_paths) < N_IMAGES:
    print(f"⚠️  Only found {len(sample_paths)}/{N_IMAGES} — consider raising the oversample factor")

def get_gt_boxes_xyxy(image_filename, label):
    img_id = int(image_filename.split(".")[0])
    cat_id = official_cats.get(label)
    if cat_id is None:
        return []
    ann_ids = coco_gt.getAnnIds(imgIds=img_id, catIds=[cat_id])
    anns = coco_gt.loadAnns(ann_ids)
    return [[a["bbox"][0], a["bbox"][1], a["bbox"][0]+a["bbox"][2], a["bbox"][1]+a["bbox"][3]]
            for a in anns]

records = []
dup_records = []
start_time = time.time()

for i, path in enumerate(sample_paths):
    fname = os.path.basename(path)
    corruption = CORRUPTIONS[i % len(CORRUPTIONS)]
    severity = SEVERITIES[i % len(SEVERITIES)]

    img = np.array(PILImage.open(path).convert("RGB"))
    seed_val = hash((fname, corruption, severity)) % (2**31)
    np.random.seed(seed_val)
    random.seed(seed_val)
    c_img = ic_corrupt(img, corruption_name=corruption, severity=severity)

    dets = b24_compute_snr(c_img)

    # --- duplicate/overlapping box check within this image ---
    by_label = {}
    for d in dets:
        by_label.setdefault(d["label"], []).append(d["box"])
    for label, boxes in by_label.items():
        if len(boxes) > 1:
            stacked = torch.stack(boxes)
            pair_ious = box_iou(stacked, stacked)
            n = len(boxes)
            for a in range(n):
                for b in range(a+1, n):
                    dup_records.append({
                        "image": fname, "label": label,
                        "pair_iou": float(pair_ious[a, b])
                    })

    # --- correctness check for each detection ---
    for d in dets:
        label = d["label"]
        snr_norm = min(d["s_snr"] / max(B24["snr_baseline"], 1e-6), 2.0)
        rjoint = (d["iou_consensus"] ** ALPHA) * (snr_norm ** BETA)
        verified = rjoint >= B24["tau_internal"]

        gt_boxes = get_gt_boxes_xyxy(fname, label)
        if not gt_boxes:
            correct = False  # predicted category not present in GT at all -> wrong
        else:
            gt_tensor = torch.tensor(gt_boxes, dtype=torch.float32).to(d["box"].device)
            ious = box_iou(d["box"].unsqueeze(0), gt_tensor)
            correct = float(ious.max()) >= IOU_THRESHOLD

        records.append({
            "image": fname, "corruption": corruption, "severity": severity,
            "label": label, "rjoint": rjoint, "verified": verified, "correct": correct,
        })

    if (i + 1) % 20 == 0:
        elapsed = time.time() - start_time
        rate = elapsed / (i + 1)
        remaining = (len(sample_paths) - (i + 1)) * rate
        print(f"  [{i+1}/{len(sample_paths)}] elapsed: {elapsed/60:.1f} min | "
              f"est. remaining: {remaining/60:.1f} min")

total_elapsed = time.time() - start_time
print(f"\n✅ Done in {total_elapsed/60:.1f} minutes")

df = pd.DataFrame(records)
dup_df = pd.DataFrame(dup_records)

df.to_csv("/kaggle/working/scaled_correctness_check_v2.csv", index=False)
dup_df.to_csv("/kaggle/working/duplicate_box_check_v2.csv", index=False)

# --- Contingency table + Fisher's exact test (mirrors paper's own method) ---
verified_correct = ((df["verified"]) & (df["correct"])).sum()
verified_incorrect = ((df["verified"]) & (~df["correct"])).sum()
rejected_correct = ((~df["verified"]) & (df["correct"])).sum()
rejected_incorrect = ((~df["verified"]) & (~df["correct"])).sum()

table = [[verified_correct, verified_incorrect],
         [rejected_correct, rejected_incorrect]]
odds_ratio, p_value = fisher_exact(table)

verified_precision = verified_correct / (verified_correct + verified_incorrect)
rejected_precision = rejected_correct / (rejected_correct + rejected_incorrect)

print(f"\n{'':20}{'Correct':>10}{'Incorrect':>12}{'Precision':>12}")
print(f"{'Verified':20}{verified_correct:>10}{verified_incorrect:>12}{verified_precision:>12.3f}")
print(f"{'Rejected':20}{rejected_correct:>10}{rejected_incorrect:>12}{rejected_precision:>12.3f}")
print(f"\nOdds ratio: {odds_ratio:.3f}")
print(f"Fisher's exact p-value: {p_value:.6f}")
print(f"Total detections: {len(df)}")

# --- How many detections had zero-GT category (for comparison to before) ---
zero_gt = df.apply(lambda r: len(get_gt_boxes_xyxy(r["image"], r["label"])) == 0, axis=1)
print(f"\nDetections with category absent from GT entirely: {zero_gt.sum()} / {len(df)} "
      f"({zero_gt.mean()*100:.1f}%)  [expect this much lower than the 10.6% seen before]")

# --- Duplicate box summary ---
if len(dup_df) > 0:
    high_overlap = (dup_df["pair_iou"] > 0.5).mean() * 100
    print(f"\nSame-category box pairs found: {len(dup_df)}")
    print(f"% of those pairs with IoU > 0.5 (likely duplicates): {high_overlap:.1f}%")
else:
    print("\nNo same-category duplicate box pairs found in this sample.")

summary = {
    "n_images": len(sample_paths), "n_detections": len(df),
    "verified_precision": float(verified_precision),
    "rejected_precision": float(rejected_precision),
    "odds_ratio": float(odds_ratio), "p_value": float(p_value),
    "pct_zero_gt": float(zero_gt.mean() * 100),
    "runtime_minutes": total_elapsed / 60,
    "calibration": {"snr_baseline": B24["snr_baseline"], "tau_internal": B24["tau_internal"]},
}
with open("/kaggle/working/scaled_correctness_summary_v2.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\n✅ Saved scaled_correctness_check_v2.csv, duplicate_box_check_v2.csv, scaled_correctness_summary_v2.json")

# ======================================================================
# CELL 11
# ======================================================================
# ============================================================
# CONDITION-MATCHED replication: motion_blur + contrast,
# severity 5 only, matching the original test's exact scope.
# Fresh random sample, no other conditions mixed in.
# ============================================================
import os, random, time, json
import numpy as np
import pandas as pd
import torch
from PIL import Image as PILImage
from torchvision.ops import box_iou
from imagecorruptions import corrupt as ic_corrupt
from scipy.stats import fisher_exact

B24["snr_baseline"] = 32.842
B24["tau_internal"] = 0.9796
ALPHA, BETA = 0.40, 0.60
IOU_THRESHOLD = 0.5

N_IMAGES = 160
CORRUPTIONS = ["motion_blur", "contrast"]  # matches original exactly
SEVERITY = 5                                # matches original exactly

random.seed(99)
sample_paths = random.sample(all_paths, N_IMAGES)

def get_gt_boxes_xyxy(image_filename, label):
    img_id = int(image_filename.split(".")[0])
    cat_id = official_cats.get(label)
    if cat_id is None:
        return []
    ann_ids = coco_gt.getAnnIds(imgIds=img_id, catIds=[cat_id])
    anns = coco_gt.loadAnns(ann_ids)
    return [[a["bbox"][0], a["bbox"][1], a["bbox"][0]+a["bbox"][2], a["bbox"][1]+a["bbox"][3]]
            for a in anns]

records = []
start_time = time.time()

for i, path in enumerate(sample_paths):
    fname = os.path.basename(path)
    corruption = CORRUPTIONS[i % len(CORRUPTIONS)]  # alternate the two, matching original's 2-corruption scope

    img = np.array(PILImage.open(path).convert("RGB"))
    seed_val = hash((fname, corruption, SEVERITY)) % (2**31)
    np.random.seed(seed_val)
    random.seed(seed_val)
    c_img = ic_corrupt(img, corruption_name=corruption, severity=SEVERITY)

    dets = b24_compute_snr(c_img)
    for d in dets:
        label = d["label"]
        snr_norm = min(d["s_snr"] / max(B24["snr_baseline"], 1e-6), 2.0)
        rjoint = (d["iou_consensus"] ** ALPHA) * (snr_norm ** BETA)
        verified = rjoint >= B24["tau_internal"]

        gt_boxes = get_gt_boxes_xyxy(fname, label)
        if not gt_boxes:
            correct = False
        else:
            gt_tensor = torch.tensor(gt_boxes, dtype=torch.float32).to(d["box"].device)
            ious = box_iou(d["box"].unsqueeze(0), gt_tensor)
            correct = float(ious.max()) >= IOU_THRESHOLD

        records.append({"image": fname, "corruption": corruption, "label": label,
                         "rjoint": rjoint, "verified": verified, "correct": correct})

    if (i + 1) % 20 == 0:
        elapsed = time.time() - start_time
        remaining = (N_IMAGES - (i + 1)) * (elapsed / (i + 1))
        print(f"  [{i+1}/{N_IMAGES}] elapsed: {elapsed/60:.1f} min | est. remaining: {remaining/60:.1f} min")

total_elapsed = time.time() - start_time
print(f"\n✅ Done in {total_elapsed/60:.1f} minutes")

df = pd.DataFrame(records)
df.to_csv("/kaggle/working/matched_condition_check.csv", index=False)

vc = ((df["verified"]) & (df["correct"])).sum()
vi = ((df["verified"]) & (~df["correct"])).sum()
rc = ((~df["verified"]) & (df["correct"])).sum()
ri = ((~df["verified"]) & (~df["correct"])).sum()

odds_ratio, p_value = fisher_exact([[vc, vi], [rc, ri]])
verified_precision = vc / (vc + vi)
rejected_precision = rc / (rc + ri)

print(f"\n{'':20}{'Correct':>10}{'Incorrect':>12}{'Precision':>12}")
print(f"{'Verified':20}{vc:>10}{vi:>12}{verified_precision:>12.3f}")
print(f"{'Rejected':20}{rc:>10}{ri:>12}{rejected_precision:>12.3f}")
print(f"\nOdds ratio: {odds_ratio:.3f}")
print(f"Fisher's exact p-value: {p_value:.6f}")
print(f"Total detections: {len(df)}")
print(f"\nFor comparison, original reported: Verified 0.658, Rejected 0.474, OR=2.135")

# ======================================================================
# CELL 12
# ======================================================================
from pycocotools.coco import COCO

if "coco_gt" not in dir():
    ANN_PATH = ("/kaggle/input/datasets/awsaf49/"
                "coco-2017-dataset/coco2017/annotations/"
                "instances_val2017.json")
    coco_gt = COCO(ANN_PATH)
    print("✅ coco_gt loaded")

official_cats = {cat['name']: cat['id'] for cat in coco_gt.loadCats(coco_gt.getCatIds())}
print(f"✅ official_cats built: {len(official_cats)} categories")

# ======================================================================
# CELL 13
# ======================================================================
# ============================================================
# PER-CORRUPTION replication at severity 5, all four types
# Isolates whether gaussian_noise and snow (not yet independently
# checked) behave consistently with motion_blur/contrast.
# ============================================================
import os, random, time, json
import numpy as np
import pandas as pd
import torch
from PIL import Image as PILImage
from torchvision.ops import box_iou
from imagecorruptions import corrupt as ic_corrupt
from scipy.stats import fisher_exact

if "coco_gt" not in dir():
    ANN_PATH = ("/kaggle/input/datasets/awsaf49/"
                "coco-2017-dataset/coco2017/annotations/"
                "instances_val2017.json")
    coco_gt = COCO(ANN_PATH)
    print("✅ coco_gt loaded")

official_cats = {cat['name']: cat['id'] for cat in coco_gt.loadCats(coco_gt.getCatIds())}
print(f"✅ official_cats built: {len(official_cats)} categories")

B24["snr_baseline"] = 32.842
B24["tau_internal"] = 0.9796
ALPHA, BETA = 0.40, 0.60
IOU_THRESHOLD = 0.5
SEVERITY = 5
N_PER_CORRUPTION = 60   # 60 images x 4 corruptions = 240 images total

def get_gt_boxes_xyxy(image_filename, label):
    img_id = int(image_filename.split(".")[0])
    cat_id = official_cats.get(label)
    if cat_id is None:
        return []
    ann_ids = coco_gt.getAnnIds(imgIds=img_id, catIds=[cat_id])
    anns = coco_gt.loadAnns(ann_ids)
    return [[a["bbox"][0], a["bbox"][1], a["bbox"][0]+a["bbox"][2], a["bbox"][1]+a["bbox"][3]]
            for a in anns]

all_records = []
random.seed(555)

for corruption in ["gaussian_noise", "motion_blur", "snow", "contrast"]:
    sample_paths = random.sample(all_paths, N_PER_CORRUPTION)
    start = time.time()

    for path in sample_paths:
        fname = os.path.basename(path)
        img = np.array(PILImage.open(path).convert("RGB"))
        seed_val = hash((fname, corruption, SEVERITY)) % (2**31)
        np.random.seed(seed_val)
        random.seed(seed_val)
        c_img = ic_corrupt(img, corruption_name=corruption, severity=SEVERITY)

        dets = b24_compute_snr(c_img)
        for d in dets:
            label = d["label"]
            snr_norm = min(d["s_snr"] / max(B24["snr_baseline"], 1e-6), 2.0)
            rjoint = (d["iou_consensus"] ** ALPHA) * (snr_norm ** BETA)
            verified = rjoint >= B24["tau_internal"]

            gt_boxes = get_gt_boxes_xyxy(fname, label)
            if not gt_boxes:
                correct = False
            else:
                gt_tensor = torch.tensor(gt_boxes, dtype=torch.float32).to(d["box"].device)
                ious = box_iou(d["box"].unsqueeze(0), gt_tensor)
                correct = float(ious.max()) >= IOU_THRESHOLD

            all_records.append({"corruption": corruption, "verified": verified, "correct": correct})

    elapsed = time.time() - start
    print(f"✅ {corruption} done in {elapsed/60:.1f} min")

df = pd.DataFrame(all_records)
df.to_csv("/kaggle/working/per_corruption_check.csv", index=False)

print(f"\n{'Corruption':<16}{'Ver.Prec':>10}{'Rej.Prec':>10}{'Gap':>8}{'OddsRatio':>11}{'p-value':>10}{'N':>7}")
print("-" * 72)
for corruption in ["gaussian_noise", "motion_blur", "snow", "contrast"]:
    sub = df[df["corruption"] == corruption]
    vc = ((sub["verified"]) & (sub["correct"])).sum()
    vi = ((sub["verified"]) & (~sub["correct"])).sum()
    rc = ((~sub["verified"]) & (sub["correct"])).sum()
    ri = ((~sub["verified"]) & (~sub["correct"])).sum()
    if vc+vi == 0 or rc+ri == 0:
        print(f"{corruption:<16} insufficient data (verified or rejected group empty)")
        continue
    vp = vc / (vc + vi)
    rp = rc / (rc + ri)
    odds_ratio, p = fisher_exact([[vc, vi], [rc, ri]])
    print(f"{corruption:<16}{vp:>10.3f}{rp:>10.3f}{vp-rp:>8.3f}{odds_ratio:>11.3f}{p:>10.6f}{vc+vi+rc+ri:>7}")

print(f"\nTotal detections: {len(df)}")

# ======================================================================
# CELL 14
# ======================================================================
# ============================================================
# PER-CORRUPTION replication at severity 5, all four types
# FIXED: isolated RNG for image sampling, so per-image corruption
# seeding can no longer contaminate which images get selected
# for the next corruption type.
# ============================================================
import os, time, json
import numpy as np
import pandas as pd
import torch
import random as pyrandom
from PIL import Image as PILImage
from torchvision.ops import box_iou
from imagecorruptions import corrupt as ic_corrupt
from scipy.stats import fisher_exact
from pycocotools.coco import COCO

if "coco_gt" not in dir():
    ANN_PATH = ("/kaggle/input/datasets/awsaf49/"
                "coco-2017-dataset/coco2017/annotations/"
                "instances_val2017.json")
    coco_gt = COCO(ANN_PATH)
    print("✅ coco_gt loaded")

official_cats = {cat['name']: cat['id'] for cat in coco_gt.loadCats(coco_gt.getCatIds())}
print(f"✅ official_cats built: {len(official_cats)} categories")

B24["snr_baseline"] = 32.842
B24["tau_internal"] = 0.9796
ALPHA, BETA = 0.40, 0.60
IOU_THRESHOLD = 0.5
SEVERITY = 5
N_PER_CORRUPTION = 60

def get_gt_boxes_xyxy(image_filename, label):
    img_id = int(image_filename.split(".")[0])
    cat_id = official_cats.get(label)
    if cat_id is None:
        return []
    ann_ids = coco_gt.getAnnIds(imgIds=img_id, catIds=[cat_id])
    anns = coco_gt.loadAnns(ann_ids)
    return [[a["bbox"][0], a["bbox"][1], a["bbox"][0]+a["bbox"][2], a["bbox"][1]+a["bbox"][3]]
            for a in anns]

# --- ISOLATED sampler: never touched by corruption-seeding calls below ---
sampler_rng = pyrandom.Random(555)

all_records = []
for corruption in ["gaussian_noise", "motion_blur", "snow", "contrast"]:
    sample_paths = sampler_rng.sample(all_paths, N_PER_CORRUPTION)  # <-- uses isolated RNG
    start = time.time()

    for path in sample_paths:
        fname = os.path.basename(path)
        img = np.array(PILImage.open(path).convert("RGB"))
        seed_val = hash((fname, corruption, SEVERITY)) % (2**31)
        np.random.seed(seed_val)
        pyrandom.seed(seed_val)   # only affects the GLOBAL random module, not sampler_rng
        c_img = ic_corrupt(img, corruption_name=corruption, severity=SEVERITY)

        dets = b24_compute_snr(c_img)
        for d in dets:
            label = d["label"]
            snr_norm = min(d["s_snr"] / max(B24["snr_baseline"], 1e-6), 2.0)
            rjoint = (d["iou_consensus"] ** ALPHA) * (snr_norm ** BETA)
            verified = rjoint >= B24["tau_internal"]

            gt_boxes = get_gt_boxes_xyxy(fname, label)
            if not gt_boxes:
                correct = False
            else:
                gt_tensor = torch.tensor(gt_boxes, dtype=torch.float32).to(d["box"].device)
                ious = box_iou(d["box"].unsqueeze(0), gt_tensor)
                correct = float(ious.max()) >= IOU_THRESHOLD

            all_records.append({"corruption": corruption, "verified": verified, "correct": correct})

    elapsed = time.time() - start
    print(f"✅ {corruption} done in {elapsed/60:.1f} min")

df = pd.DataFrame(all_records)
df.to_csv("/kaggle/working/per_corruption_check_v2.csv", index=False)

print(f"\n{'Corruption':<16}{'Ver.Prec':>10}{'Rej.Prec':>10}{'Gap':>8}{'OddsRatio':>11}{'p-value':>10}{'N':>7}")
print("-" * 72)
for corruption in ["gaussian_noise", "motion_blur", "snow", "contrast"]:
    sub = df[df["corruption"] == corruption]
    vc = ((sub["verified"]) & (sub["correct"])).sum()
    vi = ((sub["verified"]) & (~sub["correct"])).sum()
    rc = ((~sub["verified"]) & (sub["correct"])).sum()
    ri = ((~sub["verified"]) & (~sub["correct"])).sum()
    if vc+vi == 0 or rc+ri == 0:
        print(f"{corruption:<16} insufficient data")
        continue
    vp = vc / (vc + vi)
    rp = rc / (rc + ri)
    odds_ratio, p = fisher_exact([[vc, vi], [rc, ri]])
    print(f"{corruption:<16}{vp:>10.3f}{rp:>10.3f}{vp-rp:>8.3f}{odds_ratio:>11.3f}{p:>10.6f}{vc+vi+rc+ri:>7}")

print(f"\nTotal detections: {len(df)}")

# ======================================================================
# CELL 15
# ======================================================================
# ============================================================
# FOCUSED replication: motion_blur only, severity 5, larger N
# Isolates whether the negative result above is real or noise
# ============================================================
import time
sampler_rng_mb = pyrandom.Random(777)  # different seed, independent draw
N_MOTION_BLUR = 250

sample_paths_mb = sampler_rng_mb.sample(all_paths, N_MOTION_BLUR)
mb_records = []
start = time.time()

for i, path in enumerate(sample_paths_mb):
    fname = os.path.basename(path)
    img = np.array(PILImage.open(path).convert("RGB"))
    seed_val = hash((fname, "motion_blur", 5)) % (2**31)
    np.random.seed(seed_val)
    pyrandom.seed(seed_val)
    c_img = ic_corrupt(img, corruption_name="motion_blur", severity=5)

    dets = b24_compute_snr(c_img)
    for d in dets:
        label = d["label"]
        snr_norm = min(d["s_snr"] / max(B24["snr_baseline"], 1e-6), 2.0)
        rjoint = (d["iou_consensus"] ** ALPHA) * (snr_norm ** BETA)
        verified = rjoint >= B24["tau_internal"]

        gt_boxes = get_gt_boxes_xyxy(fname, label)
        if not gt_boxes:
            correct = False
        else:
            gt_tensor = torch.tensor(gt_boxes, dtype=torch.float32).to(d["box"].device)
            ious = box_iou(d["box"].unsqueeze(0), gt_tensor)
            correct = float(ious.max()) >= IOU_THRESHOLD

        mb_records.append({"verified": verified, "correct": correct})

    if (i + 1) % 50 == 0:
        elapsed = time.time() - start
        remaining = (N_MOTION_BLUR - (i+1)) * (elapsed / (i+1))
        print(f"  [{i+1}/{N_MOTION_BLUR}] elapsed: {elapsed/60:.1f} min | est. remaining: {remaining/60:.1f} min")

mb_df = pd.DataFrame(mb_records)
mb_df.to_csv("/kaggle/working/motion_blur_focused_check.csv", index=False)

vc = ((mb_df["verified"]) & (mb_df["correct"])).sum()
vi = ((mb_df["verified"]) & (~mb_df["correct"])).sum()
rc = ((~mb_df["verified"]) & (mb_df["correct"])).sum()
ri = ((~mb_df["verified"]) & (~mb_df["correct"])).sum()
vp = vc / (vc + vi)
rp = rc / (rc + ri)
odds_ratio, p = fisher_exact([[vc, vi], [rc, ri]])

print(f"\nmotion_blur (severity 5), N={len(mb_df)}")
print(f"Verified precision: {vp:.3f} ({vc}/{vc+vi})")
print(f"Rejected precision: {rp:.3f} ({rc}/{rc+ri})")
print(f"Gap: {vp-rp:.3f} | Odds ratio: {odds_ratio:.3f} | p={p:.6f}")

# ======================================================================
# CELL 16
# ======================================================================
# ============================================================
# FOCUSED replication: gaussian_noise only, severity 5, larger N
# ============================================================
import time
sampler_rng_gn = pyrandom.Random(888)  # fresh, independent seed
N_GAUSSIAN = 1200

sample_paths_gn = sampler_rng_gn.sample(all_paths, N_GAUSSIAN)
gn_records = []
start = time.time()

for i, path in enumerate(sample_paths_gn):
    fname = os.path.basename(path)
    img = np.array(PILImage.open(path).convert("RGB"))
    seed_val = hash((fname, "gaussian_noise", 5)) % (2**31)
    np.random.seed(seed_val)
    pyrandom.seed(seed_val)
    c_img = ic_corrupt(img, corruption_name="gaussian_noise", severity=5)

    dets = b24_compute_snr(c_img)
    for d in dets:
        label = d["label"]
        snr_norm = min(d["s_snr"] / max(B24["snr_baseline"], 1e-6), 2.0)
        rjoint = (d["iou_consensus"] ** ALPHA) * (snr_norm ** BETA)
        verified = rjoint >= B24["tau_internal"]

        gt_boxes = get_gt_boxes_xyxy(fname, label)
        if not gt_boxes:
            correct = False
        else:
            gt_tensor = torch.tensor(gt_boxes, dtype=torch.float32).to(d["box"].device)
            ious = box_iou(d["box"].unsqueeze(0), gt_tensor)
            correct = float(ious.max()) >= IOU_THRESHOLD

        gn_records.append({"verified": verified, "correct": correct})

    if (i + 1) % 50 == 0:
        elapsed = time.time() - start
        remaining = (N_GAUSSIAN - (i+1)) * (elapsed / (i+1))
        print(f"  [{i+1}/{N_GAUSSIAN}] elapsed: {elapsed/60:.1f} min | est. remaining: {remaining/60:.1f} min")

gn_df = pd.DataFrame(gn_records)
gn_df.to_csv("/kaggle/working/gaussian_noise_focused_check.csv", index=False)

vc = ((gn_df["verified"]) & (gn_df["correct"])).sum()
vi = ((gn_df["verified"]) & (~gn_df["correct"])).sum()
rc = ((~gn_df["verified"]) & (gn_df["correct"])).sum()
ri = ((~gn_df["verified"]) & (~gn_df["correct"])).sum()
vp = vc / (vc + vi)
rp = rc / (rc + ri)
odds_ratio, p = fisher_exact([[vc, vi], [rc, ri]])

print(f"\ngaussian_noise (severity 5), N={len(gn_df)}")
print(f"Verified precision: {vp:.3f} ({vc}/{vc+vi})")
print(f"Rejected precision: {rp:.3f} ({rc}/{rc+ri})")
print(f"Gap: {vp-rp:.3f} | Odds ratio: {odds_ratio:.3f} | p={p:.6f}")

# ======================================================================
# CELL 17
# ======================================================================
# ============================================================
# SEVERITY 1 replication, all four corruption types
# Checkpointed per corruption: if the kernel dies partway,
# completed corruptions are saved and won't need rerunning.
# ============================================================
import os, time, json
import pandas as pd
import numpy as np
import torch
import random as pyrandom
from PIL import Image as PILImage
from torchvision.ops import box_iou
from imagecorruptions import corrupt as ic_corrupt
from scipy.stats import fisher_exact

SEVERITY = 1
N_PER_CORRUPTION = 250
CHECKPOINT_DIR = "/kaggle/working/sev1_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def checkpoint_path(corruption):
    return os.path.join(CHECKPOINT_DIR, f"sev{SEVERITY}_{corruption}.csv")

all_records = []
sampler_rng = pyrandom.Random(2024)  # fixed seed, isolated from corruption seeding

for corruption in ["gaussian_noise", "motion_blur", "snow", "contrast"]:
    ckpt_path = checkpoint_path(corruption)
    if os.path.exists(ckpt_path):
        df_ckpt = pd.read_csv(ckpt_path)
        all_records.extend(df_ckpt.to_dict("records"))
        print(f"⏭️  {corruption}: loaded {len(df_ckpt)} rows from checkpoint")
        continue

    sample_paths = sampler_rng.sample(all_paths, N_PER_CORRUPTION)
    corruption_records = []
    start = time.time()

    for i, path in enumerate(sample_paths):
        fname = os.path.basename(path)
        img = np.array(PILImage.open(path).convert("RGB"))
        seed_val = hash((fname, corruption, SEVERITY)) % (2**31)
        np.random.seed(seed_val)
        pyrandom.seed(seed_val)
        c_img = ic_corrupt(img, corruption_name=corruption, severity=SEVERITY)

        dets = b24_compute_snr(c_img)
        for d in dets:
            label = d["label"]
            snr_norm = min(d["s_snr"] / max(B24["snr_baseline"], 1e-6), 2.0)
            rjoint = (d["iou_consensus"] ** ALPHA) * (snr_norm ** BETA)
            verified = rjoint >= B24["tau_internal"]

            gt_boxes = get_gt_boxes_xyxy(fname, label)
            if not gt_boxes:
                correct = False
            else:
                gt_tensor = torch.tensor(gt_boxes, dtype=torch.float32).to(d["box"].device)
                ious = box_iou(d["box"].unsqueeze(0), gt_tensor)
                correct = float(ious.max()) >= IOU_THRESHOLD

            corruption_records.append({"corruption": corruption, "verified": verified, "correct": correct})

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            remaining = (N_PER_CORRUPTION - (i+1)) * (elapsed / (i+1))
            print(f"  [{corruption} {i+1}/{N_PER_CORRUPTION}] elapsed: {elapsed/60:.1f} min | "
                  f"est. remaining: {remaining/60:.1f} min")

    pd.DataFrame(corruption_records).to_csv(ckpt_path, index=False)
    all_records.extend(corruption_records)
    print(f"✅ {corruption} done in {(time.time()-start)/60:.1f} min, checkpoint saved")

df = pd.DataFrame(all_records)
df.to_csv("/kaggle/working/severity1_full_check.csv", index=False)

print(f"\n{'Corruption':<16}{'Ver.Prec':>10}{'Rej.Prec':>10}{'Gap':>8}{'OddsRatio':>11}{'p-value':>10}{'N':>7}")
print("-" * 72)
for corruption in ["gaussian_noise", "motion_blur", "snow", "contrast"]:
    sub = df[df["corruption"] == corruption]
    vc = ((sub["verified"]) & (sub["correct"])).sum()
    vi = ((sub["verified"]) & (~sub["correct"])).sum()
    rc = ((~sub["verified"]) & (sub["correct"])).sum()
    ri = ((~sub["verified"]) & (~sub["correct"])).sum()
    vp = vc / (vc + vi)
    rp = rc / (rc + ri)
    odds_ratio, p = fisher_exact([[vc, vi], [rc, ri]])
    print(f"{corruption:<16}{vp:>10.3f}{rp:>10.3f}{vp-rp:>8.3f}{odds_ratio:>11.3f}{p:>10.6f}{vc+vi+rc+ri:>7}")

print(f"\nTotal detections: {len(df)}")

# ======================================================================
# CELL 18
# ======================================================================
# ============================================================
# SEVERITY 3 replication, all four corruption types
# Checkpointed per corruption, same pattern as severity 1.
# ============================================================
import os, time, json
import pandas as pd
import numpy as np
import torch
import random as pyrandom
from PIL import Image as PILImage
from torchvision.ops import box_iou
from imagecorruptions import corrupt as ic_corrupt
from scipy.stats import fisher_exact

SEVERITY = 3
N_PER_CORRUPTION = 250
CHECKPOINT_DIR = "/kaggle/working/sev3_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def checkpoint_path(corruption):
    return os.path.join(CHECKPOINT_DIR, f"sev{SEVERITY}_{corruption}.csv")

all_records = []
sampler_rng = pyrandom.Random(3033)  # fresh seed for severity 3, isolated from corruption seeding

for corruption in ["gaussian_noise", "motion_blur", "snow", "contrast"]:
    ckpt_path = checkpoint_path(corruption)
    if os.path.exists(ckpt_path):
        df_ckpt = pd.read_csv(ckpt_path)
        all_records.extend(df_ckpt.to_dict("records"))
        print(f"⏭️  {corruption}: loaded {len(df_ckpt)} rows from checkpoint")
        continue

    sample_paths = sampler_rng.sample(all_paths, N_PER_CORRUPTION)
    corruption_records = []
    start = time.time()

    for i, path in enumerate(sample_paths):
        fname = os.path.basename(path)
        img = np.array(PILImage.open(path).convert("RGB"))
        seed_val = hash((fname, corruption, SEVERITY)) % (2**31)
        np.random.seed(seed_val)
        pyrandom.seed(seed_val)
        c_img = ic_corrupt(img, corruption_name=corruption, severity=SEVERITY)

        dets = b24_compute_snr(c_img)
        for d in dets:
            label = d["label"]
            snr_norm = min(d["s_snr"] / max(B24["snr_baseline"], 1e-6), 2.0)
            rjoint = (d["iou_consensus"] ** ALPHA) * (snr_norm ** BETA)
            verified = rjoint >= B24["tau_internal"]

            gt_boxes = get_gt_boxes_xyxy(fname, label)
            if not gt_boxes:
                correct = False
            else:
                gt_tensor = torch.tensor(gt_boxes, dtype=torch.float32).to(d["box"].device)
                ious = box_iou(d["box"].unsqueeze(0), gt_tensor)
                correct = float(ious.max()) >= IOU_THRESHOLD

            corruption_records.append({"corruption": corruption, "verified": verified, "correct": correct})

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            remaining = (N_PER_CORRUPTION - (i+1)) * (elapsed / (i+1))
            print(f"  [{corruption} {i+1}/{N_PER_CORRUPTION}] elapsed: {elapsed/60:.1f} min | "
                  f"est. remaining: {remaining/60:.1f} min")

    pd.DataFrame(corruption_records).to_csv(ckpt_path, index=False)
    all_records.extend(corruption_records)
    print(f"✅ {corruption} done in {(time.time()-start)/60:.1f} min, checkpoint saved")

df = pd.DataFrame(all_records)
df.to_csv("/kaggle/working/severity3_full_check.csv", index=False)

print(f"\n{'Corruption':<16}{'Ver.Prec':>10}{'Rej.Prec':>10}{'Gap':>8}{'OddsRatio':>11}{'p-value':>10}{'N':>7}")
print("-" * 72)
for corruption in ["gaussian_noise", "motion_blur", "snow", "contrast"]:
    sub = df[df["corruption"] == corruption]
    vc = ((sub["verified"]) & (sub["correct"])).sum()
    vi = ((sub["verified"]) & (~sub["correct"])).sum()
    rc = ((~sub["verified"]) & (sub["correct"])).sum()
    ri = ((~sub["verified"]) & (~sub["correct"])).sum()
    vp = vc / (vc + vi)
    rp = rc / (rc + ri)
    odds_ratio, p = fisher_exact([[vc, vi], [rc, ri]])
    print(f"{corruption:<16}{vp:>10.3f}{rp:>10.3f}{vp-rp:>8.3f}{odds_ratio:>11.3f}{p:>10.6f}{vc+vi+rc+ri:>7}")

print(f"\nTotal detections: {len(df)}")
