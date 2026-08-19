# ============================================================
# BLOCK 25: Training-Free Confidence Recalibration via Online
# Memory of Attribute-Level Vulnerability (Table 4.12)
# GroundingDINO-base (Swin-B backbone)
# ============================================================
#
# WHY THIS VERSION IS STREAMLINED
# --------------------------------
# The original investigation tried several signals before landing
# on this one: score-SNR recalibration (near-zero discrimination),
# softmax-based KL divergence (confounded with raw confidence),
# and per-image relative gating (CSS) on top of it. None of these
# are reproduced here. This block goes straight to the method that
# actually worked: raw-score Bernoulli KL divergence across 6
# strong single-attribute perturbations, calibrated per
# (corruption, category, attribute) cell, tracked online via
# Welford running statistics, and validated on a genuinely
# held-out image set.
#
# NO WEIGHT UPDATES ANYWHERE IN THIS BLOCK. The model is frozen
# throughout — recalibration only rescales detection confidence
# scores at inference time.
# ============================================================

import torch
import numpy as np
import pandas as pd
import os
import random as _py_random
from PIL import Image as PILImage
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from imagecorruptions import corrupt as ic_corrupt
from torchvision.ops import box_iou
from tqdm.notebook import tqdm

print("=" * 65)
print("BLOCK 25: Online Memory Recalibration, Swin-B")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────
# Pre-flight check
# ─────────────────────────────────────────────────────────────
required_p5 = ["dino_model", "dino_processor", "DINO_TEXT_PROMPT", "device",
               "compute_map", "coco_gt", "COCO_MAP",
               "image_files", "loaded_images", "img_id_map"]

missing_p5 = [f for f in required_p5 if f not in globals()]
if missing_p5:
    print(f"❌ Missing before Block 25: {missing_p5}")
else:
    print("✅ All base dependencies present")

print(f"Current image pool size: {len(image_files)}  (need 80: 10 test + 50 calib + 20 held-out)")
print()

# ─────────────────────────────────────────────────────────────
# Expand image pool to 100 (buffer above this block's own 80
# requirement — Section E below will find this already satisfied)
# ─────────────────────────────────────────────────────────────
TARGET_N = 100

if len(image_files) < TARGET_N:
    IMAGE_DIR_CURRENT = os.path.dirname(image_files[0])
    all_candidates = sorted(f for f in os.listdir(IMAGE_DIR_CURRENT) if f.lower().endswith(".jpg"))
    existing_basenames = {os.path.basename(p) for p in image_files}
    added = 0
    for fname in all_candidates:
        if len(image_files) >= TARGET_N:
            break
        if fname in existing_basenames:
            continue
        full_path = os.path.join(IMAGE_DIR_CURRENT, fname)
        try:
            candidate_img_id = int(os.path.splitext(fname)[0])
        except ValueError:
            continue
        if len(coco_gt.getAnnIds(imgIds=[candidate_img_id])) == 0:
            continue
        img_arr = np.array(PILImage.open(full_path).convert("RGB"))
        loaded_images[full_path] = img_arr
        img_id_map[fname] = candidate_img_id
        image_files.append(full_path)
        added += 1
    print(f"✅ Added {added} images -- pool now at {len(image_files)}")
else:
    print(f"✅ Pool already at {len(image_files)}, no expansion needed")
print()

# ─────────────────────────────────────────────────────────────
# SECTION A — Safety reset: fully frozen model, no LoRA delta
# ─────────────────────────────────────────────────────────────
if "reset_lora_b" in globals():
    reset_lora_b()
for p in dino_model.parameters():
    p.requires_grad = False
dino_model.eval()
n_trainable = sum(p.numel() for p in dino_model.parameters() if p.requires_grad)
print(f"✅ Model frozen -- trainable params: {n_trainable}  (should be 0)")
print()

# ─────────────────────────────────────────────────────────────
# SECTION B — Corruption seeding + strong single-attribute views
# ─────────────────────────────────────────────────────────────
CORRUPTION_SEED_OFFSET = {
    "gaussian_noise": 1_000, "motion_blur": 2_000,
    "snow": 3_000, "contrast": 4_000,
}

def seed_for_corruption(img_id, corruption, severity):
    offset = CORRUPTION_SEED_OFFSET.get(corruption, 9_000)
    return (img_id * 100 + offset + severity) % (2**31)

def apply_corruption_deterministic(raw_img, img_id, corruption, severity):
    seed_val = seed_for_corruption(img_id, corruption, severity)
    np.random.seed(seed_val)
    _py_random.seed(seed_val)
    return ic_corrupt(raw_img, corruption_name=corruption, severity=severity)

def strong_view(image_np: np.ndarray, view_idx: int) -> np.ndarray:
    img = PILImage.fromarray(image_np.astype(np.uint8))
    if view_idx == 1:
        img = TF.adjust_brightness(img, 0.45)
    elif view_idx == 2:
        img = TF.adjust_contrast(img, 0.45)
    elif view_idx == 3:
        img = T.GaussianBlur(kernel_size=9, sigma=2.5)(img)
    elif view_idx == 4:
        arr = np.array(img).astype(np.float32)
        noise = np.random.RandomState(42).normal(0, 35, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = PILImage.fromarray(arr)
    elif view_idx == 5:
        img = TF.adjust_saturation(img, 0.25)
    elif view_idx == 6:
        img = TF.adjust_hue(img, 0.15)
    return np.array(img)

VIEW_NAMES = {1: "brightness", 2: "contrast", 3: "blur",
              4: "noise", 5: "saturation", 6: "hue"}
N_STRONG_VIEWS = 6
DIAG_CORRUPTIONS = ["motion_blur", "contrast"]
DIAG_SEVERITY    = 5
DIAG_N           = 10

print("✅ Section B: corruption seeding + 6 strong perturbation views defined")
print()

# ─────────────────────────────────────────────────────────────
# SECTION C — Category token-span mapping (needed once, then
# reused by every downstream call to run_dino_with_categories_raw)
# ─────────────────────────────────────────────────────────────
category_order = list(COCO_MAP.keys())
tokenizer = dino_processor.tokenizer
tok_with_offsets = tokenizer(
    DINO_TEXT_PROMPT, return_offsets_mapping=True,
    add_special_tokens=True, return_tensors=None,
)
offsets       = tok_with_offsets["offset_mapping"]
token_ids_ref = tok_with_offsets["input_ids"]

_sanity_inputs = dino_processor(
    images=loaded_images[image_files[0]], text=DINO_TEXT_PROMPT, return_tensors="pt"
).to(device)
actual_input_ids = _sanity_inputs.input_ids[0].cpu().tolist()
if token_ids_ref != actual_input_ids:
    raise RuntimeError(
        "Tokenization mismatch between offset mapping and model inputs -- "
        "category span mapping would be WRONG. Stop and investigate."
    )

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
print(f"✅ Section C: category token spans built -- {n_mapped}/{len(category_order)} categories mapped")
print()

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
    cat_dist = torch.softmax(cat_scores_raw, dim=-1)
    return cat_scores_raw, cat_dist

# ─────────────────────────────────────────────────────────────
# SECTION D — Corrected extraction (box-confidence gate = 0.25)
# and raw-score Bernoulli KL
# ─────────────────────────────────────────────────────────────
BOX_THRESHOLD = 0.25

def run_dino_with_categories_raw(image_np, box_threshold=BOX_THRESHOLD):
    dino_model.eval()
    with torch.no_grad():
        inputs = dino_processor(images=image_np, text=DINO_TEXT_PROMPT, return_tensors="pt").to(device)
        outputs = dino_model(**inputs)

    cat_scores_raw, cat_dist = get_category_distribution(outputs, category_token_spans, category_order)
    max_raw_scores, max_cat_idx = cat_scores_raw.max(dim=-1)
    keep_mask = max_raw_scores >= box_threshold
    keep_indices = keep_mask.nonzero(as_tuple=True)[0]

    if len(keep_indices) == 0:
        return [], [], [], [], None, None

    img_h, img_w = image_np.shape[:2]
    pred_cxcywh = outputs.pred_boxes[0]
    cx = pred_cxcywh[:, 0] * img_w
    cy = pred_cxcywh[:, 1] * img_h
    pw = pred_cxcywh[:, 2] * img_w
    ph = pred_cxcywh[:, 3] * img_h
    pred_xyxy = torch.stack([cx - pw/2, cy - ph/2, cx + pw/2, cy + ph/2], dim=-1)

    boxes     = pred_xyxy[keep_indices]
    scores    = max_raw_scores[keep_indices]
    labels    = [category_order[i] for i in max_cat_idx[keep_indices].tolist()]
    query_idx = keep_indices.tolist()
    full_dist = cat_dist[keep_indices]
    raw_dist  = cat_scores_raw[keep_indices]

    return boxes, scores, labels, query_idx, raw_dist, full_dist

def bernoulli_kl(p, q, eps=1e-6):
    p = p.clamp(eps, 1 - eps)
    q = q.clamp(eps, 1 - eps)
    kl = p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))
    return kl.sum().item()

print(f"✅ Section D: run_dino_with_categories_raw (box_threshold={BOX_THRESHOLD}) + bernoulli_kl defined")
print()

def per_attribute_kl_for_detection(c_img):
    boxes_ref, scores_ref, labels_ref, qidx_ref, raw_ref, dist_ref = run_dino_with_categories_raw(c_img)
    if len(boxes_ref) == 0:
        return []
    view_kls = {}
    for v in range(1, N_STRONG_VIEWS + 1):
        aug = strong_view(c_img, v)
        boxes_v, scores_v, labels_v, qidx_v, raw_v, dist_v = run_dino_with_categories_raw(aug)
        attr_name = VIEW_NAMES[v]
        if len(boxes_v) == 0:
            continue
        for i in range(len(boxes_ref)):
            ref_box = boxes_ref[i].unsqueeze(0)
            ious = box_iou(ref_box, boxes_v)
            max_iou, best_j = ious[0].max(0)
            if max_iou.item() >= 0.40:
                kl = bernoulli_kl(raw_ref[i], raw_v[best_j.item()])
                view_kls.setdefault(i, {})[attr_name] = kl
    records = []
    for i in range(len(boxes_ref)):
        records.append({
            "category": labels_ref[i],
            "score": scores_ref[i].item(),
            "kl_per_attr": view_kls.get(i, {}),
        })
    return records

# ─────────────────────────────────────────────────────────────
# SECTION E — Expand image pool to 80 (10 test + 50 calib + 20
# held-out). Already satisfied by the pool expansion to 100
# above, this confirms rather than adds.
# ─────────────────────────────────────────────────────────────
TARGET_N = 80
if len(image_files) < TARGET_N:
    IMAGE_DIR_CURRENT = os.path.dirname(image_files[0])
    all_candidates = sorted(f for f in os.listdir(IMAGE_DIR_CURRENT) if f.lower().endswith(".jpg"))
    existing_basenames = {os.path.basename(p) for p in image_files}
    added = 0
    for fname in all_candidates:
        if len(image_files) >= TARGET_N:
            break
        if fname in existing_basenames:
            continue
        full_path = os.path.join(IMAGE_DIR_CURRENT, fname)
        try:
            candidate_img_id = int(os.path.splitext(fname)[0])
        except ValueError:
            continue
        if len(coco_gt.getAnnIds(imgIds=[candidate_img_id])) == 0:
            continue
        img_arr = np.array(PILImage.open(full_path).convert("RGB"))
        loaded_images[full_path] = img_arr
        img_id_map[fname] = candidate_img_id
        image_files.append(full_path)
        added += 1
    print(f"✅ Section E: added {added} images -- pool now at {len(image_files)}")
else:
    print(f"✅ Section E: pool already at {len(image_files)}, no expansion needed")
print()

# ─────────────────────────────────────────────────────────────
# SECTION F — Calibration at N=50 (image_files[10:60]), once
# ─────────────────────────────────────────────────────────────
CALIB_START, CALIB_N, MIN_SAMPLES = 10, 50, 2
calib_records = []

for corruption in DIAG_CORRUPTIONS:
    for img_path in tqdm(image_files[CALIB_START:CALIB_START+CALIB_N],
                          desc=f"calibrating on {corruption}"):
        img_id  = img_id_map[os.path.basename(img_path)]
        raw_img = loaded_images[img_path]
        c_img   = apply_corruption_deterministic(raw_img, img_id, corruption, DIAG_SEVERITY)
        for r in per_attribute_kl_for_detection(c_img):
            for attr, kl in r["kl_per_attr"].items():
                calib_records.append({"corruption": corruption, "category": r["category"],
                                       "attribute": attr, "kl": kl})

df_calib = pd.DataFrame(calib_records)
print(f"\n✅ Section F: calibration complete -- {len(df_calib)} records, "
      f"{df_calib['category'].nunique()}/80 categories observed")

os.makedirs("/kaggle/working/tables", exist_ok=True)
os.makedirs("/kaggle/working/results", exist_ok=True)
df_calib.to_csv("/kaggle/working/tables/table_calibration_n50_swinb.csv", index=False)
print("✅ Saved: table_calibration_n50_swinb.csv")
print()

# ─────────────────────────────────────────────────────────────
# SECTION G — Online memory (Welford stats, warm-started from
# calibration), sequential test on image_files[:10]
# ─────────────────────────────────────────────────────────────
class RunningStats:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
    @property
    def variance(self):
        return self.M2 / self.n if self.n > 1 else 0.0
    @property
    def std(self):
        return self.variance ** 0.5
    def zscore(self, x, eps=1e-6):
        return (x - self.mean) / (self.std + eps)

memory = {}
global_stats = {}
for _, row in df_calib.iterrows():
    c, cat, attr, kl = row["corruption"], row["category"], row["attribute"], row["kl"]
    memory.setdefault(c, {}).setdefault(cat, {}).setdefault(attr, RunningStats()).update(kl)
    global_stats.setdefault(c, RunningStats()).update(kl)

def get_stats(corruption, category, attribute, min_n=2):
    cell = memory.get(corruption, {}).get(category, {}).get(attribute)
    if cell is not None and cell.n >= min_n:
        return cell
    return global_stats.setdefault(corruption, RunningStats())

LAMBDA_MEM = 1.0

def memory_recal_for_image(c_img, img_id, corruption):
    boxes_ref, scores_ref, labels_ref, qidx_ref, raw_ref, dist_ref = run_dino_with_categories_raw(c_img)
    preds_base = [{"image_id": img_id, "category_id": COCO_MAP[l],
                   "bbox": [b[0].item(), b[1].item(), (b[2]-b[0]).item(), (b[3]-b[1]).item()],
                   "score": s.item()}
                  for b, s, l in zip(boxes_ref, scores_ref, labels_ref) if l in COCO_MAP]
    mAP_base, _ = compute_map(preds_base, coco_gt, [img_id])
    if len(boxes_ref) == 0:
        return mAP_base, mAP_base, 0

    view_data = []
    for v in range(1, N_STRONG_VIEWS + 1):
        aug = strong_view(c_img, v)
        boxes_v, scores_v, labels_v, qidx_v, raw_v, dist_v = run_dino_with_categories_raw(aug)
        view_data.append((VIEW_NAMES[v], boxes_v, raw_v))

    recal_scores, detection_kls = [], []
    for i in range(len(boxes_ref)):
        ref_box = boxes_ref[i].unsqueeze(0)
        kl_per_attr = {}
        for (attr_name, boxes_v, raw_v) in view_data:
            if len(boxes_v) == 0:
                continue
            ious = box_iou(ref_box, boxes_v)
            max_iou, best_j = ious[0].max(0)
            if max_iou.item() >= 0.40:
                kl_per_attr[attr_name] = bernoulli_kl(raw_ref[i], raw_v[best_j.item()])
        cat = labels_ref[i]
        zscores = []
        for attr, kl in kl_per_attr.items():
            stats = get_stats(corruption, cat, attr)
            zscores.append(stats.zscore(kl))
            detection_kls.append((cat, attr, kl))
        surprise_z = float(np.mean(zscores)) if zscores else 0.0
        recal_scores.append(scores_ref[i].item() * np.exp(-LAMBDA_MEM * max(surprise_z, 0.0)))

    preds_recal = [{"image_id": img_id, "category_id": COCO_MAP[l],
                    "bbox": [b[0].item(), b[1].item(), (b[2]-b[0]).item(), (b[3]-b[1]).item()],
                    "score": rs}
                   for b, rs, l in zip(boxes_ref, recal_scores, labels_ref) if l in COCO_MAP]
    mAP_recal, _ = compute_map(preds_recal, coco_gt, [img_id])

    for cat, attr, kl in detection_kls:
        memory.setdefault(corruption, {}).setdefault(cat, {}).setdefault(attr, RunningStats()).update(kl)
        global_stats.setdefault(corruption, RunningStats()).update(kl)

    return mAP_base, mAP_recal, len(boxes_ref)

results_mem = []
for corruption in DIAG_CORRUPTIONS:
    for img_path in tqdm(image_files[:DIAG_N], desc=f"{corruption} sev{DIAG_SEVERITY} (online)"):
        img_id  = img_id_map[os.path.basename(img_path)]
        raw_img = loaded_images[img_path]
        c_img   = apply_corruption_deterministic(raw_img, img_id, corruption, DIAG_SEVERITY)
        mAP_base, mAP_recal, n_dets = memory_recal_for_image(c_img, img_id, corruption)
        results_mem.append({"corruption": corruption, "img_id": img_id,
                             "mAP_baseline": mAP_base, "mAP_recal": mAP_recal,
                             "vs_baseline": mAP_recal - mAP_base,
                             "beats_baseline": mAP_recal > mAP_base, "n_detections": n_dets})

df_mem = pd.DataFrame(results_mem)
print()
print("=" * 65)
print("SECTION G RESULTS -- Online Memory, sequential test (N=10)")
print("=" * 65)
print(df_mem.groupby("corruption")[["vs_baseline"]].mean())
print(f"\nOverall vs_baseline: {df_mem['vs_baseline'].mean():+.4f}")
print(f"Beats baseline: {df_mem['beats_baseline'].sum()}/{len(df_mem)}")
df_mem.to_csv("/kaggle/working/tables/table_online_memory_swinb.csv", index=False)
print("✅ Saved: table_online_memory_swinb.csv")
print()

# ─────────────────────────────────────────────────────────────
# SECTION H — Held-out validation on genuinely fresh images (60:80)
# Memory reset and re-warm-started from calibration only, no
# carryover from Section G's sequential test.
# ─────────────────────────────────────────────────────────────
memory = {}
global_stats = {}
for _, row in df_calib.iterrows():
    c, cat, attr, kl = row["corruption"], row["category"], row["attribute"], row["kl"]
    memory.setdefault(c, {}).setdefault(cat, {}).setdefault(attr, RunningStats()).update(kl)
    global_stats.setdefault(c, RunningStats()).update(kl)

HOLDOUT_START, HOLDOUT_N = 60, 20
results_holdout = []
for corruption in DIAG_CORRUPTIONS:
    for img_path in tqdm(image_files[HOLDOUT_START:HOLDOUT_START+HOLDOUT_N],
                          desc=f"{corruption} sev{DIAG_SEVERITY} (held-out)"):
        img_id  = img_id_map[os.path.basename(img_path)]
        raw_img = loaded_images[img_path]
        c_img   = apply_corruption_deterministic(raw_img, img_id, corruption, DIAG_SEVERITY)
        mAP_base, mAP_recal, n_dets = memory_recal_for_image(c_img, img_id, corruption)
        results_holdout.append({"corruption": corruption, "img_id": img_id,
                                 "mAP_baseline": mAP_base, "mAP_recal": mAP_recal,
                                 "vs_baseline": mAP_recal - mAP_base,
                                 "beats_baseline": mAP_recal > mAP_base, "n_detections": n_dets})

df_holdout = pd.DataFrame(results_holdout)
print()
print("=" * 65)
print("SECTION H RESULTS -- Held-Out Validation (N=20, fresh images)")
print("=" * 65)
print(df_holdout.groupby("corruption")[["vs_baseline"]].mean())
print(f"\nOverall vs_baseline: {df_holdout['vs_baseline'].mean():+.4f}")
print(f"Beats baseline: {df_holdout['beats_baseline'].sum()}/{len(df_holdout)}")
df_holdout.to_csv("/kaggle/working/tables/table_holdout_validation_swinb.csv", index=False)
print("✅ Saved: table_holdout_validation_swinb.csv")
print()
print("=" * 65)
print("BLOCK 25 COMPLETE -- Table 4.12 ready")
print("=" * 65)
