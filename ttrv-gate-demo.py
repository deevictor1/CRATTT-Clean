"""
STANDALONE TTRV TRUST-ROUTING DEMO
CRATTT / MRES7015 -- Dada Victor Damilare

Leaned-down extract of the full CRATTT notebook: only GroundingDINO
(Swin-B) plus the TTRV multi-view consensus gate. No CLIP, no YOLO,
no COCO dataset needed, the demo takes user-uploaded images directly.

B24['snr_baseline']/['tau_internal'] use the confirmed calibration
(32.842 / 0.9796) from cratt-rq1-new.ipynb Block 24c, the same
calibration behind the paper's reported results, independently
validated across all four corruption types at severities 1, 3, and 5
(twelve conditions, all p<0.006). See demo_calibration_record.json
for the exact values used in each run.
"""

# Must be set BEFORE torch/CUDA initializes -- required for
# deterministic cuBLAS matmul/attention ops
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

!pip install imagecorruptions -q
!pip install gradio -q

import hashlib
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision.ops import box_iou
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from PIL import Image as PILImage, ImageDraw
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from imagecorruptions import corrupt as ic_corrupt
import gradio as gr

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

hf_token = None
try:
    from kaggle_secrets import UserSecretsClient
    from huggingface_hub import login
    user_secrets = UserSecretsClient()
    hf_token = user_secrets.get_secret("HF_TOKEN")
    login(token=hf_token)
    print("✅ HuggingFace: Connected")
except Exception as e:
    print(f"ℹ️  Running without HF token ({e})")

print("Loading GroundingDINO-base (Swin-B)...")
DINO_MODEL_ID = "IDEA-Research/grounding-dino-base"
dino_processor = AutoProcessor.from_pretrained(DINO_MODEL_ID, token=hf_token)
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
    DINO_MODEL_ID, token=hf_token
).to(device)
dino_model.eval()
dino_params = sum(p.numel() for p in dino_model.parameters())
print(f"✅ GroundingDINO-base (Swin-B) loaded -- {dino_params:,} params")

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
CRATTT_PARAMS = {"dino_text_thr": 0.12}
print(f"✅ {len(COCO_MAP)} COCO categories, DINO_TEXT_PROMPT built")

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
print(f"✅ Category token spans built -- {n_mapped}/{len(category_order)} categories mapped")

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

BOX_THRESHOLD = 0.25

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

# ==============================================================
# B24 GATE CONFIGURATION
# B24['snr_baseline']/['tau_internal'] use the confirmed calibration
# (32.842 / 0.9796) from cratt-rq1-new.ipynb Block 24c. This calibration
# was independently validated across all four corruption types at
# severities 1, 3, and 5 (12 conditions, all p<0.006) via a dedicated
# debugging session; see the associated checkpoint CSVs.
# ==============================================================
B24 = {
    "n_views": 5, "iou_thr": 0.40, "alpha": 0.40, "beta": 0.60,
    "snr_eps": 1e-4, "snr_baseline": 32.842, "tau_internal": 0.9796,
}

# ============================================================
# CALIBRATION RECORD -- logs exactly which snr_baseline /
# tau_internal values this run used, so this session's demo
# output can always be traced back to a specific calibration
# pass rather than becoming a fourth untraceable number.
# ============================================================
import json, datetime
calib_record = {
    "snr_baseline": B24["snr_baseline"],
    "tau_internal": B24["tau_internal"],
    "source": "cratt-rq1-new.ipynb Block 24c -- confirmed behind the reported "
              "N=2,400 adaptation result and N=2,624 reliability classifier result, "
              "independently replicated (12 conditions, all p<0.006) via Kaggle diagnostics.",
    "timestamp": datetime.datetime.now().isoformat(),
}
with open("/kaggle/working/demo_calibration_record.json", "w") as f:
    json.dump(calib_record, f, indent=2)
print(f"✅ Calibration record saved: {calib_record}")

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
        clean_label = ref_label
        if clean_label not in COCO_MAP:
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
        score_mean = float(arr.mean())
        score_std  = float(arr.std())
        s_snr = score_mean / (score_std + B24["snr_eps"])
        iou_cons = match_count / B24["n_views"]
        detections.append({
            "box": ref_box, "label": clean_label,
            "score_mean": score_mean, "score_std": score_std,
            "s_snr": s_snr, "iou_consensus": iou_cons, "match_count": match_count,
        })
    return detections

def b24_apply_gate(detections: list):
    v_boxes, v_labels, v_rjoints = [], [], []
    for d in detections:
        snr_norm = min(d["s_snr"] / max(B24["snr_baseline"], 1e-6), 2.0)
        rjoint = (d["iou_consensus"] ** B24["alpha"]) * (snr_norm ** B24["beta"])
        if rjoint >= B24["tau_internal"]:
            v_boxes.append(d["box"])
            v_labels.append(d["label"])
            v_rjoints.append(rjoint)
    return v_boxes, v_labels, v_rjoints

print("✅ TTRV gate functions ready (b24_compute_snr, b24_apply_gate)")

# Warm-up pass -- absorbs the one-time CUDA kernel selection cost
# here at startup, rather than on the first real user interaction
print("Warming up CUDA kernels...")
_dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
_ = b24_compute_snr(_dummy_img)
print("✅ Warm-up complete -- all detections from here on should be fully consistent")

# fog removed: it has never appeared in any evaluated result in the
# paper or dissertation, so it shouldn't be offered as a demo option.
CORRUPTION_OPTIONS = ["none", "gaussian_noise", "motion_blur", "snow", "contrast"]

def draw_detections(image_np, dets, verified_boxes):
    img = PILImage.fromarray(image_np.astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(img)
    verified_set = {tuple(b.tolist()) for b in verified_boxes}
    for d in dets:
        box = d["box"].tolist()
        is_verified = tuple(d["box"].tolist()) in verified_set
        color = (29, 158, 117) if is_verified else (186, 117, 23)
        draw.rectangle(box, outline=color, width=3)
        label = f'{d["label"]} ({"verified" if is_verified else "flagged"})'
        draw.rectangle([box[0], box[1] - 16, box[0] + len(label) * 7, box[1]], fill=color)
        draw.text((box[0] + 2, box[1] - 15), label, fill=(255, 255, 255))
    return np.array(img)

def run_demo(image, corruption, severity):
    if image is None:
        return None, "Upload an image first."
    raw_img = np.array(PILImage.fromarray(image).convert("RGB"))
    if corruption == "none":
        c_img = raw_img
    else:
        h = hashlib.sha256()
        h.update(raw_img.tobytes())
        h.update(corruption.encode("utf-8"))
        h.update(str(int(severity)).encode("utf-8"))
        seed_val = int(h.hexdigest(), 16) % (2**31)
        np.random.seed(seed_val)
        random.seed(seed_val)
        c_img = ic_corrupt(raw_img, corruption_name=corruption, severity=int(severity))

    # Throwaway warm-up pass at THIS image's actual tensor shape, then
    # reseed and recompute for the real result. Doubles compute per
    # click, but directly targets the shape-specific first-call cost
    # the global dummy-image warm-up missed.
    _ = b24_compute_snr(c_img)
    np.random.seed(seed_val if corruption != "none" else 0)
    random.seed(seed_val if corruption != "none" else 0)

    dets = b24_compute_snr(c_img)
    if not dets:
        return c_img, "No detections found on this image."
    v_boxes, v_labels, v_rjoints = b24_apply_gate(dets)
    annotated = draw_detections(c_img, dets, v_boxes)
    summary = (
        f"{len(dets)} total detections | "
        f"{len(v_boxes)} verified (auto-accept) | "
        f"{len(dets) - len(v_boxes)} flagged for review\n"
        f"Gate threshold (tau): {B24['tau_internal']:.3f}"
    )
    return annotated, summary

with gr.Blocks(title="TTRV Trust-Routing Demo", analytics_enabled=False) as demo:
    gr.Markdown("## TTRV gate demo — verify before you trust")
    gr.Markdown(
        "Upload an image, optionally apply a corruption, and see which "
        "detections the gate would auto-accept (teal) versus flag for "
        "human review (amber)."
    )
    gr.Markdown(
    	"_This demo uses the confirmed calibration behind the paper's "
    	"reported adaptation and reliability-classifier results, "
    	"independently validated across twelve severity-corruption "
    	"conditions. It illustrates gate behaviour qualitatively on "
    	"individual images; it does not reproduce the full statistical "
    	"test._"
    )
    gr.Markdown(
        "_Note: detections whose consensus score sits within roughly 1% "
        "of the gate threshold may occasionally shift between verified "
        "and flagged between runs, due to GPU floating-point precision "
        "in the underlying model. This is a known property of GPU "
        "inference generally, not an instability in the gate itself._"
    )
    with gr.Row():
        with gr.Column():
            img_in = gr.Image(label="Input image", type="numpy")
            corruption_in = gr.Dropdown(CORRUPTION_OPTIONS, value="gaussian_noise", label="Corruption")
            severity_in = gr.Slider(1, 5, value=5, step=1, label="Severity")
            run_btn = gr.Button("Run gate")
        with gr.Column():
            img_out = gr.Image(label="Verified (teal) vs flagged (amber)")
            summary_out = gr.Textbox(label="Summary", lines=3)
    run_btn.click(run_demo, inputs=[img_in, corruption_in, severity_in], outputs=[img_out, summary_out])

demo.launch(share=True, debug=False)
