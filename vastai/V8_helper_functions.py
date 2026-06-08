# ============================================================
# BLOCK V8: Helper Functions
# Run on: Vast.ai
# Identical logic to Kaggle notebook
# ============================================================

import torch
import numpy as np
import json
import os
import cv2
from pycocotools.cocoeval import COCOeval


def dino_to_coco_format(res, img_id):
    preds  = []
    boxes  = res["boxes"]
    scores = res["scores"]
    labels = res.get("text_labels", res.get("labels", []))
    for i, (box, score) in enumerate(zip(boxes, scores)):
        if i >= len(labels):
            continue
        label = labels[i]
        if isinstance(label, str):
            clean = label.lower().replace(".", "").strip()
        elif isinstance(label, int):
            clean = COCO_CLASSES[label] \
                if label < len(COCO_CLASSES) else None
        else:
            continue
        if clean is None:
            continue
        cat_id = COCO_MAP.get(clean)
        if cat_id is None:
            continue
        b = box.tolist()
        preds.append({
            "image_id":    img_id,
            "category_id": cat_id,
            "bbox": [b[0], b[1], b[2]-b[0], b[3]-b[1]],
            "score":       float(score)
        })
    return preds


def yolo_to_coco_format(results, img_id):
    preds = []
    if results is None or len(results) == 0:
        return preds
    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return preds
    boxes   = result.boxes.xyxy.cpu().numpy()
    scores  = result.boxes.conf.cpu().numpy()
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    for box, score, cls_id in zip(boxes, scores, cls_ids):
        if cls_id >= len(COCO_CLASSES):
            continue
        class_name = COCO_CLASSES[cls_id]
        cat_id     = COCO_MAP.get(class_name)
        if cat_id is None:
            continue
        x1, y1, x2, y2 = box
        preds.append({
            "image_id":    img_id,
            "category_id": cat_id,
            "bbox": [float(x1), float(y1),
                     float(x2-x1), float(y2-y1)],
            "score":       float(score)
        })
    return preds


def compute_map_coco(predictions, coco_gt, img_ids):
    """
    Runs COCOeval ONCE on ALL predictions for a configuration.
    Never called per-image. Matches Kaggle methodology exactly.
    """
    if not predictions:
        return 0.0
    try:
        coco_dt   = coco_gt.loadRes(predictions)
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.params.imgIds = img_ids
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        return float(evaluator.stats[0])
    except Exception as e:
        print(f"  COCOeval error: {e}")
        return 0.0


def ckpt_filename(model_name, corruption, severity):
    return os.path.join(
        CKPT_DIR,
        f"{model_name}_{corruption}_sev{severity}.json"
    )

def save_ckpt(model_name, corruption, severity, data):
    path = ckpt_filename(model_name, corruption, severity)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def load_ckpt(model_name, corruption, severity):
    path = ckpt_filename(model_name, corruption, severity)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


print("✅ Helper functions defined:")
print("   dino_to_coco_format")
print("   yolo_to_coco_format")
print("   compute_map_coco (single COCOeval call)")
print("   save_ckpt / load_ckpt (per severity per model)")
