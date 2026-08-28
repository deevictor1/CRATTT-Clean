# ============================================================
# BLOCK 3: Global Constants
# Single source of truth for all experimental parameters.
# Never hardcode these values in subsequent blocks.
# ============================================================

import torch
import torch.nn.functional as F

# --- 3.1 COCO Category Mapping ---
# Verified against official COCO 2017 annotation file
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

COCO_CLASSES = list(COCO_MAP.keys())  # 80 classes
print(f"✅ COCO_MAP loaded: {len(COCO_MAP)} categories")

# --- 3.2 Verify COCO_MAP Against Ground Truth ---
# This catches any ID mismatches before they silently corrupt mAP scores
from pycocotools.coco import COCO

ANN_PATH = ("/kaggle/input/datasets/awsaf49/"
            "coco-2017-dataset/coco2017/annotations/"
            "instances_val2017.json")
IMAGE_DIR = ("/kaggle/input/datasets/awsaf49/"
             "coco-2017-dataset/coco2017/val2017")

coco_gt = COCO(ANN_PATH)
official_cats = {
    cat['name']: cat['id']
    for cat in coco_gt.loadCats(coco_gt.getCatIds())
}

mismatches = []
for name, our_id in COCO_MAP.items():
    official_id = official_cats.get(name)
    if official_id is None:
        mismatches.append(f"  NOT FOUND in COCO: '{name}'")
    elif official_id != our_id:
        mismatches.append(
            f"  MISMATCH: '{name}' → ours={our_id}, "
            f"official={official_id}"
        )

if mismatches:
    print("⚠️  COCO_MAP mismatches detected:")
    for m in mismatches:
        print(m)
else:
    print("✅ COCO_MAP verified: all 80 IDs match official annotations")

# --- 3.3 Text Prompts ---
# GroundingDINO expects dot-separated class names
DINO_TEXT_PROMPT = " . ".join(COCO_CLASSES) + " ."

# YOLO-World expects plain class names
# Set classes here using the CPU handoff pattern
yolo_model.to('cpu')
yolo_model.set_classes(COCO_CLASSES)
yolo_model.to(device)
print("✅ YOLO-World classes set: 80 COCO categories")


# --- 3.4 CLIP Per-Class Text Embeddings ---
print("\nGenerating per-class CLIP text embeddings...")

with torch.no_grad():
    embeddings = []
    
    for class_name in COCO_CLASSES:
        inputs = clip_processor(
            text=[class_name],
            return_tensors="pt",
            padding=True
        ).to(device)
        
        # Bypass get_text_features entirely
        # Call text_model directly, extract last hidden state,
        # then apply projection manually
        text_out = clip_model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        
        # Pooled output is the [EOS] token representation
        # Shape: [1, hidden_dim] = [1, 512]
        pooled = text_out.pooler_output
        
        # Apply the learned projection to get final embedding
        projected = clip_model.text_projection(pooled)
        embeddings.append(projected)
    
    clip_text_features = torch.cat(embeddings, dim=0)
    clip_text_features = F.normalize(clip_text_features, p=2, dim=-1)

print(f"✅ CLIP text embeddings: {clip_text_features.shape}")
print(f"   Expected: torch.Size([80, 512])")

person_idx = COCO_CLASSES.index('person')
car_idx    = COCO_CLASSES.index('car')
chair_idx  = COCO_CLASSES.index('chair')
dog_idx    = COCO_CLASSES.index('dog')

sim_person_car  = (clip_text_features[person_idx] @ clip_text_features[car_idx]).item()
sim_person_dog  = (clip_text_features[person_idx] @ clip_text_features[dog_idx]).item()
sim_car_chair   = (clip_text_features[car_idx] @ clip_text_features[chair_idx]).item()

print(f"\n   Similarity checks (all must be < 0.98):")
print(f"   person vs car:   {sim_person_car:.4f}")
print(f"   person vs dog:   {sim_person_dog:.4f}")
print(f"   car vs chair:    {sim_car_chair:.4f}")

all_distinct = all(
    s < 0.98 for s in [sim_person_car, sim_person_dog, sim_car_chair]
)
print(f"\n   {'✅ All embeddings are distinct' if all_distinct else '❌ Still identical — check CLIP model'}")


# --- 3.5 ImageNet-C Corruption Protocol ---
CORRUPTION_CATEGORIES = {
    "Noise":   ["gaussian_noise", "shot_noise", "impulse_noise"],
    "Blur":    ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"],
    "Weather": ["snow", "frost", "fog", "brightness"],
    "Digital": ["contrast", "elastic_transform",
                "pixelate", "jpeg_compression"]
}
ALL_CORRUPTIONS = [c for cats in CORRUPTION_CATEGORIES.values()
                   for c in cats]
SEVERITIES = [1, 2, 3, 4, 5]

print(f"\n✅ ImageNet-C protocol:")
print(f"   Categories: {list(CORRUPTION_CATEGORIES.keys())}")
print(f"   Total corruptions: {len(ALL_CORRUPTIONS)}")
print(f"   Severities: {SEVERITIES}")

# --- 3.6 CRATTT Hyperparameters ---
# Fixed values from Chapter 3 pilot analysis
# Do not change these for primary results
# Use ablation blocks for sensitivity analysis
CRATTT_PARAMS = {
    "alpha":          0.4,   # DINO weight in Rjoint
    "beta":           0.6,   # Oracle weight in Rjoint
    "tau":            0.25,  # Fixed verification threshold
    "dino_text_thr":  0.12,  # GroundingDINO text threshold (permissive)
    "yolo_conf":      0.12,  # YOLO confidence (matched to DINO)
    "max_regions":    15,    # BARON max region crops
    "region_size":    (224, 224),  # CLIP input size
}
print(f"\n✅ CRATTT hyperparameters locked:")
for k, v in CRATTT_PARAMS.items():
    print(f"   {k}: {v}")

# --- 3.7 Evaluation Settings ---
EVAL_PARAMS = {
    "num_images":   20,
    "num_pilot":    5,
    "save_dir":     DIRS["results"],
    "fig_dir":      DIRS["figures"],
    "table_dir":    DIRS["tables"],
    "ckpt_dir":     DIRS["checkpoints"],
}
print(f"\n✅ Evaluation parameters:")
for k, v in EVAL_PARAMS.items():
    print(f"   {k}: {v}")

print("\n" + "="*50)
print("BLOCK 3 COMPLETE — Constants defined")
print("="*50)
