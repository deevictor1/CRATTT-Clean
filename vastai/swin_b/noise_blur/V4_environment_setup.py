# ============================================================
# BLOCK V4: Environment Setup
# Run on: Vast.ai
# Sets seeds, directories, COCO ground truth, class maps
# ============================================================

import torch
import numpy as np
import random
import os
import json
from pycocotools.coco import COCO

COCO_BASE = "/workspace/coco"
IMAGE_DIR = f"{COCO_BASE}/val2017"
ANN_PATH  = f"{COCO_BASE}/annotations/instances_val2017.json"
WORKSPACE   = "/workspace/crattt"
RESULTS_DIR = f"{WORKSPACE}/results"
TABLES_DIR  = f"{WORKSPACE}/tables"
CKPT_DIR    = f"{WORKSPACE}/checkpoints"

for d in [RESULTS_DIR, TABLES_DIR, CKPT_DIR]:
    os.makedirs(d, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# NumPy compatibility patches
if not hasattr(np, 'float'):   np.float   = float
if not hasattr(np, 'int'):     np.int     = int
if not hasattr(np, 'bool'):    np.bool    = bool
if not hasattr(np, 'complex'): np.complex = complex

device = torch.device("cuda" if torch.cuda.is_available()
                       else "cpu")
print(f"✅ Device: {device}")

if torch.cuda.is_available():
    gpu  = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory/1e9
    print(f"   GPU  : {gpu}")
    print(f"   VRAM : {vram:.1f} GB")

coco_gt = COCO(ANN_PATH)
print(f"✅ COCO GT loaded")

COCO_CLASSES = [
    'person','bicycle','car','motorcycle','airplane','bus',
    'train','truck','boat','traffic light','fire hydrant',
    'stop sign','parking meter','bench','bird','cat','dog',
    'horse','sheep','cow','elephant','bear','zebra','giraffe',
    'backpack','umbrella','handbag','tie','suitcase','frisbee',
    'skis','snowboard','sports ball','kite','baseball bat',
    'baseball glove','skateboard','surfboard','tennis racket',
    'bottle','wine glass','cup','fork','knife','spoon','bowl',
    'banana','apple','sandwich','orange','broccoli','carrot',
    'hot dog','pizza','donut','cake','chair','couch',
    'potted plant','bed','dining table','toilet','tv','laptop',
    'mouse','remote','keyboard','cell phone','microwave','oven',
    'toaster','sink','refrigerator','book','clock','vase',
    'scissors','teddy bear','hair drier','toothbrush'
]

cats     = coco_gt.loadCats(coco_gt.getCatIds())
COCO_MAP = {c['name']: c['id'] for c in cats}
print(f"✅ COCO_MAP: {len(COCO_MAP)} categories verified")

CRATTT_PARAMS = {
    "alpha":         ALPHA,
    "beta":          BETA,
    "tau":           TAU,
    "dino_text_thr": DINO_TEXT_THR,
    "max_regions":   15
}
print(f"✅ CRATTT_PARAMS: tau={TAU}, alpha={ALPHA}, beta={BETA}")
print(f"✅ Workspace: {WORKSPACE}")
