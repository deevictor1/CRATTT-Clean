# ============================================================
# BLOCK V5: Load Models
# Run on: Vast.ai
# DINO sessions: loads GroundingDINO + CLIP
# YOLO sessions: loads YOLO-World + CLIP
# ============================================================

import torch
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    CLIPProcessor,
    CLIPModel
)

print("Loading CLIP ViT-B/32 (Transformers)...")
CLIP_MODEL_ID  = "openai/clip-vit-base-patch32"
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
clip_model     = CLIPModel.from_pretrained(
    CLIP_MODEL_ID
).to(device)
clip_model.eval()
for param in clip_model.parameters():
    param.requires_grad = False

clip_params = sum(p.numel() for p in clip_model.parameters())
print(f"✅ CLIP ViT-B/32: {clip_params:,} params")

dino_model     = None
dino_processor = None
yolo_model     = None
DINO_TEXT_PROMPT = None

if MODEL == "dino":
    print("\nLoading GroundingDINO-base (Swin-B)...")
    DINO_TEXT_PROMPT = (
        "person . bicycle . car . motorcycle . airplane . bus . "
        "train . truck . boat . traffic light . fire hydrant . "
        "stop sign . parking meter . bench . bird . cat . dog . "
        "horse . sheep . cow . elephant . bear . zebra . giraffe . "
        "backpack . umbrella . handbag . tie . suitcase . frisbee . "
        "skis . snowboard . sports ball . kite . baseball bat . "
        "baseball glove . skateboard . surfboard . tennis racket . "
        "bottle . wine glass . cup . fork . knife . spoon . bowl . "
        "banana . apple . sandwich . orange . broccoli . carrot . "
        "hot dog . pizza . donut . cake . chair . couch . "
        "potted plant . bed . dining table . toilet . tv . laptop . "
        "mouse . remote . keyboard . cell phone . microwave . oven . "
        "toaster . sink . refrigerator . book . clock . vase . "
        "scissors . teddy bear . hair drier . toothbrush ."
    )
    dino_processor = AutoProcessor.from_pretrained(DINO_MODEL_ID)
    dino_model = AutoModelForZeroShotObjectDetection\
        .from_pretrained(DINO_MODEL_ID).to(device)
    dino_model.eval()
    for param in dino_model.parameters():
        param.requires_grad = False

    dino_params = sum(
        p.numel() for p in dino_model.parameters()
    )
    print(f"✅ GroundingDINO-base (Swin-B): {dino_params:,} params")

elif MODEL == "yolo":
    print("\nLoading YOLO-World-large...")
    from ultralytics import YOLOWorld
    yolo_model = YOLOWorld("yolov8x-worldv2.pt")
    yolo_model.set_classes(COCO_CLASSES)
    print(f"✅ YOLO-World-large loaded")

vram_used  = torch.cuda.memory_allocated() / 1e9
vram_total = torch.cuda.get_device_properties(0).total_memory/1e9
vram_free  = vram_total - vram_used
print(f"\n✅ VRAM used : {vram_used:.2f} GB")
print(f"   VRAM free : {vram_free:.2f} GB")
print(f"   VRAM total: {vram_total:.2f} GB")
