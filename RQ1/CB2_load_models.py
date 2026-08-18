# ============================================================
# BLOCK 2: Model Loading
# Loads GroundingDINO, YOLO-World, and CLIP Oracle once.
# Models are not reloaded in subsequent blocks.
# ============================================================
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLOWorld

print("Loading models to device:", device)
print("="*50)

# --- 2.1 GroundingDINO ---
print("\n[1/3] Loading GroundingDINO-base (Swin-B)...")
DINO_MODEL_ID = "IDEA-Research/grounding-dino-base"

dino_processor = AutoProcessor.from_pretrained(DINO_MODEL_ID, token=hf_token)
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
    DINO_MODEL_ID, token=hf_token
).to(device)
dino_model.eval()

dino_params = sum(p.numel() for p in dino_model.parameters())
print(f"✅ GroundingDINO-base (Swin-B) loaded")
print(f"   Parameters: {dino_params:,}")
print(f"   VRAM used so far: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# --- 2.2 YOLO-World ---
print("\n[2/3] Loading YOLO-World...")

# CPU handoff pattern: set classes on CPU to avoid CUDA text-encoding bug
yolo_model = YOLOWorld('yolov8x-worldv2.pt')
yolo_model.to('cpu')

print(f"✅ YOLO-World loaded")
print(f"   VRAM used so far: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# --- 2.3 CLIP Oracle ---
print("\n[3/3] Loading CLIP Oracle (ViT-B/32)...")
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device)
clip_model.eval()

clip_params = sum(p.numel() for p in clip_model.parameters())
print(f"✅ CLIP Oracle loaded")
print(f"   Parameters: {clip_params:,}")
print(f"   VRAM used so far: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# --- 2.4 VRAM Summary ---
vram_allocated = torch.cuda.memory_allocated() / 1e9
vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
vram_free = vram_total - vram_allocated

print("\n--- VRAM Summary ---")
print(f"Allocated: {vram_allocated:.2f} GB")
print(f"Free:      {vram_free:.2f} GB")
print(f"Total:     {vram_total:.2f} GB")

if vram_free < 4.0:
    print("⚠️  WARNING: Less than 4GB free.")
else:
    print("✅ VRAM headroom is sufficient for inference.")

print("\n" + "="*50)
print("BLOCK 2 COMPLETE — All models loaded")
print("="*50)
