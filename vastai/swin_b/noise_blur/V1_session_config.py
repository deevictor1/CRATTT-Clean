# ============================================================
# BLOCK V1: Session Configuration
# Vast.ai, Swin-B DINO, Session 2: Noise+Blur
# Same instance as Session 1, continuing
# ============================================================
MODEL          = "dino"
CORRUPTIONS    = "Noise+Blur"
N_IMAGES       = 5000
GLASS_BLUR_N   = 1000               # documented exception, matches tiny-model sweep

SEED           = 42
DINO_TEXT_THR  = 0.12
BOX_THRESHOLD  = 0.25
TAU            = 0.335
ALPHA          = 0.4
BETA           = 0.6
CLEAN_MAP_DINO = 0.5461
CLEAN_MAP_YOLO = 0.5221

DINO_MODEL_ID  = "IDEA-Research/grounding-dino-base"

RUN_NAME = f"{MODEL}_{CORRUPTIONS.replace('+', '_')}_swinb"

print("=" * 55)
print("SESSION CONFIGURATION")
print("=" * 55)
print(f"Model           : {MODEL.upper()} (Swin-B)")
print(f"Corruptions     : {CORRUPTIONS}")
print(f"Images          : {N_IMAGES} (glass_blur exception: {GLASS_BLUR_N})")
print(f"Run name        : {RUN_NAME}")
print(f"Clean mAP       : {CLEAN_MAP_DINO}")
