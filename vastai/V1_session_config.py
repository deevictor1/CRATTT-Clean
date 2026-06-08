# ============================================================
# BLOCK V1: Session Configuration
# THE ONLY BLOCK YOU CHANGE BETWEEN SESSIONS
# Run on: Vast.ai
#
# Session 1: MODEL="dino", CORRUPTIONS="Weather+Digital"
# Session 2: MODEL="dino", CORRUPTIONS="Noise+Blur"
# Session 3: MODEL="yolo", CORRUPTIONS="Weather+Digital"
# Session 4: MODEL="yolo", CORRUPTIONS="Noise+Blur"
# ============================================================

# --- CHANGE THESE THREE LINES EACH SESSION ---
MODEL       = "dino"            # "dino" or "yolo"
CORRUPTIONS = "Weather+Digital" # "Weather+Digital" or "Noise+Blur"
N_IMAGES    = 5000              # Keep at 5000 for all sessions

# --- DO NOT CHANGE BELOW THIS LINE ---
SEED           = 42
DINO_TEXT_THR  = 0.12
YOLO_CONF_THR  = 0.12
TAU            = 0.335
ALPHA          = 0.4
BETA           = 0.6
CLEAN_MAP_DINO = 0.3598  # From Kaggle Block 5
CLEAN_MAP_YOLO = 0.5221  # From Kaggle Block 5

RUN_NAME = f"{MODEL}_{CORRUPTIONS.replace('+', '_')}"

print("=" * 55)
print("SESSION CONFIGURATION")
print("=" * 55)
print(f"Model           : {MODEL.upper()}")
print(f"Corruptions     : {CORRUPTIONS}")
print(f"Images          : {N_IMAGES}")
print(f"Run name        : {RUN_NAME}")
print(f"Clean mAP       : "
      f"{CLEAN_MAP_DINO if MODEL == 'dino' else CLEAN_MAP_YOLO}")
print()
print("Verify these match your Kaggle settings:")
print(f"  TAU={TAU}, ALPHA={ALPHA}, BETA={BETA}")
print(f"  DINO threshold={DINO_TEXT_THR}")
