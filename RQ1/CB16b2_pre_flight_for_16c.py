# ============================================================
# PRE-FLIGHT CHECK before Block 16c
# Verifies everything Block 16c needs is present, and reports
# dino_model's CURRENT LoRA state rather than assuming it.
# ============================================================
import torch

checks = {
    "device":                       'device' in dir(),
    "dino_model":                   'dino_model' in dir(),
    "dino_processor":                'dino_processor' in dir(),
    "DINO_MODEL_ID":                'DINO_MODEL_ID' in dir(),
    "hf_token":                     'hf_token' in dir(),
    "CRATTT_PARAMS":                'CRATTT_PARAMS' in dir(),
    "image_files":                  'image_files' in dir() and len(image_files) == 20,
    "loaded_images":                'loaded_images' in dir(),
    "coco_gt":                      'coco_gt' in dir(),
    "COCO_CLASSES":                 'COCO_CLASSES' in dir(),
    "clip_text_features":           'clip_text_features' in dir(),
    "compute_map":                  'compute_map' in dir(),
    "dino_to_coco_format":          'dino_to_coco_format' in dir(),
    "run_crattt_inference":         'run_crattt_inference' in dir(),
    "DINO_TEXT_PROMPT":             'DINO_TEXT_PROMPT' in dir(),
    "LORA_RANK":                    'LORA_RANK' in dir(),
    "LORA_ALPHA":                   'LORA_ALPHA' in dir(),
    "inject_lora_grounding_dino":   'inject_lora_grounding_dino' in dir(),
}

all_ok = True
for name, status in checks.items():
    icon = "✅" if status else "❌"
    print(f"{icon} {name}")
    if not status:
        all_ok = False

print()

# Report dino_model's ACTUAL current state, don't assume it
if checks["dino_model"]:
    current_trainable = sum(
        p.numel() for p in dino_model.parameters() if p.requires_grad
    )
    current_lora_b = sum(
        1 for n, _ in dino_model.named_parameters() if 'lora_B' in n
    )
    print(f"ℹ️  dino_model current state:")
    print(f"   Trainable params now : {current_trainable:,}")
    print(f"   lora_B matrices now  : {current_lora_b}")
    if current_trainable == 73_728:
        print(f"   → Looks like rank=4 (Block 14 config) is active")
    elif current_trainable == 294_912:
        print(f"   → Looks like rank=16 is ALREADY active — "
              f"did Block 16c already run in this session?")
    elif current_trainable == 0:
        print(f"   → No trainable params — model may be freshly "
              f"loaded with no LoRA injected yet")
    else:
        print(f"   ⚠️  Unrecognised trainable param count — "
              f"inspect before proceeding")

print()
if all_ok:
    print("✅ All dependencies present — safe to run Block 16c")
else:
    print("❌ Missing dependencies — rerun the flagged blocks first")
