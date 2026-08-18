# ============================================================
# BLOCK 14: LoRA Adapter Injection -- RANK 4 (primary)
# Injects trainable LoRA matrices into GroundingDINO's
# cross-modal neck (encoder + decoder query/value projections).
# All backbone weights remain frozen.
# Only LoRA matrices A and B are trainable.
# Reference: Hu et al. (2021) "LoRA: Low-Rank Adaptation"
# ============================================================
#
# NOTE ON RANK: this is the rank=4 configuration used for every
# primary result in Chapter 4 (Tables 4.6-4.11), including the
# formally powered N=2400 statistical test (Block 24e, Table 4.10).
# A separate rank=16 capacity-ablation pass (Table 4.11's last two
# rows) is run later, after Block 24f, by reloading the base model
# and re-injecting at rank=16 -- see "Block 14 (capacity ablation)"
# below. This cell stays at rank=4 throughout the main run.
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import os

# --- 14.1 LoRA Linear Layer ---
class LoRALinear(nn.Module):
    """
    Replaces a Linear layer with a LoRA-augmented version.

    Weight update: W' = W + (B @ A) * scale
    Where:
        A : [rank, in_features]  -- down-projection (Kaiming init)
        B : [out_features, rank] -- up-projection   (zero init)
        scale = lora_alpha / rank

    B=0 at initialisation ensures no change to model output
    at the start of TTT. Only A and B are trainable.
    """
    def __init__(self, linear_layer, rank=4, lora_alpha=8):
        super().__init__()

        self.in_features  = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rank         = rank
        self.scale        = lora_alpha / rank

        # Frozen original weights
        self.weight = nn.Parameter(
            linear_layer.weight.data.clone(),
            requires_grad=False
        )
        self.bias = nn.Parameter(
            linear_layer.bias.data.clone(),
            requires_grad=False
        ) if linear_layer.bias is not None else None

        # Trainable LoRA matrices
        self.lora_A = nn.Parameter(
            torch.zeros(rank, self.in_features)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_features, rank)
        )

        # Kaiming init for A, zeros for B
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base  = nn.functional.linear(x, self.weight, self.bias)
        delta = (x @ self.lora_A.T @ self.lora_B.T) * self.scale
        return base + delta

    def extra_repr(self):
        return (f"in={self.in_features}, out={self.out_features}, "
                f"rank={self.rank}, scale={self.scale:.2f}")


# --- 14.2 LoRA Injection for GroundingDINO ---
def inject_lora_grounding_dino(model, rank=4, lora_alpha=8):
    """
    Injects LoRA into query and value projections of
    GroundingDINO's cross-modal neck only.

    Targeted components:
      - model.encoder: text enhancer + fusion attention layers
      - model.decoder: self_attn + encoder_attn_text layers

    Skipped components (frozen throughout):
      - model.backbone (vision encoder)
      - model.text_backbone (BERT text encoder)
    """
    n_injected = 0

    for name, module in list(model.named_modules()):

        if not (name.startswith('model.encoder') or
                name.startswith('model.decoder')):
            continue

        if not (name.endswith('.query') or
                name.endswith('.value')):
            continue

        if not isinstance(module, nn.Linear):
            continue

        parts  = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)

        lora_layer = LoRALinear(
            module, rank=rank, lora_alpha=lora_alpha
        ).to(device)
        setattr(parent, parts[-1], lora_layer)
        n_injected += 1

    for param in model.parameters():
        param.requires_grad = False

    lora_params = []
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            param.requires_grad = True
            lora_params.append((name, param))

    return n_injected, lora_params


# --- 14.3 Inject ---
LORA_RANK  = 4
LORA_ALPHA = 8

print("Injecting LoRA adapters into GroundingDINO...")
print(f"Rank      : {LORA_RANK}")
print(f"Alpha     : {LORA_ALPHA}")
print(f"Scale     : {LORA_ALPHA / LORA_RANK:.2f}")
print(f"Targets   : query, value (encoder + decoder only)")
print()

n_injected, lora_params = inject_lora_grounding_dino(
    dino_model, rank=LORA_RANK, lora_alpha=LORA_ALPHA
)

print(f"✅ LoRA layers injected : {n_injected}")
print(f"✅ Trainable LoRA params: {len(lora_params)}")

# --- 14.4 Parameter Audit ---
total_params     = sum(p.numel() for p in dino_model.parameters())
trainable_params = sum(
    p.numel() for p in dino_model.parameters()
    if p.requires_grad
)
frozen_params = total_params - trainable_params

print(f"\n--- Parameter Audit ---")
print(f"Total parameters    : {total_params:,}")
print(f"Trainable (LoRA)    : {trainable_params:,}")
print(f"Frozen (backbone)   : {frozen_params:,}")
print(f"Trainable ratio     : "
      f"{100 * trainable_params / total_params:.4f}%")

assert trainable_params > 0, \
    "No trainable parameters -- injection failed"
assert trainable_params < total_params * 0.05, \
    "Too many trainable params -- check injection scope"
print(f"✅ Parameter audit passed")

# --- 14.5 Forward Pass Verification ---
print(f"\nVerifying forward pass after LoRA injection...")
test_img = loaded_images[image_files[0]]

inputs = dino_processor(
    images=test_img,
    text=DINO_TEXT_PROMPT,
    return_tensors="pt"
).to(device)

with torch.no_grad():
    outputs = dino_model(**inputs)

test_res = dino_processor\
    .post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        target_sizes=[test_img.shape[:2]],
        text_threshold=CRATTT_PARAMS["dino_text_thr"]
    )[0]

print(f"✅ Forward pass OK -- {len(test_res['boxes'])} detections")
print(f"   (B=0 init: output identical to pre-LoRA)")

# --- 14.6 VRAM Check ---
vram_used  = torch.cuda.memory_allocated() / 1e9
vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
vram_free  = vram_total - vram_used

print(f"\n--- VRAM After LoRA Injection ---")
print(f"Used  : {vram_used:.2f} GB")
print(f"Free  : {vram_free:.2f} GB")
print(f"Total : {vram_total:.2f} GB")

if vram_free < 3.0:
    print("⚠️  Less than 3GB free -- monitor carefully during backprop")
else:
    print("✅ Sufficient VRAM for TTT backpropagation")

# --- 14.7 Save LoRA Configuration ---
lora_config = {
    "rank":             LORA_RANK,
    "alpha":            LORA_ALPHA,
    "scale":            LORA_ALPHA / LORA_RANK,
    "target_modules":   ["query", "value"],
    "target_scope":     "model.encoder + model.decoder only",
    "n_injected":       n_injected,
    "n_lora_params":    len(lora_params),
    "trainable_params": trainable_params,
    "total_params":     total_params,
    "trainable_ratio":  round(
        100 * trainable_params / total_params, 6
    ),
    "reference":        "Hu et al. (2021) LoRA"
}

lora_path = os.path.join(
    EVAL_PARAMS["save_dir"], "lora_config.json"
)
with open(lora_path, "w") as f:
    json.dump(lora_config, f, indent=2)

print(f"\n✅ LoRA config saved: {lora_path}")
print("\n" + "="*50)
print("BLOCK 14 COMPLETE -- LoRA rank=4 injection verified")
print("="*50)
