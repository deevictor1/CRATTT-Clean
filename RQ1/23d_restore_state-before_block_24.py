# ============================================================
# Restore state before Block 24
#
# 1. Re-enable LoRA gradients: Block 23.1 froze ALL of
#    dino_model's parameters, including the lora_A/lora_B
#    tensors injected back in Block 22. The layers themselves
#    are untouched (still rank=4, 36 of them), only their
#    requires_grad flag was turned off. Re-enabling it here,
#    not re-injecting, since re-injection on an already-wrapped
#    model would silently match 0 layers (the same bug pattern
#    from 16c/17/18/20/21/22).
#
# 2. Restore CRATTT_PARAMS['tau'] to the confirmed Swin-B
#    calibration value (Block 10c original run:
#    rjoint_clean_mean=0.3263, rjoint_corrupt_mean=0.3158).
# ============================================================

# --- 1. Re-enable LoRA gradients ---
n_lora_reenabled = 0
for name, param in dino_model.named_parameters():
    if 'lora_A' in name or 'lora_B' in name:
        param.requires_grad = True
        n_lora_reenabled += 1

trainable = sum(p.numel() for p in dino_model.parameters() if p.requires_grad)
print(f"✅ Re-enabled grad on {n_lora_reenabled} LoRA tensors")
print(f"   Trainable params: {trainable:,} (should be 73,728)")
assert trainable == 73_728, f"Expected 73,728, got {trainable:,} — check for leftover state from another block"

# --- 2. Restore tau ---
rjoint_clean_mean   = 0.3263
rjoint_corrupt_mean = 0.3158

TAU_CALIBRATED = round((rjoint_clean_mean + rjoint_corrupt_mean) / 2, 3)
CRATTT_PARAMS["tau"] = TAU_CALIBRATED

print(f"✅ CRATTT_PARAMS['tau'] restored to {TAU_CALIBRATED}")
