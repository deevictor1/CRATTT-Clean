# ============================================================
# Revert to Block 14's rank=4 LoRA before continuing Section A.
# Block 17 onward (co-adaptation, encoder gate, detection-head
# LoRA) all expect the rank=4 starting state from Block 14.
# ============================================================
import gc

print("Reverting to rank=4 LoRA (Block 14 configuration)...")
del dino_model
gc.collect()
torch.cuda.empty_cache()
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
    DINO_MODEL_ID, token=hf_token
).to(device)
dino_model.eval()

n_injected, lora_params = inject_lora_grounding_dino(
    dino_model, rank=LORA_RANK, lora_alpha=LORA_ALPHA
)
trainable_params = sum(p.numel() for p in dino_model.parameters() if p.requires_grad)
print(f"✅ LoRA layers injected : {n_injected} (should be 36)")
print(f"✅ Trainable parameters : {trainable_params:,} (should be 73,728)")
assert n_injected == 36
assert trainable_params == 73_728
print("✅ Back to rank=4 — safe to continue to Block 17")
