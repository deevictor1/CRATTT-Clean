# ============================================================
# Post-injection verification for Block 14
# Confirms LoRA B-matrix count and trainable parameter count
# match the expected values.
# ============================================================
n = sum(1 for n, p in dino_model.named_parameters() if "lora_B" in n)
t = sum(p.numel() for p in dino_model.parameters() if p.requires_grad)
print(f"LoRA B matrices : {n}   (should be 36)")
print(f"Trainable params: {t:,}  (should be 73,728)")
