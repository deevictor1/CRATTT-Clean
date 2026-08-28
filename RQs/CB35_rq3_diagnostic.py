"""
RQ3 DIAGNOSTIC — Adaptation Speed Analysis (Swin-B)
Adapted from cells 42 (LoRA vs full fine-tune timing) and 33
(rank=16 LoRA injection) in crattt-clean-2.ipynb.

BUG FIXED from the historical source: cell 33's rank=16 cell had
LORA_RANK=4 and LORA_ALPHA=8 left over from a copy-paste of the
rank=4 cell -- every comment in it describes rank=16/alpha=32,
but the actual values were never updated. Corrected here to
LORA_RANK=16, LORA_ALPHA=32, matching the surrounding comments
and the confirmed real rank=16 result already used elsewhere
in this chapter (Table 4.8, Block 16c).

STRUCTURE:
  Pre-flight 1 -- confirm rank=4 LoRA is genuinely active (73,728
                  trainable params). v2 FIX: this now actually
                  INJECTS rank=4 LoRA if it isn't already present,
                  rather than just warning -- the consolidated
                  recovery cell used for Phase 6 work deliberately
                  never injects LoRA (Phase 6 is inference-only),
                  so it can't be assumed active. v1 of this script
                  only warned here, which let Arm A proceed to time
                  a backward pass through the ENTIRE unfrozen 232M+
                  parameter model, causing an immediate OOM before
                  ever reaching the full-fine-tune arm's protected
                  try/except block.
  Arm A        -- time 5 adaptation steps at rank=4, then attempt
                  full fine-tuning (OOM expected but handled).
  Pre-flight 2 -- confirm rank=4 state was correctly restored
                  after Arm A's full-fine-tune attempt.
  Reload       -- fresh, unmodified dino_model (LoRA injection
                  can't be layered a second time onto the same
                  instance -- it only converts plain nn.Linear
                  layers, which none remain after rank=4's pass).
  Pre-flight 3 -- confirm the reload is genuinely LoRA-free.
  Rank=16 inject + Arm B -- inject corrected rank=16 LoRA, verify
                  param count, time 5 adaptation steps.
  Final summary -- prints the complete Table 4.17 numbers.
"""

# ====================================================================
# PRE-FLIGHT 1 -- confirm rank=4 LoRA is genuinely active
# ====================================================================
import numpy as np  # explicit here, since Arm A uses np.mean/np.std
                     # before Arm B's own numpy import further down
import torch.nn as nn
import math

required = ["dino_model", "dino_processor", "DINO_TEXT_PROMPT", "device",
            "image_files", "loaded_images", "EVAL_PARAMS", "CRATTT_PARAMS", "hf_token"]
missing = [f for f in required if f not in globals()]
if missing:
    raise RuntimeError(f"Missing: {missing} -- run the consolidated recovery cell first.")

# ── LoRA injection helpers, defined ONCE, reused for both rank=4 (here,
#    if not already injected) and rank=16 (later, on the fresh reload) ──
class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank=4, lora_alpha=8):
        super().__init__()
        self.in_features  = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rank         = rank
        self.scale        = lora_alpha / rank
        self.weight = nn.Parameter(linear_layer.weight.data.clone(), requires_grad=False)
        self.bias = nn.Parameter(linear_layer.bias.data.clone(), requires_grad=False) \
            if linear_layer.bias is not None else None
        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base  = nn.functional.linear(x, self.weight, self.bias)
        delta = (x @ self.lora_A.T @ self.lora_B.T) * self.scale
        return base + delta

def inject_lora_grounding_dino(model, rank=4, lora_alpha=8):
    n_injected = 0
    for name, module in list(model.named_modules()):
        if not (name.startswith('model.encoder') or name.startswith('model.decoder')):
            continue
        if not (name.endswith('.query') or name.endswith('.value')):
            continue
        if not isinstance(module, nn.Linear):
            continue
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        lora_layer = LoRALinear(module, rank=rank, lora_alpha=lora_alpha).to(device)
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

print("=" * 65)
print("PRE-FLIGHT 1 -- confirm (or establish) rank=4 LoRA")
print("=" * 65)
_trainable = sum(p.numel() for p in dino_model.parameters() if p.requires_grad)
print(f"Trainable params right now: {_trainable:,}")

if _trainable == 73728:
    print("Confirmed: rank=4 LoRA is already active (73,728 trainable params)")
else:
    print(f"Rank=4 LoRA is NOT active (found {_trainable:,} trainable params, "
          f"expected 73,728) -- injecting it now rather than assuming it exists.")
    n_inj, lora_params_r4 = inject_lora_grounding_dino(dino_model, rank=4, lora_alpha=8)
    _trainable_after_inject = sum(p.numel() for p in dino_model.parameters() if p.requires_grad)
    print(f"✅ LoRA rank=4 injected: {n_inj} layers, {_trainable_after_inject:,} trainable params")
    assert _trainable_after_inject == 73728, (
        f"Injected rank=4 LoRA but got {_trainable_after_inject:,} trainable params, "
        f"not the expected 73,728 -- stopping rather than trusting Arm A's timing."
    )
    print("Confirmed: rank=4 LoRA now genuinely active")

# ====================================================================
# ARM A: rank=4 LoRA timing + full fine-tune attempt (verbatim)
# ====================================================================
import time
import torch

def time_adaptation_step(image_np, params_to_train, n_steps=5, lr=1e-4, n_repeats=10, warmup=2):
    inputs = dino_processor(images=image_np, text=DINO_TEXT_PROMPT, return_tensors="pt").to(device)
    opt = torch.optim.AdamW(params_to_train, lr=lr)

    def run_steps():
        dino_model.train()
        for _ in range(n_steps):
            opt.zero_grad()
            outputs = dino_model(**inputs)
            loss = -outputs.logits.sigmoid().max(dim=-1).values.mean()  # timing proxy loss only
            loss.backward()
            opt.step()
        dino_model.eval()

    for _ in range(warmup):
        run_steps()
    torch.cuda.synchronize()

    times = []
    for _ in range(n_repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_steps()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return times

print("=" * 65)
print("RQ3: Adaptation Speed — LoRA TTT vs Full Fine-Tuning")
print("=" * 65)
print()

test_img = loaded_images[image_files[0]]

# ── Arm 1: LoRA-only (current state — already the only trainable params) ──
lora_params = [p for p in dino_model.parameters() if p.requires_grad]
n_lora = sum(p.numel() for p in lora_params)
print(f"LoRA trainable params: {n_lora:,}")

lora_times = time_adaptation_step(test_img, lora_params)
print(f"LoRA TTT — mean per-image (5 steps): {np.mean(lora_times)*1000:.1f}ms "
      f"(±{np.std(lora_times)*1000:.1f}ms), per-step: {np.mean(lora_times)/5*1000:.1f}ms")
print()

# ── Arm 2: Full fine-tuning — temporarily unfreeze everything ──
original_grad_state = {name: p.requires_grad for name, p in dino_model.named_parameters()}

try:
    for p in dino_model.parameters():
        p.requires_grad = True
    full_params = list(dino_model.parameters())
    n_full = sum(p.numel() for p in full_params)
    print(f"Full fine-tune trainable params: {n_full:,}  ({n_full/n_lora:.0f}x more than LoRA)")

    full_times = time_adaptation_step(test_img, full_params)
    print(f"Full FT — mean per-image (5 steps): {np.mean(full_times)*1000:.1f}ms "
          f"(±{np.std(full_times)*1000:.1f}ms), per-step: {np.mean(full_times)/5*1000:.1f}ms")

    speedup = np.mean(full_times) / np.mean(lora_times)
    print(f"\nSpeedup factor (LoRA vs full FT): {speedup:.2f}x")

except torch.cuda.OutOfMemoryError:
    print("⚠️  Full fine-tuning OOM'd on this GPU — cannot complete this comparison arm as designed.")
    print("   This itself is a relevant finding: full fine-tuning may be infeasible on")
    print("   deployment-class hardware (T4, 15.6GB) where LoRA succeeds.")
    torch.cuda.empty_cache()

finally:
    # Restore exact original frozen state
    for name, p in dino_model.named_parameters():
        p.requires_grad = original_grad_state[name]
    print("\n✅ Original frozen/LoRA state restored")

print()
print("─" * 65)
print("Reference: 30 FPS real-time benchmark = 33.3ms/frame")
print("─" * 65)

# ====================================================================
# PRE-FLIGHT 2 -- confirm rank=4 state was restored after Arm A
# ====================================================================
_trainable_after_a = sum(p.numel() for p in dino_model.parameters() if p.requires_grad)
print(f"Trainable params after Arm A: {_trainable_after_a:,}")
assert _trainable_after_a == 73728, (
    f"Expected rank=4 state (73,728) to be restored after Arm A's finally-block, "
    f"got {_trainable_after_a:,} instead -- do not trust downstream results without investigating."
)
print("Confirmed: rank=4 state correctly restored after Arm A")

# ====================================================================
# RELOAD: fresh, unmodified GroundingDINO for the rank=16 arm
# (LoRA injection only converts plain nn.Linear layers, so it
#  cannot be layered a second time onto the already-converted
#  rank=4 instance -- a genuinely fresh model is required)
# ====================================================================
import gc
del dino_model
gc.collect()
torch.cuda.empty_cache()

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
DINO_MODEL_ID = "IDEA-Research/grounding-dino-base"
dino_processor = AutoProcessor.from_pretrained(DINO_MODEL_ID, token=hf_token)
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
    DINO_MODEL_ID, token=hf_token
).to(device)
dino_model.eval()

dino_params_fresh = sum(p.numel() for p in dino_model.parameters())
print(f"✅ Fresh GroundingDINO-base reloaded -- {dino_params_fresh:,} params")

# ====================================================================
# PRE-FLIGHT 3 -- confirm the reload is genuinely LoRA-free
# ====================================================================
_trainable_fresh = sum(p.numel() for p in dino_model.parameters() if p.requires_grad)
print(f"Trainable params on fresh reload: {_trainable_fresh:,}")
assert _trainable_fresh == dino_params_fresh, (
    "Expected ALL parameters trainable by default on a fresh model load "
    "(no LoRA injected yet) -- something is already frozen/modified unexpectedly."
)
print("Confirmed: fresh model has no LoRA layers, all parameters are plain nn.Linear")

# ====================================================================
# RANK=16 LoRA injection (bug-fixed version of cell 33 --
# see docstring above for what was wrong and why)
# ====================================================================
# ============================================================
# BLOCK 14: LoRA Adapter Injection — RANK 16 (MODIFIED)
# Injects trainable LoRA matrices into GroundingDINO's
# cross-modal neck (encoder + decoder query/value projections).
# All backbone weights remain frozen.
# Only LoRA matrices A and B are trainable.
# Reference: Hu et al. (2021) "LoRA: Low-Rank Adaptation"
# ============================================================
#
# WHY RANK=16 INSTEAD OF RANK=4
# ──────────────────────────────
# Block 24e (N=200, formally powered) showed vs_baseline = -0.0055
# (p<0.0001) with rank=4. Subsequent diagnostic (Block 24f) showed
# that switching from confidence-only to a full detection loss
# (L1 + GIoU + classification) produced IDENTICAL results —
# bit-for-bit, to 4 decimal places — confirming the loss function
# is NOT the bottleneck. The consistent LoRA Δ ≈ 0.0005 across all
# variants points to adaptation CAPACITY as the limiting factor:
# rank=4 (73,728 params) compresses all gradient signals —
# regardless of their source — into such a tiny weight perturbation
# that the model's predictions barely shift from the frozen baseline.
#
# RANK=16 gives 4× more capacity (≈294,912 params) without jumping
# to rank=32 (which risks overfitting to the sparse 1-2 verified
# detections per image). LORA_ALPHA is scaled proportionally
# (8→32) to maintain the same effective scale ratio (alpha/rank=2.0)
# as the original design — keeping the contribution magnitude
# consistent with the original intent.
#
# ONLY TWO LINES CHANGED vs the original Block 14:
#   LORA_RANK  : 4  → 16
#   LORA_ALPHA : 8  → 32
# Everything else is identical.
# ============================================================

# NOTE: LoRALinear and inject_lora_grounding_dino are already defined
# above (Pre-flight 1) and reused here unchanged -- not redefined.

# --- 14.3 Inject ---
LORA_RANK  = 16   # FIXED: historical source left this at 4 (copy-paste bug); corrected to 16
LORA_ALPHA = 32   # FIXED: historical source left this at 8 (copy-paste bug); corrected to 32

print("Injecting LoRA adapters into GroundingDINO...")
print(f"Rank      : {LORA_RANK}  (was 4)")
print(f"Alpha     : {LORA_ALPHA}  (was 8)")
print(f"Scale     : {LORA_ALPHA / LORA_RANK:.2f}  (unchanged — same as rank=4 design)")
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
print(f"vs rank=4 baseline  : {trainable_params / 73728:.1f}× more capacity")

assert trainable_params > 0, \
    "No trainable parameters — injection failed"
assert trainable_params < total_params * 0.05, \
    "Too many trainable params — check injection scope"
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

print(f"✅ Forward pass OK — {len(test_res['boxes'])} detections")
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
    print("⚠️  Less than 3GB free — monitor carefully during backprop")
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
    "rank4_baseline_params": 73728,
    "capacity_multiplier": round(trainable_params / 73728, 1),
    "reference":        "Hu et al. (2021) LoRA"
}

lora_path = os.path.join(
    EVAL_PARAMS["save_dir"], "lora_config_rank16.json"
)
with open(lora_path, "w") as f:
    json.dump(lora_config, f, indent=2)

print(f"\n✅ LoRA config saved: {lora_path}")
print("\n" + "="*50)
print("BLOCK 14 COMPLETE — LoRA rank=4 injection verified")
print("="*50)

# ====================================================================
# PRE-FLIGHT 4 -- confirm rank=16 LoRA is genuinely active
# ====================================================================
_trainable_r16 = sum(p.numel() for p in dino_model.parameters() if p.requires_grad)
print(f"Trainable params after rank=16 injection: {_trainable_r16:,}")
assert _trainable_r16 == 294912, (
    f"Expected 294,912 trainable params for rank=16, got {_trainable_r16:,} instead -- "
    f"do not trust Arm B's timing without investigating."
)
print("Confirmed: rank=16 LoRA is active (294,912 trainable params)")

# ====================================================================
# ARM B: rank=16 LoRA timing only (full fine-tune result from
# Arm A already covers that arm -- rank doesn't change whether
# full fine-tuning OOMs, since it's independent of LoRA entirely)
# ====================================================================
import time
import numpy as np

def time_adaptation_step(image_np, params_to_train, n_steps=5, lr=1e-4, n_repeats=10, warmup=2):
    inputs = dino_processor(images=image_np, text=DINO_TEXT_PROMPT, return_tensors="pt").to(device)
    opt = torch.optim.AdamW(params_to_train, lr=lr)

    def run_steps():
        dino_model.train()
        for _ in range(n_steps):
            opt.zero_grad()
            outputs = dino_model(**inputs)
            loss = -outputs.logits.sigmoid().max(dim=-1).values.mean()
            loss.backward()
            opt.step()
        dino_model.eval()

    for _ in range(warmup):
        run_steps()
    torch.cuda.synchronize()

    times = []
    for _ in range(n_repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_steps()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return times

print("=" * 65)
print("ARM B: LoRA rank=16 timing")
print("=" * 65)

test_img = loaded_images[image_files[0]]
lora_params_r16 = [p for p in dino_model.parameters() if p.requires_grad]
n_lora_r16 = sum(p.numel() for p in lora_params_r16)
print(f"LoRA trainable params: {n_lora_r16:,}  (rank=16)")

lora_times_r16 = time_adaptation_step(test_img, lora_params_r16)
print(f"LoRA TTT (rank=16) — mean per-image (5 steps): {np.mean(lora_times_r16)*1000:.1f}ms "
      f"(±{np.std(lora_times_r16)*1000:.1f}ms), per-step: {np.mean(lora_times_r16)/5*1000:.1f}ms")

# ====================================================================
# FINAL SUMMARY -- Table 4.17 numbers, all in one place
# ====================================================================
print()
print("=" * 65)
print("TABLE 4.17 SUMMARY")
print("=" * 65)
print(f"LoRA rank=4  : {73728:,} trainable params, "
      f"{np.mean(lora_times)*1000:.1f}ms total (5 steps), "
      f"{np.mean(lora_times)/5*1000:.1f}ms/step")
print(f"LoRA rank=16 : {n_lora_r16:,} trainable params, "
      f"{np.mean(lora_times_r16)*1000:.1f}ms total (5 steps), "
      f"{np.mean(lora_times_r16)/5*1000:.1f}ms/step")
print(f"Full fine-tune: {dino_params_fresh:,} trainable params -- "
      f"see Arm A output above for OOM status or timing")
