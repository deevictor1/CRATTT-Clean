# ============================================================
# BLOCK 24 — PRE-FLIGHT CHECK
# Run this immediately before Block 24.
# Block 24 does NOT need CLIP, compute_soracle,
# find_consistent_boxes, get_pseudo_gt, or map_clean_dino.
# It defines all its own helper functions internally.
# ============================================================

import torch

def _exists(name):
    """True if the variable exists in the global namespace."""
    import builtins
    return name in globals() or name in vars(builtins)

def _get(name, default=None):
    return globals().get(name, default)

# ── Required variables ────────────────────────────────────────
checks = {
    # Core model objects
    "dino_model":
        _exists("dino_model"),
    "dino_processor":
        _exists("dino_processor"),

    # LoRA injection: 36 layers, exactly 73,728 trainable params
    "lora_injected":
        _exists("dino_model") and
        sum(p.numel() for p in _get("dino_model").parameters()
            if p.requires_grad) == 73_728,

    # CRATTT config with correct tau
    "CRATTT_PARAMS":
        _exists("CRATTT_PARAMS") and
        isinstance(_get("CRATTT_PARAMS"), dict),
    "tau_is_0.321":
        _exists("CRATTT_PARAMS") and
        _get("CRATTT_PARAMS", {}).get("tau") == 0.321,
    "dino_text_thr_set":
        _exists("CRATTT_PARAMS") and
        "dino_text_thr" in _get("CRATTT_PARAMS", {}),

    # Text prompt
    "DINO_TEXT_PROMPT":
        _exists("DINO_TEXT_PROMPT") and
        isinstance(_get("DINO_TEXT_PROMPT"), str) and
        len(_get("DINO_TEXT_PROMPT", "")) > 0,

    # COCO class map (dict str → int)
    "COCO_MAP":
        _exists("COCO_MAP") and
        isinstance(_get("COCO_MAP"), dict) and
        len(_get("COCO_MAP", {})) > 0,

    # Image data
    "image_files_loaded":
        _exists("image_files") and
        len(_get("image_files", [])) >= 20,
    "loaded_images_dict":
        _exists("loaded_images") and
        len(_get("loaded_images", {})) >= 20,
    "img_id_map":
        _exists("img_id_map") and
        len(_get("img_id_map", {})) >= 20,

    # COCO ground truth + evaluation
    "coco_gt":
        _exists("coco_gt"),
    "compute_map":
        _exists("compute_map") and
        callable(_get("compute_map")),
}

# ── VRAM ──────────────────────────────────────────────────────
vram_total = torch.cuda.get_device_properties(0).total_memory
vram_used  = torch.cuda.memory_allocated()
vram_free  = (vram_total - vram_used) / 1e9
vram_ok    = vram_free > 8.0

# ── Print results ─────────────────────────────────────────────
print("=" * 55)
print("BLOCK 24 PRE-FLIGHT CHECK")
print("=" * 55)

MAX_LEN  = max(len(k) for k in checks)
all_ok   = True
failures = []

for name, status in checks.items():
    icon = "✅" if status else "❌"
    print(f"  {icon}  {name:<{MAX_LEN}}")
    if not status:
        all_ok = False
        failures.append(name)

print()
vram_icon = "✅" if vram_ok else "❌"
print(f"  {vram_icon}  {'VRAM free':<{MAX_LEN}}  {vram_free:.2f} GB "
      f"({'≥ 8 GB required' if vram_ok else '< 8 GB — clear memory first'})")
print()

# ── Final verdict ─────────────────────────────────────────────
if all_ok and vram_ok:
    print("✅  All checks passed — safe to run Block 24.")
else:
    print("❌  Pre-flight failed. Fix the items below before proceeding:\n")
    if not vram_ok:
        print("  • VRAM: run  torch.cuda.empty_cache()  or restart the session.")
    for f in failures:
        guide = {
            "dino_model":           "Re-run the GroundingDINO model-loading block.",
            "dino_processor":       "Re-run the GroundingDINO model-loading block.",
            "lora_injected":        "Re-run the LoRA injection block (should print '36 layers, 73,728 params').",
            "CRATTT_PARAMS":        "Re-run the CRATTT configuration / params block.",
            "tau_is_0.321":         "Re-run the CRATTT params block and verify tau=0.321.",
            "dino_text_thr_set":    "Re-run the CRATTT params block; ensure 'dino_text_thr' key exists.",
            "DINO_TEXT_PROMPT":     "Re-run the CRATTT params block; ensure DINO_TEXT_PROMPT is defined.",
            "COCO_MAP":             "Re-run the COCO class mapping block (defines str→int category dict).",
            "image_files_loaded":   "Re-run the image loading block; need ≥ 20 entries in image_files.",
            "loaded_images_dict":   "Re-run the image loading block; need ≥ 20 entries in loaded_images.",
            "img_id_map":           "Re-run the image loading block; need ≥ 20 entries in img_id_map.",
            "coco_gt":              "Re-run the COCO ground-truth loading block.",
            "compute_map":          "Re-run the evaluation utilities block (defines compute_map function).",
        }.get(f, "Re-run the relevant setup block.")
        print(f"  • {f}: {guide}")

print()
print("=" * 55)
