# CRATTT Vast.ai Sweep — Complete Code Blocks

## Platform
- Blocks V1–V13, including V7b: run on Vast.ai (fresh instance each session)
- Block V13: run on Kaggle after all sessions complete

## Session Order
| Session | Block V1 Settings | Model | Corruptions |
|---------|------------------|-------|-------------|
| 1 | MODEL="dino", CORRUPTIONS="Weather+Digital" | DINO | snow, frost, fog, brightness, contrast, elastic_transform, pixelate, jpeg_compression |
| 2 | MODEL="dino", CORRUPTIONS="Noise+Blur" | DINO | gaussian_noise, shot_noise, impulse_noise, defocus_blur, glass_blur, motion_blur, zoom_blur |
| 3 | MODEL="yolo", CORRUPTIONS="Weather+Digital" | YOLO | same as Session 1 |
| 4 | MODEL="yolo", CORRUPTIONS="Noise+Blur" | YOLO | same as Session 2 |

## Critical Notes
- DINO sessions load **GroundingDINO-base (Swin-B)**, not the tiny variant. `DINO_MODEL_ID` is set once in Block V1 and referenced in Block V5, do not hardcode a model ID elsewhere.
- Block V7b defines `run_dino_with_categories` and `category_token_spans`. It replaces the library's `post_process_grounded_object_detection`, which silently drops detections on composite phrases (e.g. "chair couch"). Block V10's DINO branch depends on this function; skipping V7b will crash V10 with a `NameError`.
- `CLEAN_MAP_DINO` (0.5281) and `CLEAN_MAP_YOLO` (0.4327) in Block V1 are the verified dissertation baselines (Table 4.1/4.2a). Every CE value Block V10 produces is normalised against these, do not edit them without re-checking against that table.
- Block V6 is the CORRECTED version — fixes BaseModelOutputWithPooling error
- Block V10 includes the fog fix — uint8 conversion before every corruption
- Block V9 skips checkpoints where n_images=0 (corrupted fog entries)
- Always upload previous session ZIP before running Block V10
- Always run Block V12 and download ZIP before destroying instance

## Checkpoint Upload (between V9 and V10)
```python
import zipfile, os
zip_path = "/workspace/dino_Weather_Digital_results_v2.zip"
with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall("/workspace")
print("✅ Checkpoints restored")
```

## Instance Requirements
- GPU: RTX 4090 (recommended)
- Disk: 40GB minimum
- RAM: 32GB minimum
- Template: PyTorch (Vast)
- Type: On-Demand
