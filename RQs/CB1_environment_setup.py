
# ============================================================
# BLOCK 1: Environment Setup
# CRATTT Clean Implementation
# ============================================================

# --- 1.1 Install Required Libraries ---
!pip install imagecorruptions -q
!pip install wandb -q
!pip install pycocotools -q
!pip install ultralytics -q

# --- 1.2 Core Imports ---
import os
import sys
import random
import json
import glob
import numpy as np
import torch
import skimage
import skimage.filters
import imagecorruptions.corruptions as cor_mod
import imagecorruptions
import wandb
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login

# --- 1.3 Reproducibility: Fix All Seeds ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
print(f"✅ Seeds fixed: {SEED}")

# --- 1.4 Surgical Patch: NumPy 2.0 + scikit-image compatibility ---
# imagecorruptions uses the deprecated 'multichannel' kwarg removed in
# scikit-image 0.19+. This patch redirects it to the new 'channel_axis'.
if not hasattr(skimage.filters.gaussian, '_is_patched'):
    _real_gaussian = skimage.filters.gaussian

    def _patched_gaussian(*args, **kwargs):
        if 'multichannel' in kwargs:
            val = kwargs.pop('multichannel')
            kwargs['channel_axis'] = -1 if val else None
        return _real_gaussian(*args, **kwargs)

    _patched_gaussian._is_patched = True
    skimage.filters.gaussian = _patched_gaussian
    cor_mod.gaussian = _patched_gaussian
    print("✅ scikit-image patch applied")
else:
    print("✅ scikit-image patch already active")

# --- 1.5 Log Library Versions for Reproducibility ---
print("\n--- Library Versions ---")
print(f"Python:            {sys.version.split()[0]}")
print(f"PyTorch:           {torch.__version__}")
print(f"NumPy:             {np.__version__}")
print(f"scikit-image:      {skimage.__version__}")

import importlib.metadata
try:
    ic_version = importlib.metadata.version("imagecorruptions")
except importlib.metadata.PackageNotFoundError:
    ic_version = "installed (version unknown)"
print(f"imagecorruptions:  {ic_version}")

# --- 1.6 Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n✅ Device: {device}")
if device.type == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# --- 1.7 Secure API Connections ---
user_secrets = UserSecretsClient()
print("\n--- API Connections ---")

try:
    wb_key = user_secrets.get_secret("WANDB_API_KEY")
    wandb.login(key=wb_key, relogin=True)
    print("✅ W&B: Connected")
except Exception as e:
    print(f"⚠️  W&B: Not connected — {e}")

try:
    hf_token = user_secrets.get_secret("HF_TOKEN")
    login(token=hf_token)
    print("✅ HuggingFace: Connected")
except Exception as e:
    print(f"❌ HuggingFace: Failed — {e}")

# --- 1.8 Create Output Directories ---
DIRS = {
    "results":  "/kaggle/working/results",
    "figures":  "/kaggle/working/figures",
    "checkpoints": "/kaggle/working/checkpoints",
    "tables":   "/kaggle/working/tables"
}
for name, path in DIRS.items():
    os.makedirs(path, exist_ok=True)

print(f"\n✅ Output directories created:")
for name, path in DIRS.items():
    print(f"   {name}: {path}")

print("\n" + "="*50)
print("BLOCK 1 COMPLETE — Environment ready")
print("="*50)
