# ============================================================
# BLOCK V2: Install Packages
# Run on: Vast.ai
# Run once per instance — takes 3-5 minutes
# Safe to rerun if needed
# Includes fixes for:
#   - NumPy 2.0 compatibility (fog corruption)
#   - scikit-image multichannel (glass_blur corruption)
# ============================================================

import subprocess, sys

packages = [
    "pycocotools",
    "imagecorruptions",
    "transformers>=4.37.0",
    "scikit-image",
    "ultralytics",
    "Pillow",
    "tqdm",
    "pandas",
    "numpy",
    "opencv-python",
    "wand",
]

print("Installing packages...")
print()

all_ok = True
for pkg in packages:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", pkg, "-q"],
        capture_output=True
    )
    if result.returncode != 0:
        print(f"  ❌ {pkg}")
        print(f"     {result.stderr.decode()[:100]}")
        all_ok = False
    else:
        print(f"  ✅ {pkg}")

print()
if all_ok:
    print("✅ All packages installed successfully")
else:
    print("⚠️  Some packages failed — check errors above")

# -------------------------------------------------------
# FIX 1: NumPy 2.0 compatibility patch
# Restores removed aliases needed by imagecorruptions
# Fixes: fog, and other corruptions using np.float_
# Must apply BEFORE importing imagecorruptions
# -------------------------------------------------------
import numpy as np
np.float_   = np.float64
np.int_     = np.int64
np.complex_ = np.complex128
np.bool_    = np.bool_
np.object_  = object
np.str_     = np.str_
print("✅ NumPy 2.0 aliases patched")

# -------------------------------------------------------
# FIX 2: scikit-image gaussian patch
# Removes deprecated 'multichannel' argument
# Fixes: glass_blur corruption
# Must apply BEFORE importing imagecorruptions
# -------------------------------------------------------
import skimage.filters as skfilters
from functools import wraps

original_gaussian = skfilters.gaussian

@wraps(original_gaussian)
def patched_gaussian(*args, **kwargs):
    kwargs.pop('multichannel', None)
    if 'channel_axis' not in kwargs:
        kwargs['channel_axis'] = -1
    return original_gaussian(*args, **kwargs)

skfilters.gaussian = patched_gaussian
print("✅ scikit-image gaussian patched for glass_blur")

# -------------------------------------------------------
# Verify both fixes work before proceeding
# -------------------------------------------------------
from imagecorruptions import corrupt as ic_corrupt
import numpy as np
from PIL import Image
import os

# Use a small test image
test_array = np.zeros((100, 100, 3), dtype=np.uint8)
test_array[:] = 128  # Grey image

fog_ok  = False
blur_ok = False

try:
    ic_corrupt(test_array, corruption_name='fog', severity=1)
    fog_ok = True
    print("✅ Fog corruption verified")
except Exception as e:
    print(f"⚠️  Fog test failed: {e}")

try:
    ic_corrupt(test_array, corruption_name='glass_blur', severity=1)
    blur_ok = True
    print("✅ glass_blur corruption verified")
except Exception as e:
    print(f"⚠️  glass_blur test failed: {e}")

print()
if fog_ok and blur_ok:
    print("✅ All fixes verified — ready to proceed")
elif fog_ok:
    print("✅ Fog fixed — glass_blur still needs attention")
else:
    print("⚠️  Some fixes failed — check errors above")
