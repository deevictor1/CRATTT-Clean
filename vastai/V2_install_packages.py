# ============================================================
# BLOCK V2: Install Packages
# Run on: Vast.ai
# Run once per instance — takes 3-5 minutes
# Safe to rerun if needed
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
