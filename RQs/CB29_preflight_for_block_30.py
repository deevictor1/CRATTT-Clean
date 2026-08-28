# ============================================================
# BLOCK 29: Pre-flight check + backup archive
# before Stratified Reliability Analysis
# ============================================================
checks = {
    "dino_model":                 'dino_model' in dir(),
    "dino_processor":              'dino_processor' in dir(),
    "B24 dict":                    'B24' in dir(),
    "B24['snr_baseline'] set":     'B24' in dir() and 'snr_baseline' in B24,
    "B24['tau_internal'] set":     'B24' in dir() and 'tau_internal' in B24,
    "b24_compute_snr":             'b24_compute_snr' in dir(),
    "apply_corruption_deterministic": 'apply_corruption_deterministic' in dir(),
    "reset_lora_b":                'reset_lora_b' in dir(),
    "COCO_MAP":                    'COCO_MAP' in dir(),
    "coco_gt":                     'coco_gt' in dir(),
    "image_files >= 30":           'image_files' in dir() and len(image_files) >= 30,
    "loaded_images >= 30":         'loaded_images' in dir() and len(loaded_images) >= 30,
    "img_id_map >= 30":            'img_id_map' in dir() and len(img_id_map) >= 30,
}

all_ok = True
for name, status in checks.items():
    icon = "✅" if status else "❌"
    print(f"{icon} {name}")
    if not status:
        all_ok = False

print()
if all_ok:
    print(f"✅ image_files actual count: {len(image_files)}")
    print(f"✅ B24 snr_baseline={B24.get('snr_baseline'):.3f}  tau_internal={B24.get('tau_internal'):.4f}")
    # Sanity check: these should NOT be the un-calibrated raw values
    # (snr_baseline in the 100s, tau_internal > 1) — if they are,
    # the trimmed Block 24c calibration cell didn't run correctly.
    if B24.get('tau_internal', 0) > 1.5:
        print("⚠️  tau_internal looks like the UNCALIBRATED raw-scale value")
        print("    (should be < 1.5 after proper two-pass calibration) —")
        print("    re-run the trimmed Block 24c cell before proceeding.")
        all_ok = False
    print()
    print("✅ Ready to run Stratified Reliability Analysis" if all_ok else "❌ Fix the flagged item above first")
else:
    print("❌ Missing dependencies — re-run the flagged setup blocks")

# ─────────────────────────────────────────────────────────────
# Backup archive — the analysis below is the last, most
# expensive step before Phase 6 concludes, worth a checkpoint
# before running it.
# ─────────────────────────────────────────────────────────────
import shutil, os

os.makedirs('/kaggle/temp', exist_ok=True)

# Build the archive OUTSIDE /kaggle/working so it can never
# try to include itself while zipping the whole working directory.
shutil.make_archive('/kaggle/temp/full_checkpoint', 'zip', '/kaggle/working')

# Move the finished zip into /kaggle/working so it shows up
# in the Output panel for download.
shutil.move('/kaggle/temp/full_checkpoint.zip', '/kaggle/working/full_checkpoint.zip')

size_mb = os.path.getsize('/kaggle/working/full_checkpoint.zip') / (1024*1024)
print(f"✅ full_checkpoint.zip created — {size_mb:.1f} MB")
print("Download it now from the Output panel before continuing.")

from IPython.display import FileLink
display(FileLink('/kaggle/working/full_checkpoint.zip'))

# Confirm the file genuinely exists at the expected path
print(os.path.exists('/kaggle/working/full_checkpoint.zip'))
print(os.path.getsize('/kaggle/working/full_checkpoint.zip') / (1024*1024), "MB")
