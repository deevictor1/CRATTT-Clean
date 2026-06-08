# ============================================================
# BLOCK V12: Archive and Download
# Run on: Vast.ai — ALWAYS run before destroying instance
# ============================================================

import shutil, os, zipfile

print("Creating download archive...")

archive_name = f"/workspace/{RUN_NAME}_results"
shutil.make_archive(
    archive_name,
    'zip',
    '/workspace',
    'crattt'
)

size_mb = os.path.getsize(archive_name + '.zip') / 1e6
print(f"✅ Archive: {archive_name}.zip ({size_mb:.2f} MB)")

# Verify ZIP contents
with zipfile.ZipFile(archive_name + '.zip', 'r') as z:
    files     = z.namelist()
    json_files = [f for f in files if f.endswith('.json')]
    csv_files  = [f for f in files if f.endswith('.csv')]

print(f"   JSON checkpoints : {len(json_files)}")
print(f"   CSV result files : {len(csv_files)}")
print()

if len(json_files) == 0:
    print("⚠️  No JSON files found in archive")
    print("   Check /workspace/crattt/checkpoints/ manually")
else:
    print("✅ Archive verified — safe to download and destroy")

print()
print("=" * 55)
print("DOWNLOAD INSTRUCTIONS")
print("=" * 55)
print()
print("In Jupyter file browser:")
print("  1. Navigate to /workspace/")
print(f"  2. Right-click {RUN_NAME}_results.zip")
print("  3. Click Download")
print("  4. Wait for download to complete fully")
print()
print("AFTER DOWNLOADING — DESTROY YOUR INSTANCE")
print("  Vast.ai dashboard → trash icon → confirm")
print("  Billing stops immediately")
print()
print("Next session — upload this ZIP before Block V10")
