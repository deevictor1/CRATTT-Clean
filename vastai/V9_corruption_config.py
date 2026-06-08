# ============================================================
# BLOCK V9: Corruption Configuration
# Run on: Vast.ai
# Selects corruptions for this session based on Block V1
# ============================================================

ALL_CORRUPTIONS = {
    "Weather": [
        ("Weather", "snow"),
        ("Weather", "frost"),
        ("Weather", "fog"),
        ("Weather", "brightness"),
    ],
    "Digital": [
        ("Digital", "contrast"),
        ("Digital", "elastic_transform"),
        ("Digital", "pixelate"),
        ("Digital", "jpeg_compression"),
    ],
    "Noise": [
        ("Noise", "gaussian_noise"),
        ("Noise", "shot_noise"),
        ("Noise", "impulse_noise"),
    ],
    "Blur": [
        ("Blur", "defocus_blur"),
        ("Blur", "glass_blur"),
        ("Blur", "motion_blur"),
        ("Blur", "zoom_blur"),
    ],
}

SEVERITIES = [1, 2, 3, 4, 5]

selected_categories = CORRUPTIONS.split("+")
THIS_RUN = []
for cat in selected_categories:
    cat = cat.strip()
    if cat in ALL_CORRUPTIONS:
        THIS_RUN.extend(ALL_CORRUPTIONS[cat])
    else:
        print(f"⚠️  Unknown category: {cat}")

total_configs = len(THIS_RUN) * len(SEVERITIES)

print(f"Session         : {RUN_NAME}")
print(f"Model           : {MODEL.upper()}")
print(f"Corruptions     : {[c[1] for c in THIS_RUN]}")
print(f"Severities      : {SEVERITIES}")
print(f"Images          : {len(image_files)}")
print(f"Total configs   : {total_configs}")
print()

completed = 0
remaining = []
for cat, corruption in THIS_RUN:
    for severity in SEVERITIES:
        ckpt = load_ckpt(MODEL, corruption, severity)
        if ckpt is not None and ckpt.get('n_images', 0) > 0:
            completed += 1
        else:
            remaining.append((cat, corruption, severity))

print(f"Already done    : {completed}/{total_configs}")
print(f"Still to run    : {len(remaining)}/{total_configs}")

if completed == total_configs:
    print()
    print("✅ All configurations already checkpointed")
    print("   Skip to Block V11 for summary")
