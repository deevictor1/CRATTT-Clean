# ============================================================
# Expand image pool to 50, then confirm readiness for Block 24e
# ============================================================

# --- 1. Expand image pool to 50 before Block 24e ---
TARGET_N = 50

if len(image_files) < TARGET_N:
    IMAGE_DIR_CURRENT = os.path.dirname(image_files[0])
    all_candidates = sorted(f for f in os.listdir(IMAGE_DIR_CURRENT) if f.lower().endswith(".jpg"))
    existing_basenames = {os.path.basename(p) for p in image_files}
    added = 0
    for fname in all_candidates:
        if len(image_files) >= TARGET_N:
            break
        if fname in existing_basenames:
            continue
        full_path = os.path.join(IMAGE_DIR_CURRENT, fname)
        try:
            candidate_img_id = int(os.path.splitext(fname)[0])
        except ValueError:
            continue
        if len(coco_gt.getAnnIds(imgIds=[candidate_img_id])) == 0:
            continue
        img_arr = np.array(PILImage.open(full_path).convert("RGB"))
        loaded_images[full_path] = img_arr
        img_id_map[fname] = candidate_img_id
        image_files.append(full_path)
        added += 1
    print(f"✅ Added {added} images -- pool now at {len(image_files)}")
else:
    print(f"✅ Pool already at {len(image_files)}, no expansion needed")

print()

# --- 2. Readiness check for Block 24e ---
checks = {
    "dino_model trainable == 73,728": sum(p.numel() for p in dino_model.parameters() if p.requires_grad) == 73_728,
    "B24 calibrated (tau<1.5)":        'B24' in dir() and B24.get('tau_internal', 99) < 1.5,
    "b24_compute_snr":                 'b24_compute_snr' in dir(),
    "b24_apply_gate":                  'b24_apply_gate' in dir(),
    "b24_confidence_loss":             'b24_confidence_loss' in dir(),
    "b24_run_dino":                    'b24_run_dino' in dir(),
    "reset_lora_b":                    'reset_lora_b' in dir(),
    "apply_corruption_deterministic":  'apply_corruption_deterministic' in dir(),
    "compute_map":                     'compute_map' in dir(),
    "coco_gt":                         'coco_gt' in dir(),
    "COCO_MAP":                        'COCO_MAP' in dir(),
    "device":                          'device' in dir(),
    "loaded_images":                   'loaded_images' in dir(),
    "img_id_map":                      'img_id_map' in dir(),
    "image_files >= 50":               'image_files' in dir() and len(image_files) >= 50,
}
all_ok = True
for name, status in checks.items():
    print(f"{'✅' if status else '❌'} {name}")
    if not status: all_ok = False
print()
print("✅ Ready for Block 24e" if all_ok else "❌ Fix flagged items first")
