# --- Block 22b note: Relaxed Parameters (tau fixed) ---
print("Block 22b: Relaxed consistency thresholds")
print("Consistency: 2/5 views (was 3/5)")
print("IoU thresh : 0.3 (was 0.5)")
print()
print("Pseudo-GT counts at relaxed params (5 corrupted images):")

corruption = 'snow'
severity   = 5

for img_path in image_files[:5]:
    raw_img = loaded_images[img_path]
    c_img   = ic_corrupt(raw_img, corruption_name=corruption, severity=severity)

    consistent_boxes, consistent_labels, c_scores = find_consistent_boxes(
        c_img, n_views=5, consistency_threshold=2, iou_threshold=0.3
    )
    verified_boxes, verified_labels = oracle_verify_consistent_boxes(
        c_img, consistent_boxes, consistent_labels
    )

    fname = os.path.basename(img_path)
    print(f"  {fname}: consistent={len(consistent_boxes)} "
          f"oracle_verified={len(verified_boxes)} labels={verified_labels[:3]}")

# For comparison: what did the STRICT params (Block 22) actually get on these
# same 5 images? Worth knowing before assuming relaxation is necessary.
print("\nFor comparison — strict params (Block 22's actual settings) on the same images:")
for img_path in image_files[:5]:
    raw_img = loaded_images[img_path]
    c_img   = ic_corrupt(raw_img, corruption_name=corruption, severity=severity)
    cb, cl, _ = find_consistent_boxes(c_img, n_views=5, consistency_threshold=3, iou_threshold=0.5)
    vb, vl = oracle_verify_consistent_boxes(c_img, cb, cl)
    fname = os.path.basename(img_path)
    print(f"  {fname}: consistent={len(cb)} oracle_verified={len(vb)}")
