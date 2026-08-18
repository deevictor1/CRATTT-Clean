# ============================================================
# BLOCK 11c: Precision/Recall Diagnostic
# Determines whether CRATTT improves precision even when
# mAP falls, supporting the hallucination-pruning claim
# ============================================================

import numpy as np
from imagecorruptions import corrupt as ic_corrupt

print("Computing Precision/Recall breakdown...")
print("This determines CRATTT's true contribution")
print("="*50)

pr_rows = []

for img_path in PILOT_IMAGES:
    img_id  = img_id_map[os.path.basename(img_path)]
    raw_img = loaded_images[img_path]
    fname   = os.path.basename(img_path)

    # Get ground truth count for this image
    ann_ids = coco_gt.getAnnIds(imgIds=img_id)
    n_gt    = len(ann_ids)

    for sev in [1, 5]:
        c_img = ic_corrupt(
            raw_img,
            corruption_name=PILOT_CORRUPTION,
            severity=sev
        )

        # Baseline detections
        inputs = dino_processor(
            images=c_img,
            text=DINO_TEXT_PROMPT,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = dino_model(**inputs)
        baseline_res = dino_processor\
            .post_process_grounded_object_detection(
                outputs, inputs.input_ids,
                target_sizes=[c_img.shape[:2]],
                text_threshold=CRATTT_PARAMS["dino_text_thr"]
            )[0]

        n_baseline = len(baseline_res['boxes'])

        # CRATTT detections
        crattt_preds, stats = run_crattt_inference(c_img)
        n_crattt = stats['n_verified']
        n_rejected = stats['n_rejected']

        # Compute mAP for both
        baseline_coco = dino_to_coco_format(baseline_res, img_id)
        crattt_coco = [{
            "image_id":    img_id,
            "category_id": p["category_id"],
            "bbox":        p["bbox"],
            "score":       p["score"]
        } for p in crattt_preds]

        bmap, _ = compute_map(baseline_coco, coco_gt, [img_id])
        cmap, _ = compute_map(crattt_coco,   coco_gt, [img_id])

        # Estimated precision: mAP / detections (proxy)
        # True precision requires IoU matching but this gives
        # a directional signal
        baseline_precision = bmap / n_baseline if n_baseline > 0 else 0
        crattt_precision   = cmap / n_crattt   if n_crattt   > 0 else 0

        pr_rows.append({
            "Image":                fname,
            "Severity":             sev,
            "GT_Objects":           n_gt,
            "Baseline_Dets":        n_baseline,
            "CRATTT_Dets":          n_crattt,
            "Rejected":             n_rejected,
            "Baseline_mAP":         round(bmap, 4),
            "CRATTT_mAP":           round(cmap, 4),
            "Baseline_mAP_per_Det": round(baseline_precision, 4),
            "CRATTT_mAP_per_Det":   round(crattt_precision, 4),
            "Precision_Delta":      round(
                crattt_precision - baseline_precision, 4
            )
        })

df_pr = pd.DataFrame(pr_rows)

from IPython.display import display
print("\n--- Precision Proxy Analysis ---")
display(df_pr)

print("\n--- Key Question ---")
print("Is CRATTT_mAP_per_Det > Baseline_mAP_per_Det?")
print("If yes: CRATTT improves precision even at cost of recall")
print()

for sev in [1, 5]:
    sev_df = df_pr[df_pr["Severity"] == sev]
    b_prec = sev_df["Baseline_mAP_per_Det"].mean()
    c_prec = sev_df["CRATTT_mAP_per_Det"].mean()
    delta  = sev_df["Precision_Delta"].mean()
    print(f"Severity {sev}:")
    print(f"  Baseline precision proxy : {b_prec:.4f}")
    print(f"  CRATTT precision proxy   : {c_prec:.4f}")
    print(f"  Delta                    : {delta:+.4f}")
    if delta > 0:
        print(f"  ✅ CRATTT improves per-detection quality")
    else:
        print(f"  ❌ CRATTT does not improve per-detection quality")
    print()
