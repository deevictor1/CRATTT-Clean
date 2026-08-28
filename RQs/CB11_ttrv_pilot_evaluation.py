# ============================================================
# BLOCK 11: TTRV Pilot Evaluation
# Compares Baseline DINO vs CRATTT on 5 images
# across three corruption severities.
# Produces the pilot results table for Chapter 3 Section 3.9
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
from imagecorruptions import corrupt as ic_corrupt
from pycocotools.cocoeval import COCOeval
import json, os

# --- 11.1 Pilot Configuration ---
PILOT_IMAGES      = image_files[:EVAL_PARAMS["num_pilot"]]
PILOT_CORRUPTION  = 'snow'
PILOT_SEVERITIES  = [1, 3, 5]  # Low, medium, extreme

print("="*50)
print("BLOCK 11: TTRV PILOT EVALUATION")
print("="*50)
print(f"Images     : {len(PILOT_IMAGES)}")
print(f"Corruption : {PILOT_CORRUPTION}")
print(f"Severities : {PILOT_SEVERITIES}")
print(f"Tau        : {CRATTT_PARAMS['tau']}")
print(f"Alpha      : {CRATTT_PARAMS['alpha']}")
print(f"Beta       : {CRATTT_PARAMS['beta']}")
print()

# --- 11.2 Pilot Loop ---
pilot_rows = []
results_store = []  # For visualisation

for img_path in PILOT_IMAGES:
    img_id  = img_id_map[os.path.basename(img_path)]
    raw_img = loaded_images[img_path]
    fname   = os.path.basename(img_path)

    for sev in PILOT_SEVERITIES:

        if sev == 0:
            c_img = raw_img.copy()
        else:
            c_img = ic_corrupt(
                raw_img,
                corruption_name=PILOT_CORRUPTION,
                severity=sev
            )

        # --- Baseline: DINO only ---
        inputs = dino_processor(
            images=c_img,
            text=DINO_TEXT_PROMPT,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = dino_model(**inputs)

        baseline_res = dino_processor\
            .post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                target_sizes=[c_img.shape[:2]],
                text_threshold=CRATTT_PARAMS["dino_text_thr"]
            )[0]

        # Format baseline for mAP
        baseline_preds = dino_to_coco_format(baseline_res, img_id)

        # --- CRATTT ---
        crattt_preds, stats = run_crattt_inference(c_img)

        # Set correct image_id on CRATTT predictions
        for p in crattt_preds:
            p["image_id"] = img_id

        # Format CRATTT for mAP
        crattt_coco_preds = [{
            "image_id":    p["image_id"],
            "category_id": p["category_id"],
            "bbox":        p["bbox"],
            "score":       p["score"]
        } for p in crattt_preds]

        # Compute per-image mAP
        bmap, _ = compute_map(
            baseline_preds, coco_gt, [img_id]
        )
        cmap, _ = compute_map(
            crattt_coco_preds, coco_gt, [img_id]
        )

        improvement = round(cmap - bmap, 4)

        pilot_rows.append({
            "Image":             fname,
            "Severity":          sev,
            "Baseline_mAP":      round(bmap, 4),
            "CRATTT_mAP":        round(cmap, 4),
            "mAP_Improvement":   improvement,
            "Baseline_Dets":     len(baseline_res['boxes']),
            "CRATTT_Dets":       stats['n_verified'],
            "Rejected":          stats['n_rejected'],
            "Improvement_Yield": (stats['n_verified']
                                  - len(baseline_res['boxes']))
        })

        # Store for visualisation (severity 5 only)
        if sev == 5:
            results_store.append({
                "img_path":       img_path,
                "img_id":         img_id,
                "fname":          fname,
                "corrupted_img":  c_img,
                "baseline_boxes": baseline_res['boxes'],
                "crattt_preds":   crattt_preds,
                "baseline_dets":  len(baseline_res['boxes']),
                "crattt_dets":    stats['n_verified'],
                "baseline_mAP":   bmap,
                "crattt_mAP":     cmap
            })

    print(f"  ✅ {fname} complete")

# --- 11.3 Pilot Results Table ---
df_pilot = pd.DataFrame(pilot_rows)

print("\n" + "="*70)
print("TABLE 3.1: TTRV PILOT RESULTS")
print(f"Corruption: {PILOT_CORRUPTION.upper()} | "
      f"Tau={CRATTT_PARAMS['tau']}")
print("="*70)

from IPython.display import display
display(df_pilot)

# Summary statistics
print("\n--- Summary ---")
for sev in PILOT_SEVERITIES:
    sev_df = df_pilot[df_pilot["Severity"] == sev]
    print(f"Severity {sev}:")
    print(f"  Mean Baseline mAP : "
          f"{sev_df['Baseline_mAP'].mean():.4f}")
    print(f"  Mean CRATTT mAP   : "
          f"{sev_df['CRATTT_mAP'].mean():.4f}")
    print(f"  Mean Improvement  : "
          f"{sev_df['mAP_Improvement'].mean():.4f}")
    print(f"  Mean Rejected     : "
          f"{sev_df['Rejected'].mean():.1f}")

# --- 11.4 Save Pilot Results ---
csv_path = os.path.join(
    EVAL_PARAMS["table_dir"], "table_3_1_pilot.csv"
)
df_pilot.to_csv(csv_path, index=False)
print(f"\n✅ Pilot table saved: {csv_path}")

# --- 11.5 Visual Comparison Gallery (Severity 5) ---
print(f"\nGenerating visual gallery for severity 5...")

pdf_path = os.path.join(
    EVAL_PARAMS["fig_dir"], "figure_3_1_pilot_gallery.pdf"
)

with PdfPages(pdf_path) as pdf:

    # Page 1: Summary table
    fig_t, ax_t = plt.subplots(figsize=(14, 4))
    ax_t.axis('off')
    summary_data = df_pilot[
        df_pilot["Severity"] == 5
    ][[
        "Image", "Baseline_mAP", "CRATTT_mAP",
        "mAP_Improvement", "Baseline_Dets", "CRATTT_Dets"
    ]].values
    summary_cols = [
        "Image", "Baseline mAP", "CRATTT mAP",
        "Improvement", "Baseline Dets", "CRATTT Dets"
    ]
    ax_t.table(
        cellText=summary_data,
        colLabels=summary_cols,
        loc='center',
        cellLoc='center'
    )
    ax_t.set_title(
        f"Table 3.1: TTRV Pilot — Snow Severity 5 | "
        f"τ={CRATTT_PARAMS['tau']}",
        fontsize=13, fontweight='bold', pad=20
    )
    pdf.savefig(fig_t, bbox_inches='tight')
    plt.close(fig_t)

    # Pages 2+: Visual comparisons
    for data in results_store:
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        # Left: Baseline
        axes[0].imshow(data['corrupted_img'])
        axes[0].set_title(
            f"BASELINE (DINO only)\n"
            f"Detections: {data['baseline_dets']} | "
            f"mAP: {data['baseline_mAP']:.4f}",
            color='red', fontsize=12, fontweight='bold'
        )
        for box in data['baseline_boxes']:
            b = box.cpu().numpy()
            axes[0].add_patch(patches.Rectangle(
                (b[0], b[1]), b[2]-b[0], b[3]-b[1],
                linewidth=2, edgecolor='red',
                facecolor='none', alpha=0.8
            ))
        axes[0].axis('off')

        # Right: CRATTT
        axes[1].imshow(data['corrupted_img'])
        axes[1].set_title(
            f"CRATTT (BARON + Oracle TTRV)\n"
            f"Verified: {data['crattt_dets']} | "
            f"mAP: {data['crattt_mAP']:.4f}",
            color='lime', fontsize=12, fontweight='bold'
        )
        for pred in data['crattt_preds']:
            b = pred['bbox']
            axes[1].add_patch(patches.Rectangle(
                (b[0], b[1]), b[2], b[3],
                linewidth=2, edgecolor='lime',
                facecolor='none'
            ))
            axes[1].text(
                b[0], b[1] - 5,
                f"{pred['label']} R={pred['rjoint']:.2f}",
                color='lime', fontsize=7,
                bbox=dict(facecolor='black', alpha=0.5)
            )
        axes[1].axis('off')

        plt.suptitle(
            f"Pilot Visual Audit: {data['fname']}\n"
            f"Snow Severity 5 | τ={CRATTT_PARAMS['tau']}",
            fontsize=14, fontweight='bold'
        )
        pdf.savefig(fig, bbox_inches='tight')
        plt.show()
        plt.close(fig)

print(f"✅ Visual gallery saved: {pdf_path}")

print("\n" + "="*50)
print("BLOCK 11 COMPLETE — Pilot evaluation done")
print("="*50)
