# ============================================================
# BLOCK 13: Visual Comparison Gallery
# Generates a multi-page PDF showing Baseline vs CRATTT
# side-by-side for representative corruptions.
# Produces Figure 4.2 for the dissertation.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
from imagecorruptions import corrupt as ic_corrupt
import torch
import os
import json
from PIL import Image
from IPython.display import display, FileLink

# --- 13.1 Configuration ---
# Select one representative corruption per category
GALLERY_CONFIGS = [
    ("Noise",   "gaussian_noise", 5),
    ("Blur",    "motion_blur",    5),
    ("Weather", "fog",            5),
    ("Digital", "contrast",       5),
    # Add medium severity for comparison
    ("Noise",   "gaussian_noise", 3),
    ("Weather", "snow",           3),
]

# Use first 4 images for gallery
GALLERY_IMAGES = image_files[:4]

print("="*50)
print("BLOCK 13: VISUAL COMPARISON GALLERY")
print("="*50)
print(f"Configurations : {len(GALLERY_CONFIGS)}")
print(f"Images         : {len(GALLERY_IMAGES)}")
print(f"Total pages    : {len(GALLERY_CONFIGS) * len(GALLERY_IMAGES) + 1}")
print(f"Tau            : {CRATTT_PARAMS['tau']}")

# --- 13.2 Helper: Draw boxes on axis ---
def draw_dino_boxes(ax, image_np, result, threshold=0.12,
                    color='red', linewidth=2):
    """Draw GroundingDINO boxes on a matplotlib axis."""
    ax.imshow(image_np)
    labels = result.get("text_labels", result.get("labels", []))

    for i, (box, score) in enumerate(
        zip(result["boxes"], result["scores"])
    ):
        if score.item() < threshold:
            continue
        b = box.tolist()
        x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
        ax.add_patch(patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=linewidth, edgecolor=color,
            facecolor='none', alpha=0.9
        ))
        label = labels[i] if i < len(labels) else ""
        if isinstance(label, str):
            clean = label.replace(".", "").strip()
            ax.text(x1, max(y1-4, 8), clean,
                   color=color, fontsize=6, fontweight='bold',
                   bbox=dict(facecolor='black', alpha=0.4, pad=1))
    ax.axis('off')


def draw_crattt_boxes(ax, image_np, crattt_preds,
                      color='lime', linewidth=2):
    """Draw CRATTT verified boxes on a matplotlib axis."""
    ax.imshow(image_np)
    for pred in crattt_preds:
        b = pred['bbox']  # [x, y, w, h]
        ax.add_patch(patches.Rectangle(
            (b[0], b[1]), b[2], b[3],
            linewidth=linewidth, edgecolor=color,
            facecolor='none', alpha=0.9
        ))
        label = pred.get('label', '')
        rjoint = pred.get('rjoint', 0)
        ax.text(b[0], max(b[1]-4, 8),
               f"{label} R={rjoint:.2f}",
               color=color, fontsize=6, fontweight='bold',
               bbox=dict(facecolor='black', alpha=0.4, pad=1))
    ax.axis('off')


# --- 13.3 Main Gallery Loop ---
pdf_path = os.path.join(
    EVAL_PARAMS["fig_dir"],
    "figure_4_2_comparison_gallery.pdf"
)

gallery_summary = []

with PdfPages(pdf_path) as pdf:

    # --- Page 1: Cover Summary Table ---
    fig_cover, ax_cover = plt.subplots(figsize=(14, 6))
    ax_cover.axis('off')

    # Load Block 12 results for the summary
    comp_csv = os.path.join(
        EVAL_PARAMS["table_dir"], "table_4_5_comparative.csv"
    )
    import pandas as pd
    df_comp = pd.read_csv(comp_csv)

    # Show category summary on cover
    cat_summary = df_comp.groupby("Category").agg(
        Baseline_mAP=("Baseline_mAP", "mean"),
        CRATTT_mAP=("CRATTT_mAP", "mean"),
        mAP_Delta=("mAP_Delta", "mean"),
        Mean_Rejected=("Mean_Rejected", "mean")
    ).round(3).reset_index()

    table_data = cat_summary.values
    table_cols = ["Category", "Baseline mAP",
                  "CRATTT mAP", "mAP Delta", "Mean Rejected"]

    tbl = ax_cover.table(
        cellText=table_data,
        colLabels=table_cols,
        loc='center', cellLoc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 2)

    ax_cover.set_title(
        "Figure 4.2: CRATTT vs Baseline — Category Summary\n"
        f"N=20 images | τ={CRATTT_PARAMS['tau']} | "
        f"α={CRATTT_PARAMS['alpha']} | β={CRATTT_PARAMS['beta']}",
        fontsize=13, fontweight='bold', pad=20
    )
    pdf.savefig(fig_cover, bbox_inches='tight')
    plt.show()
    plt.close(fig_cover)

    # --- Pages 2+: Visual Comparisons ---
    for cat_name, corruption, severity in GALLERY_CONFIGS:
        print(f"\n📸 Generating gallery: {corruption} sev{severity}")

        for img_path in GALLERY_IMAGES:
            img_id  = img_id_map[os.path.basename(img_path)]
            raw_img = loaded_images[img_path]
            fname   = os.path.basename(img_path)

            # Apply corruption
            c_img = ic_corrupt(
                raw_img,
                corruption_name=corruption,
                severity=severity
            )

            # Baseline inference
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

            # CRATTT inference
            crattt_preds, stats = run_crattt_inference(c_img)
            for p in crattt_preds:
                p["image_id"] = img_id

            # Compute mAP for caption
            baseline_coco = dino_to_coco_format(
                baseline_res, img_id
            )
            crattt_coco = [{
                "image_id":    p["image_id"],
                "category_id": p["category_id"],
                "bbox":        p["bbox"],
                "score":       p["score"]
            } for p in crattt_preds]

            bmap, _ = compute_map(
                baseline_coco, coco_gt, [img_id]
            )
            cmap, _ = compute_map(
                crattt_coco, coco_gt, [img_id]
            )

            n_baseline = len(baseline_res['boxes'])
            n_crattt   = stats['n_verified']
            n_rejected = stats['n_rejected']

            # Store for summary
            gallery_summary.append({
                "corruption": corruption,
                "severity":   severity,
                "image":      fname,
                "baseline_mAP": round(bmap, 4),
                "crattt_mAP":   round(cmap, 4),
                "delta":        round(cmap - bmap, 4),
                "n_baseline":   n_baseline,
                "n_crattt":     n_crattt,
                "n_rejected":   n_rejected
            })

            # --- Create figure ---
            fig, axes = plt.subplots(1, 3, figsize=(24, 8))

            # Panel 1: Clean reference
            axes[0].imshow(raw_img)
            axes[0].set_title(
                f"CLEAN REFERENCE\n{fname}",
                fontsize=10, fontweight='bold', color='white',
                backgroundcolor='black'
            )
            axes[0].axis('off')

            # Panel 2: Baseline DINO
            draw_dino_boxes(
                axes[1], c_img, baseline_res,
                threshold=CRATTT_PARAMS["dino_text_thr"],
                color='red'
            )
            axes[1].set_title(
                f"BASELINE (GroundingDINO)\n"
                f"Detections: {n_baseline} | "
                f"mAP: {bmap:.4f}",
                fontsize=10, fontweight='bold', color='red'
            )

            # Panel 3: CRATTT verified
            draw_crattt_boxes(
                axes[2], c_img, crattt_preds,
                color='lime'
            )
            axes[2].set_title(
                f"CRATTT (BARON + Oracle TTRV)\n"
                f"Verified: {n_crattt} | Rejected: {n_rejected} | "
                f"mAP: {cmap:.4f}",
                fontsize=10, fontweight='bold', color='lime'
            )

            plt.suptitle(
                f"Corruption: {corruption.replace('_',' ').title()} "
                f"| Severity: {severity} | τ={CRATTT_PARAMS['tau']}",
                fontsize=13, fontweight='bold', y=1.01
            )
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.show()
            plt.close(fig)

        print(f"  ✅ {corruption} sev{severity} gallery complete")

    # --- Final Page: Gallery Summary Table ---
    df_gallery = pd.DataFrame(gallery_summary)
    fig_end, ax_end = plt.subplots(figsize=(16, 8))
    ax_end.axis('off')

    end_data = df_gallery[[
        "corruption", "severity", "image",
        "baseline_mAP", "crattt_mAP", "delta",
        "n_baseline", "n_crattt", "n_rejected"
    ]].values

    end_cols = [
        "Corruption", "Sev", "Image",
        "Baseline mAP", "CRATTT mAP", "Δ",
        "Base Dets", "CRATTT Dets", "Rejected"
    ]

    end_tbl = ax_end.table(
        cellText=end_data,
        colLabels=end_cols,
        loc='center', cellLoc='center'
    )
    end_tbl.auto_set_font_size(False)
    end_tbl.set_fontsize(8)
    end_tbl.scale(1, 1.5)

    ax_end.set_title(
        "Gallery Summary: All Visual Comparison Results",
        fontsize=13, fontweight='bold', pad=20
    )
    pdf.savefig(fig_end, bbox_inches='tight')
    plt.close(fig_end)

    # Add PDF metadata
    d = pdf.infodict()
    d['Title']   = 'CRATTT Visual Comparison Gallery'
    d['Author']  = 'Dada Victor Damilare'
    d['Subject'] = 'MRES7015 Dissertation Figure 4.2'

# --- 13.4 Save Gallery Summary ---
gallery_csv = os.path.join(
    EVAL_PARAMS["table_dir"], "gallery_summary.csv"
)
df_gallery.to_csv(gallery_csv, index=False)

print(f"\n✅ Gallery PDF saved: {pdf_path}")
print(f"✅ Gallery summary saved: {gallery_csv}")
display(FileLink(pdf_path))

print("\n" + "="*50)
print("BLOCK 13 COMPLETE — Visual gallery generated")
print("="*50)
