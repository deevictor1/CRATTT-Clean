# ============================================================
# BLOCK 12: CRATTT vs Baseline Comparative Sweep
# Runs CRATTT against baseline DINO across all 15 corruptions
# at severity 3 and 5 on 20 images.
# Produces the primary comparison table for Chapter 4.
# ============================================================

import pandas as pd
import numpy as np
import json, os
from imagecorruptions import corrupt as ic_corrupt
from tqdm.notebook import tqdm

# --- 12.1 Configuration ---
EVAL_SEVERITIES = [3, 5]  # Medium and extreme
print("="*50)
print("BLOCK 12: FULL CRATTT COMPARATIVE SWEEP")
print("="*50)
print(f"Images      : {len(image_files)}")
print(f"Corruptions : {len(ALL_CORRUPTIONS)}")
print(f"Severities  : {EVAL_SEVERITIES}")
print(f"Tau         : {CRATTT_PARAMS['tau']}")
print(f"Total runs  : "
      f"{len(image_files)*len(ALL_CORRUPTIONS)*len(EVAL_SEVERITIES)}")
print()

# --- 12.2 Resume Logic ---
comp_results = {}
comp_ckpt_dir = os.path.join(
    EVAL_PARAMS["ckpt_dir"], "comparative"
)
os.makedirs(comp_ckpt_dir, exist_ok=True)

def load_comp_checkpoint(corruption, severity):
    path = os.path.join(
        comp_ckpt_dir, f"{corruption}_sev{severity}.json"
    )
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def save_comp_checkpoint(corruption, severity, data):
    path = os.path.join(
        comp_ckpt_dir, f"{corruption}_sev{severity}.json"
    )
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

# --- 12.3 Main Loop ---
comp_rows = []

for cat_name, corruptions in CORRUPTION_CATEGORIES.items():
    print(f"\n📂 {cat_name}")

    for corruption in corruptions:
        for sev in EVAL_SEVERITIES:

            # Check checkpoint
            ckpt = load_comp_checkpoint(corruption, sev)
            if ckpt is not None:
                print(f"  ↩️  {corruption} sev{sev}: "
                      f"from checkpoint")
                comp_rows.append(ckpt)
                continue

            print(f"  ▶  {corruption} sev{sev}...",
                  end=" ", flush=True)

            baseline_preds_all = []
            crattt_preds_all   = []

            baseline_dets_list = []
            crattt_dets_list   = []
            rejected_list      = []

            for img_path in image_files:
                img_id  = img_id_map[
                    os.path.basename(img_path)
                ]
                raw_img = loaded_images[img_path]

                c_img = ic_corrupt(
                    raw_img,
                    corruption_name=corruption,
                    severity=sev
                )

                # Baseline DINO
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
                        text_threshold=CRATTT_PARAMS[
                            "dino_text_thr"
                        ]
                    )[0]

                baseline_preds_all.extend(
                    dino_to_coco_format(baseline_res, img_id)
                )
                baseline_dets_list.append(
                    len(baseline_res['boxes'])
                )

                # CRATTT
                crattt_preds, stats = run_crattt_inference(c_img)
                for p in crattt_preds:
                    p["image_id"] = img_id

                crattt_preds_all.extend([{
                    "image_id":    p["image_id"],
                    "category_id": p["category_id"],
                    "bbox":        p["bbox"],
                    "score":       p["score"]
                } for p in crattt_preds])

                crattt_dets_list.append(stats['n_verified'])
                rejected_list.append(stats['n_rejected'])

            # Compute mAP
            bmap, bs = compute_map(
                baseline_preds_all, coco_gt, coco_img_ids
            )
            cmap, cs = compute_map(
                crattt_preds_all, coco_gt, coco_img_ids
            )

            row = {
                "Category":          cat_name,
                "Corruption":        corruption,
                "Severity":          sev,
                "Baseline_mAP":      round(bmap, 4),
                "CRATTT_mAP":        round(cmap, 4),
                "mAP_Delta":         round(cmap - bmap, 4),
                "Mean_Baseline_Dets": round(
                    np.mean(baseline_dets_list), 2
                ),
                "Mean_CRATTT_Dets":  round(
                    np.mean(crattt_dets_list), 2
                ),
                "Mean_Rejected":     round(
                    np.mean(rejected_list), 2
                ),
                "Baseline_status":   bs,
                "CRATTT_status":     cs
            }

            save_comp_checkpoint(corruption, sev, row)
            comp_rows.append(row)

            print(f"B={bmap:.3f} C={cmap:.3f} "
                  f"Δ={cmap-bmap:+.3f}")

# --- 12.4 Results Table ---
df_comp = pd.DataFrame(comp_rows)

print("\n" + "="*70)
print("TABLE 4.5: CRATTT vs BASELINE COMPARATIVE RESULTS")
print("="*70)
from IPython.display import display
display(df_comp[[
    "Category", "Corruption", "Severity",
    "Baseline_mAP", "CRATTT_mAP", "mAP_Delta",
    "Mean_Baseline_Dets", "Mean_CRATTT_Dets", "Mean_Rejected"
]])

# --- 12.5 Summary by Category and Severity ---
print("\n--- Summary by Severity ---")
for sev in EVAL_SEVERITIES:
    sev_df = df_comp[df_comp["Severity"] == sev]
    print(f"\nSeverity {sev}:")
    print(f"  Mean Baseline mAP : "
          f"{sev_df['Baseline_mAP'].mean():.4f}")
    print(f"  Mean CRATTT mAP   : "
          f"{sev_df['CRATTT_mAP'].mean():.4f}")
    print(f"  Mean Delta        : "
          f"{sev_df['mAP_Delta'].mean():+.4f}")
    print(f"  Mean Rejected     : "
          f"{sev_df['Mean_Rejected'].mean():.1f}")

print("\n--- Summary by Category ---")
cat_comp = df_comp.groupby("Category").agg(
    Baseline_mAP=("Baseline_mAP", "mean"),
    CRATTT_mAP=("CRATTT_mAP", "mean"),
    mAP_Delta=("mAP_Delta", "mean"),
    Mean_Rejected=("Mean_Rejected", "mean")
).round(4)
display(cat_comp)

# --- 12.6 Save ---
csv_path = os.path.join(
    EVAL_PARAMS["table_dir"], "table_4_5_comparative.csv"
)
df_comp.to_csv(csv_path, index=False)
print(f"\n✅ Comparative table saved: {csv_path}")

print("\n" + "="*50)
print("BLOCK 12 COMPLETE — Full comparison done")
print("="*50)
