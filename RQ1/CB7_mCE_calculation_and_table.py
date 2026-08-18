# ============================================================
# BLOCK 7: mCE Calculation & Summary Table
# Computes Corruption Error per corruption and overall mCE.
# Produces Table 4.2a
# ============================================================

import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from IPython.display import display

# --- 7.1 Load Results (supports re-running from checkpoints) ---
# Use in-memory results if available, otherwise reload from disk
if 'all_sweep_results' not in dir() or len(all_sweep_results) == 0:
    full_results_path = os.path.join(
        EVAL_PARAMS["save_dir"], "full_sweep_results.json"
    )
    with open(full_results_path) as f:
        all_sweep_results = json.load(f)
    print(f"✅ Results reloaded from disk: {len(all_sweep_results)} corruptions")
else:
    print(f"✅ Using in-memory results: {len(all_sweep_results)} corruptions")

# --- 7.2 Compute CE and mCE ---
# CE formula: mean(1 - mAP_corrupted) / (1 - mAP_clean)
# This is Relative Corruption Error (RCE) using each model's own
# clean performance as the baseline denominator.
# Referred to as mCE throughout the dissertation per Section 3.8.2.

err_clean_dino = 1 - map_clean_dino
err_clean_yolo = 1 - map_clean_yolo

rows = []
for corruption, data in all_sweep_results.items():
    dino_maps = data["dino_mAP"]
    yolo_maps = data["yolo_mAP"]

    # Mean mAP across severities 1-5
    mean_dino = float(np.mean(dino_maps))
    mean_yolo = float(np.mean(yolo_maps))

    # CE = mean error under corruption / error on clean
    ce_dino = float(np.mean([1 - m for m in dino_maps])) / err_clean_dino
    ce_yolo = float(np.mean([1 - m for m in yolo_maps])) / err_clean_yolo

    # Worst severity mAP (severity 5)
    worst_dino = float(dino_maps[4])
    worst_yolo = float(yolo_maps[4])

    rows.append({
        "Category":       data["category"],
        "Corruption":     corruption.replace("_", " ").title(),
        "DINO_Mean_mAP":  round(mean_dino, 4),
        "YOLO_Mean_mAP":  round(mean_yolo, 4),
        "DINO_Worst_mAP": round(worst_dino, 4),
        "YOLO_Worst_mAP": round(worst_yolo, 4),
        "DINO_CE":        round(ce_dino, 4),
        "YOLO_CE":        round(ce_yolo, 4),
    })

df = pd.DataFrame(rows)
df = df.sort_values(["Category", "Corruption"]).reset_index(drop=True)

# --- 7.3 Overall mCE ---
mCE_dino = round(float(df["DINO_CE"].mean()), 4)
mCE_yolo = round(float(df["YOLO_CE"].mean()), 4)

# --- 7.4 Category-Level mCE ---
cat_summary = df.groupby("Category").agg(
    DINO_Cat_mCE=("DINO_CE", "mean"),
    YOLO_Cat_mCE=("YOLO_CE", "mean"),
    DINO_Cat_mAP=("DINO_Mean_mAP", "mean"),
    YOLO_Cat_mAP=("YOLO_Mean_mAP", "mean"),
).round(4).reset_index()

# --- 7.5 Display Results ---
print("\n" + "="*70)
print("TABLE FOR BASELINE ROBUSTNESS SUMMARY (ImageNet-C Protocol)")
print("="*70)
print(f"Clean Baseline — DINO: {map_clean_dino:.4f} | "
      f"YOLO: {map_clean_yolo:.4f}")
print(f"Overall mCE   — DINO: {mCE_dino:.4f} | "
      f"YOLO: {mCE_yolo:.4f}")
print(f"(Lower mCE = more robust)")
print("="*70)
display(df)

print("\n--- Category-Level Summary ---")
display(cat_summary)

# --- 7.6 Save Tables ---
csv_path = os.path.join(EVAL_PARAMS["table_dir"], "table_4_1_baseline.csv")
df.to_csv(csv_path, index=False)

cat_csv_path = os.path.join(
    EVAL_PARAMS["table_dir"], "table_4_1_category_summary.csv"
)
cat_summary.to_csv(cat_csv_path, index=False)

summary_dict = {
    "map_clean_dino": map_clean_dino,
    "map_clean_yolo": map_clean_yolo,
    "mCE_dino":       mCE_dino,
    "mCE_yolo":       mCE_yolo,
    "n_images":       len(image_files),
    "n_corruptions":  len(all_sweep_results),
    "per_corruption": rows
}
with open(os.path.join(EVAL_PARAMS["save_dir"],
                       "mCE_summary.json"), "w") as f:
    json.dump(summary_dict, f, indent=2)

print(f"\n✅ Tables saved:")
print(f"   {csv_path}")
print(f"   {cat_csv_path}")

# --- 7.7 Generate Dissertation Figure ---
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(
    "Figure 4.1: Baseline Robustness — mAP Decay under ImageNet-C Corruptions\n"
    f"(N=20 images, GroundingDINO clean={map_clean_dino:.3f}, "
    f"YOLO-World clean={map_clean_yolo:.3f})",
    fontsize=13, fontweight='bold'
)
axes_flat = axes.flatten()

for idx, (cat_name, corruptions) in enumerate(
    CORRUPTION_CATEGORIES.items()
):
    ax = axes_flat[idx]

    for corruption in corruptions:
        data = all_sweep_results[corruption]
        label_name = corruption.replace("_", " ").title()

        ax.plot(SEVERITIES, data["dino_mAP"],
                marker='o', linewidth=1.5, alpha=0.8,
                label=f"DINO {label_name}")
        ax.plot(SEVERITIES, data["yolo_mAP"],
                marker='s', linewidth=1.5, alpha=0.8,
                linestyle='--',
                label=f"YOLO {label_name}")

    # Add clean baseline reference line
    ax.axhline(y=map_clean_dino, color='green',
               linestyle=':', alpha=0.5, label='DINO clean')
    ax.axhline(y=map_clean_yolo, color='blue',
               linestyle=':', alpha=0.5, label='YOLO clean')

    ax.set_title(f"{cat_name} Corruptions", fontweight='bold')
    ax.set_xlabel("Severity Level")
    ax.set_ylabel("mAP@0.50:0.95")
    ax.set_ylim(0, 0.65)
    ax.set_xticks(SEVERITIES)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)

plt.tight_layout()

fig_path = os.path.join(EVAL_PARAMS["fig_dir"], "figure_4_1_baseline.pdf")
plt.savefig(fig_path, format='pdf', dpi=300, bbox_inches='tight')
plt.show()
print(f"✅ Figure 4.1 saved: {fig_path}")

print("\n" + "="*50)
print("BLOCK 7 COMPLETE — mCE table and figures generated")
print("="*50)
