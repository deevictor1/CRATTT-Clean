# ============================================================
# BLOCK V13: Final Merge and Table 4.2 Generation
# Run on: KAGGLE (free) — after ALL 4 Vast.ai sessions done
#
# Before running:
# 1. Download all 4 session ZIPs from Vast.ai
# 2. Extract all ZIPs to one folder on your computer
# 3. Upload that folder as a Kaggle dataset
# 4. Update RESULTS_PATH below
# ============================================================

import glob, json, os
import pandas as pd
import numpy as np

# UPDATE THIS PATH to your uploaded Kaggle dataset
RESULTS_PATH = "/kaggle/input/your-vastai-results"

print("Loading all session checkpoint files...")

all_rows   = []
ckpt_files = sorted(glob.glob(
    os.path.join(RESULTS_PATH, "**/*.json"),
    recursive=True
))

print(f"Found {len(ckpt_files)} checkpoint files")

for path in ckpt_files:
    try:
        with open(path) as f:
            row = json.load(f)
        if all(k in row for k in
               ['model', 'corruption', 'severity', 'map', 'ce']):
            if row.get('n_images', 0) > 0:
                all_rows.append(row)
    except Exception as e:
        print(f"  Skipping {os.path.basename(path)}: {e}")

df = pd.DataFrame(all_rows)
print(f"\nTotal valid records : {len(df)}")
print(f"Models              : {df['model'].unique().tolist()}")
print(f"Corruptions         : {df['corruption'].nunique()}")
print(f"Images per config   : {df['n_images'].iloc[0]}")

CLEAN_MAPS = {
    "dino": 0.3598,
    "yolo": 0.5221
}

corruptions_ordered = [
    ("Noise",   "gaussian_noise"),
    ("Noise",   "shot_noise"),
    ("Noise",   "impulse_noise"),
    ("Blur",    "defocus_blur"),
    ("Blur",    "glass_blur"),
    ("Blur",    "motion_blur"),
    ("Blur",    "zoom_blur"),
    ("Weather", "snow"),
    ("Weather", "frost"),
    ("Weather", "fog"),
    ("Weather", "brightness"),
    ("Digital", "contrast"),
    ("Digital", "elastic_transform"),
    ("Digital", "pixelate"),
    ("Digital", "jpeg_compression"),
]

table_rows = []
for cat, corruption in corruptions_ordered:
    row = {"Category": cat, "Corruption": corruption}
    for model in ["dino", "yolo"]:
        model_df = df[
            (df['model']      == model) &
            (df['corruption'] == corruption)
        ]
        if len(model_df) == 0:
            print(f"  ⚠️  Missing: {model} {corruption}")
            row[f"{model.upper()}_Mean_mAP"]  = None
            row[f"{model.upper()}_Worst_mAP"] = None
            row[f"{model.upper()}_CE"]        = None
            continue

        mean_map  = model_df['map'].mean()
        worst_map = model_df[
            model_df['severity'] == 5
        ]['map'].mean()
        clean_map = CLEAN_MAPS[model]
        ce        = (1 - mean_map) / (1 - clean_map)

        row[f"{model.upper()}_Mean_mAP"]  = round(mean_map,  4)
        row[f"{model.upper()}_Worst_mAP"] = round(worst_map, 4)
        row[f"{model.upper()}_CE"]        = round(ce,        4)

    table_rows.append(row)

final_table = pd.DataFrame(table_rows)
dino_mce    = final_table['DINO_CE'].mean()
yolo_mce    = final_table['YOLO_CE'].mean()

mce_row = {
    "Category":       "Overall",
    "Corruption":     "mCE",
    "DINO_Mean_mAP":  None,
    "DINO_Worst_mAP": None,
    "DINO_CE":        round(dino_mce, 4),
    "YOLO_Mean_mAP":  None,
    "YOLO_Worst_mAP": None,
    "YOLO_CE":        round(yolo_mce, 4),
}
final_table = pd.concat(
    [final_table, pd.DataFrame([mce_row])],
    ignore_index=True
)

print()
print("=" * 75)
print("FINAL TABLE 4.2 — Full COCO val2017 Robustness Results")
print(f"N = {df['n_images'].iloc[0]} images per configuration")
print("=" * 75)
print(final_table.to_string(index=False))

print(f"\n{'='*55}")
print(f"FINAL mCE SUMMARY")
print(f"{'='*55}")
print(f"GroundingDINO mCE : {dino_mce:.4f} "
      f"(Kaggle 20-image: 1.1983)")
print(f"YOLO-World mCE    : {yolo_mce:.4f} "
      f"(Kaggle 20-image: 1.4586)")
print(f"Robustness gap    : {yolo_mce - dino_mce:+.4f}")

if yolo_mce > dino_mce:
    print("✅ CONFIRMED: GroundingDINO more robust than YOLO-World")
else:
    print("⚠️  REVERSED: YOLO-World more robust at full scale")
    print("   Update Chapter 4 findings accordingly")

output_path = "/kaggle/working/FINAL_TABLE_4_2_full_coco.csv"
final_table.to_csv(output_path, index=False)
print(f"\n✅ Final Table 4.2 saved: {output_path}")
print("   Replace your Kaggle 20-image Table 4.2 with this")
