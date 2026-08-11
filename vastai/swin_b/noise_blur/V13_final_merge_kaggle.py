# ============================================================
# BLOCK V13: Final Merge — Swin-B DINO, Two Sessions
# Run on: Kaggle (free) — after both Vast.ai sessions done
# DINO-only (Swin-B). YOLO-World was never run on Swin-B,
# so it's excluded here rather than left as misleading blanks.
#
# Before running:
# 1. Upload both extracted results folders (or the merged
#    'crattt' folder containing both sessions' checkpoints)
#    as a Kaggle dataset
# 2. Update RESULTS_PATH below
# ============================================================

import glob, json, os
import pandas as pd

RESULTS_PATH = "/kaggle/input/your-swinb-results"  # update this

print("Loading all session checkpoint files...")

all_rows   = []
ckpt_files = sorted(glob.glob(os.path.join(RESULTS_PATH, "**/*.json"), recursive=True))
print(f"Found {len(ckpt_files)} checkpoint files")

for path in ckpt_files:
    try:
        with open(path) as f:
            row = json.load(f)
        if all(k in row for k in ['model', 'corruption', 'severity', 'map', 'ce']):
            if row.get('n_images', 0) > 0 and row.get('model') == 'dino':
                all_rows.append(row)
    except Exception as e:
        print(f"  Skipping {os.path.basename(path)}: {e}")

df = pd.DataFrame(all_rows)
print(f"\nTotal valid DINO records : {len(df)}")
print(f"Corruptions              : {df['corruption'].nunique()} (expected 15)")
print(f"Images per config        : {df['n_images'].min()}–{df['n_images'].max()} "
      f"(varies: glass_blur=1000, others=5000)")

CLEAN_MAP_DINO_SWINB = 0.5461

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
    sub = df[df['corruption'] == corruption]
    if len(sub) == 0:
        print(f"  ⚠️  Missing: {corruption}")
        table_rows.append({"Category": cat, "Corruption": corruption,
                            "Mean_mAP": None, "Worst_mAP": None, "CE": None, "N_images": None})
        continue

    mean_map  = sub['map'].mean()
    worst_map = sub[sub['severity'] == 5]['map'].mean()
    ce        = (1 - mean_map) / (1 - CLEAN_MAP_DINO_SWINB)
    n_img     = int(sub['n_images'].iloc[0])

    table_rows.append({
        "Category": cat, "Corruption": corruption,
        "Mean_mAP": round(mean_map, 4), "Worst_mAP": round(worst_map, 4),
        "CE": round(ce, 4), "N_images": n_img
    })

final_table = pd.DataFrame(table_rows)
dino_mce = final_table['CE'].mean()

mce_row = {"Category": "Overall", "Corruption": "mCE",
           "Mean_mAP": None, "Worst_mAP": None, "CE": round(dino_mce, 4), "N_images": None}
final_table = pd.concat([final_table, pd.DataFrame([mce_row])], ignore_index=True)

print()
print("=" * 75)
print("FINAL TABLE — Swin-B DINO, Full COCO val2017 Robustness Results")
print("=" * 75)
print(final_table.to_string(index=False))

print(f"\n{'='*55}")
print(f"FINAL mCE SUMMARY (Swin-B DINO)")
print(f"{'='*55}")
print(f"GroundingDINO (Swin-B) mCE : {dino_mce:.4f}")
print(f"Clean mAP baseline used    : {CLEAN_MAP_DINO_SWINB:.4f}")

if len(df) != 75:
    print(f"⚠️  Expected 75 total checkpoint records (15 corruptions × 5 severities), got {len(df)}")
else:
    print("✅ All 75 records present, no missing configurations")

output_path = "/kaggle/working/swinb_dino_full_coco_table.csv"
final_table.to_csv(output_path, index=False)
print(f"\n✅ Final table saved: {output_path}")
