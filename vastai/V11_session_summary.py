# ============================================================
# BLOCK V11: Session Summary
# Run on: Vast.ai — after Block V10 completes or when stopping
# ============================================================

import pandas as pd

df = pd.DataFrame(all_session_results)

print(f"\n{'='*65}")
print(f"SESSION SUMMARY: {RUN_NAME}")
print(f"Model: {MODEL.upper()} | "
      f"Images per config: {df['n_images'].iloc[0]}")
print(f"{'='*65}")

summary = df.groupby(['category', 'corruption']).agg(
    Mean_mAP  =('map', 'mean'),
    Worst_mAP =('map', 'min'),
    Mean_CE   =('ce',  'mean'),
    N_images  =('n_images', 'first')
).round(4).reset_index()

print(summary.to_string(index=False))

session_mce = df['ce'].mean()
print(f"\nSession partial mCE ({MODEL.upper()}): "
      f"{session_mce:.4f}")

kaggle_mce = 1.1983 if MODEL == "dino" else 1.4586
print(f"Kaggle 20-image mCE ({MODEL.upper()}): "
      f"{kaggle_mce:.4f}")
print(f"Difference: {session_mce - kaggle_mce:+.4f}")

if abs(session_mce - kaggle_mce) < 0.15:
    print("✅ Results consistent with Kaggle baseline")
else:
    print("⚠️  Large difference — check corruption implementations")

summary_path = os.path.join(
    TABLES_DIR, f"{RUN_NAME}_summary.csv"
)
summary.to_csv(summary_path, index=False)
print(f"\n✅ Summary saved: {summary_path}")
