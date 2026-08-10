# ============================================================
# BLOCK V11: Session Summary — Session 2 (Noise+Blur)
# Combines two fixes discovered during Session 1:
# 1. groupby-free summary construction (pandas/NumPy version
#    mismatch causes KeyError: 'object64' inside pandas'
#    internal factorize/hashtable lookup on any .groupby()
#    call against string columns in this environment)
# 2. n_images varies per row now (1000 for glass_blur,
#    5000 for everything else), so header must show a range
#    and per-row N_images in the table itself
# ============================================================

import pandas as pd

df = pd.DataFrame(all_session_results)

print(f"\n{'='*65}")
print(f"SESSION SUMMARY: {RUN_NAME}")
print(f"Model: {MODEL.upper()} (Swin-B) | "
      f"Images per config: {df['n_images'].min()}–{df['n_images'].max()} "
      f"(varies: glass_blur={GLASS_BLUR_N}, others={N_IMAGES})")
print(f"{'='*65}")

combos = sorted(set(zip(df['category'], df['corruption'])))
records = []
for cat, corr in combos:
    sub = df[(df['category'] == cat) & (df['corruption'] == corr)]
    records.append({
        'category':   cat,
        'corruption': corr,
        'Mean_mAP':   round(sub['map'].mean(), 4),
        'Worst_mAP':  round(sub['map'].min(), 4),
        'Mean_CE':    round(sub['ce'].mean(), 4),
        'N_images':   int(sub['n_images'].iloc[0]),  # correct per-row now, not global
    })

summary = pd.DataFrame(records)
print(summary.to_string(index=False))

session_mce = df['ce'].mean()
print(f"\nSession mCE ({MODEL.upper()}, Swin-B): {session_mce:.4f}")
print(f"Clean mAP baseline used for CE calc: {CLEAN_MAP_DINO:.4f}")

# Flag glass_blur rows specifically, since they're the reduced-N exception
glass_blur_rows = df[df['corruption'] == 'glass_blur']
if len(glass_blur_rows) > 0:
    print(f"\nglass_blur exception check:")
    print(f"  Configs at N={GLASS_BLUR_N}: {len(glass_blur_rows)} "
          f"(expected {len(SEVERITIES)})")
    if (glass_blur_rows['n_images'] != GLASS_BLUR_N).any():
        print(f"  ⚠️  Some glass_blur configs did not use N={GLASS_BLUR_N}, check individually")
    else:
        print(f"  ✅ All glass_blur configs correctly used N={GLASS_BLUR_N}")

if df['ce'].min() < -0.05:
    print("\n⚠️  Some CE values are notably negative — worth reviewing")
else:
    print("\n✅ CE values within expected range relative to clean baseline")

if df['n_errors'].sum() > 0:
    print(f"⚠️  Total errors across session: {df['n_errors'].sum()}")
else:
    print("✅ Zero errors across all configurations")

summary_path = os.path.join(TABLES_DIR, f"{RUN_NAME}_summary.csv")
summary.to_csv(summary_path, index=False)
print(f"\n✅ Summary saved: {summary_path}")
