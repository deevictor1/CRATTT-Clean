# ============================================================
# BLOCK 10b: Rjoint Calibration Diagnostic
# Determines the correct tau for meaningful TTRV filtering
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from imagecorruptions import corrupt as ic_corrupt

print("Running Rjoint calibration across 5 images...")
print("Clean vs Severity-5 Snow comparison")
print("="*50)

all_rjoints_clean = []
all_rjoints_corrupt = []

for img_path in image_files[:5]:
    img = loaded_images[img_path]
    corrupted = ic_corrupt(img, corruption_name='snow', severity=5)

    # Collect all Rjoint scores including rejected ones
    # We need to run inference at tau=0 to see full distribution
    verified_c, _ = run_crattt_inference(img, tau=0.0)
    verified_n, _ = run_crattt_inference(corrupted, tau=0.0)

    all_rjoints_clean.extend([v['rjoint'] for v in verified_c])
    all_rjoints_corrupt.extend([v['rjoint'] for v in verified_n])

print(f"\nClean Rjoint distribution (N={len(all_rjoints_clean)}):")
print(f"  Min  : {min(all_rjoints_clean):.4f}")
print(f"  Max  : {max(all_rjoints_clean):.4f}")
print(f"  Mean : {np.mean(all_rjoints_clean):.4f}")
print(f"  Std  : {np.std(all_rjoints_clean):.4f}")
print(f"  P25  : {np.percentile(all_rjoints_clean, 25):.4f}")
print(f"  P50  : {np.percentile(all_rjoints_clean, 50):.4f}")
print(f"  P75  : {np.percentile(all_rjoints_clean, 75):.4f}")

print(f"\nCorrupted Rjoint distribution (N={len(all_rjoints_corrupt)}):")
print(f"  Min  : {min(all_rjoints_corrupt):.4f}")
print(f"  Max  : {max(all_rjoints_corrupt):.4f}")
print(f"  Mean : {np.mean(all_rjoints_corrupt):.4f}")
print(f"  Std  : {np.std(all_rjoints_corrupt):.4f}")
print(f"  P25  : {np.percentile(all_rjoints_corrupt, 25):.4f}")
print(f"  P50  : {np.percentile(all_rjoints_corrupt, 50):.4f}")
print(f"  P75  : {np.percentile(all_rjoints_corrupt, 75):.4f}")

# Plot distributions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(all_rjoints_clean, bins=20, color='green',
             alpha=0.7, label='Clean')
axes[0].hist(all_rjoints_corrupt, bins=20, color='red',
             alpha=0.7, label='Snow Sev-5')
axes[0].set_xlabel("Rjoint Score")
axes[0].set_ylabel("Count")
axes[0].set_title("Rjoint Distribution: Clean vs Corrupted")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Suggest calibrated tau values
clean_mean   = np.mean(all_rjoints_clean)
corrupt_mean = np.mean(all_rjoints_corrupt)
suggested_tau = (clean_mean + corrupt_mean) / 2

axes[1].hist(all_rjoints_clean, bins=20, color='green',
             alpha=0.7, label='Clean')
axes[1].hist(all_rjoints_corrupt, bins=20, color='red',
             alpha=0.7, label='Snow Sev-5')
axes[1].axvline(x=suggested_tau, color='black', linewidth=2,
                linestyle='--',
                label=f'Suggested τ={suggested_tau:.3f}')
axes[1].axvline(x=CRATTT_PARAMS["tau"], color='blue',
                linewidth=2, linestyle=':',
                label=f'Current τ={CRATTT_PARAMS["tau"]}')
axes[1].set_xlabel("Rjoint Score")
axes[1].set_title("Threshold Calibration")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle("TTRV Gate Calibration Diagnostic",
             fontweight='bold')
plt.tight_layout()

fig_path = os.path.join(
    EVAL_PARAMS["fig_dir"], "figure_rjoint_calibration.pdf"
)
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"\n--- Calibration Recommendation ---")
print(f"Current tau    : {CRATTT_PARAMS['tau']:.3f}")
print(f"Suggested tau  : {suggested_tau:.3f}")
print(f"Clean mean     : {clean_mean:.4f}")
print(f"Corrupt mean   : {corrupt_mean:.4f}")
print(f"Mean separation: {clean_mean - corrupt_mean:.4f}")
print(f"\n✅ Calibration figure saved: {fig_path}")
