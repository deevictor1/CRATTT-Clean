# ============================================================
# BLOCK 11b: Dynamic Tau Ablation
# Tests CRATTT with severity-adaptive threshold.
# Justified by Block 11 finding that fixed tau=0.321
# is too aggressive at low severity.
# ============================================================

import pandas as pd
import numpy as np
from imagecorruptions import corrupt as ic_corrupt

def get_dynamic_tau(severity):
    """
    Empirically calibrated dynamic threshold.
    At severity 1: stricter (closer to clean Rjoint mean 0.350)
    At severity 5: more permissive (closer to corrupt mean 0.320)
    Linear interpolation between the two.
    """
    # Severity 1 → tau=0.340 (near clean mean, strict)
    # Severity 5 → tau=0.310 (near corrupt mean, permissive)
    tau_high = 0.340  # severity 1
    tau_low  = 0.310  # severity 5
    tau = tau_high - (tau_high - tau_low) * (severity - 1) / 4
    return round(tau, 3)

print("Dynamic tau schedule:")
for s in [1, 2, 3, 4, 5]:
    print(f"  Severity {s}: τ={get_dynamic_tau(s)}")

print()

# --- Run Dynamic Tau Pilot ---
dynamic_rows = []

for img_path in PILOT_IMAGES:
    img_id  = img_id_map[os.path.basename(img_path)]
    raw_img = loaded_images[img_path]
    fname   = os.path.basename(img_path)

    for sev in PILOT_SEVERITIES:
        c_img = ic_corrupt(
            raw_img,
            corruption_name=PILOT_CORRUPTION,
            severity=sev
        )

        # Baseline
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
        baseline_preds = dino_to_coco_format(baseline_res, img_id)

        # CRATTT with dynamic tau
        dynamic_tau = get_dynamic_tau(sev)
        crattt_preds, stats = run_crattt_inference(
            c_img, tau=dynamic_tau
        )
        for p in crattt_preds:
            p["image_id"] = img_id
        crattt_coco = [{
            "image_id":    p["image_id"],
            "category_id": p["category_id"],
            "bbox":        p["bbox"],
            "score":       p["score"]
        } for p in crattt_preds]

        bmap, _ = compute_map(baseline_preds, coco_gt, [img_id])
        cmap, _ = compute_map(crattt_coco, coco_gt, [img_id])

        dynamic_rows.append({
            "Image":           fname,
            "Severity":        sev,
            "Dynamic_Tau":     dynamic_tau,
            "Baseline_mAP":    round(bmap, 4),
            "CRATTT_mAP":      round(cmap, 4),
            "mAP_Improvement": round(cmap - bmap, 4),
            "Baseline_Dets":   len(baseline_res['boxes']),
            "CRATTT_Dets":     stats['n_verified'],
            "Rejected":        stats['n_rejected']
        })

    print(f"  ✅ {fname} complete")

df_dynamic = pd.DataFrame(dynamic_rows)

print("\n" + "="*70)
print("ABLATION: Dynamic Tau Results")
print("="*70)
from IPython.display import display
display(df_dynamic)

print("\n--- Dynamic Tau Summary ---")
for sev in PILOT_SEVERITIES:
    sev_df = df_dynamic[df_dynamic["Severity"] == sev]
    fixed_df = df_pilot[df_pilot["Severity"] == sev]
    print(f"\nSeverity {sev} (τ={get_dynamic_tau(sev)}):")
    print(f"  Dynamic CRATTT mAP  : "
          f"{sev_df['CRATTT_mAP'].mean():.4f}")
    print(f"  Fixed CRATTT mAP    : "
          f"{fixed_df['CRATTT_mAP'].mean():.4f}")
    print(f"  Baseline mAP        : "
          f"{sev_df['Baseline_mAP'].mean():.4f}")
    print(f"  Dynamic improvement : "
          f"{sev_df['mAP_Improvement'].mean():.4f}")

# Save
csv_path = os.path.join(
    EVAL_PARAMS["table_dir"], "table_ablation_dynamic_tau.csv"
)
df_dynamic.to_csv(csv_path, index=False)
print(f"\n✅ Ablation table saved: {csv_path}")

print("\n" + "="*50)
print("BLOCK 11b COMPLETE")
print("="*50)
