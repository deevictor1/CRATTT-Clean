# ============================================================
# BLOCK 16: End-to-End CRATTT+TTT Evaluation (Table 4.7)
# Compares three conditions on 20 images across 4 corruptions:
#   A. Baseline DINO (no adaptation)
#   B. CRATTT only (BARON + Oracle TTRV, no TTT)
#   C. CRATTT + TTT (full framework with LoRA updates)
# Produces Table 4.7 — the primary contribution table.
# ============================================================

import torch
import numpy as np
import pandas as pd
import json
import os
from imagecorruptions import corrupt as ic_corrupt
from tqdm.notebook import tqdm
from IPython.display import display

# --- 16.1 Configuration ---
# One representative corruption per category at severity 5
# Primary result — worst-case conditions
EVAL_CORRUPTIONS = [
    ("Noise",   "gaussian_noise", 5),
    ("Blur",    "motion_blur",    5),
    ("Weather", "snow",           5),
    ("Digital", "contrast",       5),
]

TTT_STEPS = 3    # Gradient steps per image
TTT_LR    = 5e-5  # Conservative lr for stable convergence

print("="*50)
print("BLOCK 16: END-TO-END CRATTT+TTT EVALUATION")
print("="*50)
print(f"Images       : {len(image_files)}")
print(f"Corruptions  : {len(EVAL_CORRUPTIONS)}")
print(f"TTT steps    : {TTT_STEPS}")
print(f"TTT lr       : {TTT_LR}")
print(f"Tau          : {CRATTT_PARAMS['tau']}")
print()

# --- 16.2 Fresh Optimizer for Block 16 ---
# Reset optimizer state before full evaluation
lora_parameters = [
    p for p in dino_model.parameters()
    if p.requires_grad
]
optimizer_b16 = torch.optim.AdamW(
    lora_parameters,
    lr=TTT_LR,
    weight_decay=1e-4
)

# --- 16.3 Resume Logic ---
b16_ckpt_dir = os.path.join(
    EVAL_PARAMS["ckpt_dir"], "block16"
)
os.makedirs(b16_ckpt_dir, exist_ok=True)

def load_b16_checkpoint(corruption, severity):
    path = os.path.join(
        b16_ckpt_dir, f"{corruption}_sev{severity}.json"
    )
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def save_b16_checkpoint(corruption, severity, data):
    path = os.path.join(
        b16_ckpt_dir, f"{corruption}_sev{severity}.json"
    )
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

# --- 16.4 Main Evaluation Loop (Fixed) ---
all_rows = []

for cat_name, corruption, severity in EVAL_CORRUPTIONS:

    # Check checkpoint
    ckpt = load_b16_checkpoint(corruption, severity)
    if ckpt is not None:
        print(f"↩️  {corruption} sev{severity}: from checkpoint")
        all_rows.extend(ckpt)
        continue

    print(f"\n▶  {cat_name}: {corruption} sev{severity}")

    # Reset LoRA weights to B=0 before each corruption
    for name, param in dino_model.named_parameters():
        if 'lora_B' in name:
            param.data.zero_()

    # Recreate optimizer fresh — avoids state KeyError
    # on PyTorch 2.10 when resetting between corruptions
    optimizer_b16 = torch.optim.AdamW(
        [p for p in dino_model.parameters() if p.requires_grad],
        lr=TTT_LR,
        weight_decay=1e-4
    )

    corruption_rows = []
    pbar = tqdm(image_files, desc=f"  {corruption}", leave=False)

    for img_path in pbar:
        img_id  = img_id_map[os.path.basename(img_path)]
        raw_img = loaded_images[img_path]
        fname   = os.path.basename(img_path)

        c_img = ic_corrupt(
            raw_img,
            corruption_name=corruption,
            severity=severity
        )

        # === CONDITION A: Baseline DINO ===
        with torch.no_grad():
            inputs = dino_processor(
                images=c_img,
                text=DINO_TEXT_PROMPT,
                return_tensors="pt"
            ).to(device)
            outputs = dino_model(**inputs)
            baseline_res = dino_processor\
                .post_process_grounded_object_detection(
                    outputs, inputs.input_ids,
                    target_sizes=[c_img.shape[:2]],
                    text_threshold=CRATTT_PARAMS["dino_text_thr"]
                )[0]

        baseline_preds = dino_to_coco_format(baseline_res, img_id)
        bmap, _ = compute_map(baseline_preds, coco_gt, [img_id])

        # === CONDITION B: CRATTT only (no TTT) ===
        with torch.no_grad():
            crattt_preds_b, stats_b = run_crattt_inference(c_img)
        for p in crattt_preds_b:
            p["image_id"] = img_id

        crattt_coco_b = [{
            "image_id":    p["image_id"],
            "category_id": p["category_id"],
            "bbox":        p["bbox"],
            "score":       p["score"]
        } for p in crattt_preds_b]

        cmap_b, _ = compute_map(
            crattt_coco_b, coco_gt, [img_id]
        )

        # === CONDITION C: CRATTT + TTT ===
        # Use CRATTT verified preds as pseudo-labels for update
        dino_model.train()
        for step in range(TTT_STEPS):
            optimizer_b16.zero_grad()
            loss = compute_pseudo_label_loss(
                dino_model, dino_processor,
                c_img, crattt_preds_b, device
            )
            if loss is not None:
                loss.backward()
                optimizer_b16.step()
        dino_model.eval()

        # Re-run CRATTT with updated weights
        with torch.no_grad():
            crattt_preds_c, stats_c = run_crattt_inference(c_img)
        for p in crattt_preds_c:
            p["image_id"] = img_id

        crattt_coco_c = [{
            "image_id":    p["image_id"],
            "category_id": p["category_id"],
            "bbox":        p["bbox"],
            "score":       p["score"]
        } for p in crattt_preds_c]

        cmap_c, _ = compute_map(
            crattt_coco_c, coco_gt, [img_id]
        )

        row = {
            "category":            cat_name,
            "corruption":          corruption,
            "severity":            severity,
            "image":               fname,
            "img_id":              img_id,
            "baseline_mAP":        round(float(bmap), 4),
            "crattt_mAP":          round(float(cmap_b), 4),
            "crattt_ttt_mAP":      round(float(cmap_c), 4),
            "delta_crattt":        round(float(cmap_b - bmap), 4),
            "delta_ttt":           round(float(cmap_c - bmap), 4),
            "delta_ttt_vs_crattt": round(float(cmap_c - cmap_b), 4),
            "n_baseline":          len(baseline_res['boxes']),
            "n_crattt":            stats_b['n_verified'],
            "n_ttt":               stats_c['n_verified'],
            "n_rejected_b":        stats_b['n_rejected'],
            "n_rejected_c":        stats_c['n_rejected']
        }
        corruption_rows.append(row)

        pbar.set_postfix({
            "B": f"{bmap:.3f}",
            "C": f"{cmap_b:.3f}",
            "T": f"{cmap_c:.3f}"
        })

    save_b16_checkpoint(corruption, severity, corruption_rows)
    all_rows.extend(corruption_rows)

    c_df = pd.DataFrame(corruption_rows)
    print(f"  Baseline mAP      : {c_df['baseline_mAP'].mean():.4f}")
    print(f"  CRATTT mAP        : {c_df['crattt_mAP'].mean():.4f}")
    print(f"  CRATTT+TTT mAP    : {c_df['crattt_ttt_mAP'].mean():.4f}")
    print(f"  TTT gain vs CRATTT: "
          f"{c_df['delta_ttt_vs_crattt'].mean():+.4f}")

# --- 16.5 Results Table ---
df_final = pd.DataFrame(all_rows)

print("\n" + "="*70)
print("TABLE 4.7: END-TO-END CRATTT+TTT EVALUATION")
print("="*70)

summary = df_final.groupby(
    ["category", "corruption", "severity"]
).agg(
    Baseline_mAP    =("baseline_mAP",    "mean"),
    CRATTT_mAP      =("crattt_mAP",      "mean"),
    CRATTT_TTT_mAP  =("crattt_ttt_mAP",  "mean"),
    Delta_CRATTT    =("delta_crattt",     "mean"),
    Delta_TTT       =("delta_ttt",        "mean"),
    Delta_TTT_vs_C  =("delta_ttt_vs_crattt", "mean"),
    Mean_Rejected_B =("n_rejected_b",     "mean"),
    Mean_Rejected_C =("n_rejected_c",     "mean"),
).round(4).reset_index()

display(summary)

# Overall summary
print(f"\n--- Overall Summary (N={len(image_files)} images) ---")
print(f"Mean Baseline mAP    : "
      f"{df_final['baseline_mAP'].mean():.4f}")
print(f"Mean CRATTT mAP      : "
      f"{df_final['crattt_mAP'].mean():.4f}")
print(f"Mean CRATTT+TTT mAP  : "
      f"{df_final['crattt_ttt_mAP'].mean():.4f}")
print(f"TTT gain vs Baseline : "
      f"{df_final['delta_ttt'].mean():+.4f}")
print(f"TTT gain vs CRATTT   : "
      f"{df_final['delta_ttt_vs_crattt'].mean():+.4f}")

# --- 16.6 Save ---
csv_path = os.path.join(
    EVAL_PARAMS["table_dir"], "table_4_7_end_to_end.csv"
)
df_final.to_csv(csv_path, index=False)

summary_path = os.path.join(
    EVAL_PARAMS["table_dir"], "table_4_7_summary.csv"
)
summary.to_csv(summary_path, index=False)

final_metrics = {
    "mean_baseline_mAP":   round(
        float(df_final['baseline_mAP'].mean()), 4
    ),
    "mean_crattt_mAP":     round(
        float(df_final['crattt_mAP'].mean()), 4
    ),
    "mean_crattt_ttt_mAP": round(
        float(df_final['crattt_ttt_mAP'].mean()), 4
    ),
    "ttt_gain_vs_baseline": round(
        float(df_final['delta_ttt'].mean()), 4
    ),
    "ttt_gain_vs_crattt":  round(
        float(df_final['delta_ttt_vs_crattt'].mean()), 4
    ),
    "n_images":            len(image_files),
    "n_corruptions":       len(EVAL_CORRUPTIONS),
    "lora_rank":           LORA_RANK,
    "lora_alpha":          LORA_ALPHA,
    "ttt_steps":           TTT_STEPS,
    "ttt_lr":              TTT_LR,
    "tau":                 CRATTT_PARAMS["tau"]
}

metrics_path = os.path.join(
    EVAL_PARAMS["save_dir"], "final_metrics.json"
)
with open(metrics_path, "w") as f:
    json.dump(final_metrics, f, indent=2)

print(f"\n✅ Table 4.7 saved: {csv_path}")
print(f"✅ Summary saved:   {summary_path}")
print(f"✅ Metrics saved:   {metrics_path}")

print("\n" + "="*50)
print("BLOCK 16 COMPLETE — End-to-end evaluation done")
print("="*50)
