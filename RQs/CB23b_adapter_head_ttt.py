# ============================================================
# BLOCK 23b: Adapter Head TTT — Scale to N=20
# Confirms Block 23 result holds at full eval scale.
# Same configuration. All 20 images. 4 corruptions.
# ============================================================

import torch
import torch.nn as nn  
import numpy as np
import pandas as pd
import json
import os
from PIL import Image as PILImage
from imagecorruptions import corrupt as ic_corrupt
from tqdm.notebook import tqdm
from IPython.display import display

print("="*55)
print("BLOCK 23b: ADAPTER HEAD TTT — N=20 SCALE VALIDATION")
print("="*55)
print(f"Configuration: identical to Block 23")
print(f"Scale: {len(image_files)} images (was 5)")
print()

EVAL_CORRUPTIONS_23B = [
    ("Noise",   "gaussian_noise", 5),
    ("Blur",    "motion_blur",    5),
    ("Weather", "snow",           5),
    ("Digital", "contrast",       5),
]
TTT_STEPS_23B = 10
TTT_LR_23B    = 1e-3

b23b_ckpt_dir = os.path.join(
    EVAL_PARAMS["ckpt_dir"], "block23b"
)
os.makedirs(b23b_ckpt_dir, exist_ok=True)

all_rows_23b = []

for cat_name, corruption, severity in EVAL_CORRUPTIONS_23B:

    ckpt_path = os.path.join(
        b23b_ckpt_dir, f"{corruption}_sev{severity}.json"
    )
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            rows = json.load(f)
        print(f"↩️  {corruption}: from checkpoint")
        all_rows_23b.extend(rows)
        continue

    print(f"\n▶  {cat_name}: {corruption} sev{severity}")

    # Reset adapter to identity
    nn.init.zeros_(adapter.net[-1].weight)
    nn.init.zeros_(adapter.net[-1].bias)

    optimizer_23b = torch.optim.AdamW(
        adapter.parameters(),
        lr=TTT_LR_23B,
        weight_decay=1e-4
    )

    corruption_rows = []
    pbar = tqdm(
        image_files, desc=f"  {corruption}", leave=False
    )

    for img_path in pbar:
        img_id  = img_id_map[os.path.basename(img_path)]
        raw_img = loaded_images[img_path]
        fname   = os.path.basename(img_path)

        c_img = ic_corrupt(
            raw_img, corruption_name=corruption, severity=severity
        )

        # Baseline
        with torch.no_grad():
            inp = dino_processor(
                images=c_img, text=DINO_TEXT_PROMPT,
                return_tensors="pt"
            ).to(device)
            out      = dino_model(**inp)
            base_res = dino_processor\
                .post_process_grounded_object_detection(
                    out, inp.input_ids,
                    target_sizes=[c_img.shape[:2]],
                    text_threshold=CRATTT_PARAMS["dino_text_thr"]
                )[0]
        base_preds = dino_to_coco_format(base_res, img_id)
        bmap, _    = compute_map(base_preds, coco_gt, [img_id])

        # Adapter CRATTT — no TTT
        adapter.eval()
        with torch.no_grad():
            crattt_b, stats_b = run_crattt_with_adapter(
                c_img, adapter
            )
        for p in crattt_b:
            p["image_id"] = img_id
        coco_b = [{
            "image_id": p["image_id"],
            "category_id": p["category_id"],
            "bbox": p["bbox"], "score": p["score"]
        } for p in crattt_b]
        cmap_b, _ = compute_map(coco_b, coco_gt, [img_id])

        # Pseudo-GT (Block 23's strict-params version, still in
        # scope from the same kernel session)
        pseudo_gt_boxes, pseudo_gt_labels = get_pseudo_gt(c_img)
        n_pseudo_gt = len(pseudo_gt_boxes)

        # TTT update
        loss_values = []
        adapter.train()
        if n_pseudo_gt > 0:
            for step in range(TTT_STEPS_23B):
                optimizer_23b.zero_grad()
                loss = compute_adapter_loss(
                    adapter, dino_model, dino_processor,
                    c_img, pseudo_gt_boxes,
                    pseudo_gt_labels, device
                )
                if loss is not None:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        adapter.parameters(), max_norm=1.0
                    )
                    optimizer_23b.step()
                    loss_values.append(round(loss.item(), 6))
        adapter.eval()

        # Re-run with adapted adapter
        with torch.no_grad():
            crattt_c, stats_c = run_crattt_with_adapter(
                c_img, adapter
            )
        for p in crattt_c:
            p["image_id"] = img_id
        coco_c = [{
            "image_id": p["image_id"],
            "category_id": p["category_id"],
            "bbox": p["bbox"], "score": p["score"]
        } for p in crattt_c]
        cmap_c, _ = compute_map(coco_c, coco_gt, [img_id])

        wt_change = adapter.net[-1].weight.abs().max().item()

        row = {
            "category":            cat_name,
            "corruption":          corruption,
            "image":               fname,
            "baseline_mAP":        round(float(bmap),   4),
            "adapter_crattt_mAP":  round(float(cmap_b), 4),
            "adapter_ttt_mAP":     round(float(cmap_c), 4),
            "delta_ttt_vs_crattt": round(float(cmap_c - cmap_b), 4),
            "delta_ttt_vs_base":   round(float(cmap_c - bmap),   4),
            "n_pseudo_gt":         n_pseudo_gt,
            "ttt_fired":           n_pseudo_gt > 0,
            "loss_mean":           round(float(np.mean(loss_values))
                                         if loss_values else 0, 4),
            "weight_change":       round(float(wt_change), 6)
        }
        corruption_rows.append(row)
        pbar.set_postfix({
            "B": f"{bmap:.3f}",
            "C": f"{cmap_b:.3f}",
            "T": f"{cmap_c:.3f}",
            "Δ": f"{cmap_c-cmap_b:+.3f}"
        })

    with open(ckpt_path, "w") as f:
        json.dump(corruption_rows, f, indent=2)
    all_rows_23b.extend(corruption_rows)

    c_df = pd.DataFrame(corruption_rows)
    print(f"  Baseline mAP       : {c_df['baseline_mAP'].mean():.4f}")
    print(f"  Adapter CRATTT     : {c_df['adapter_crattt_mAP'].mean():.4f}")
    print(f"  Adapter TTT mAP    : {c_df['adapter_ttt_mAP'].mean():.4f}")
    print(f"  TTT gain vs CRATTT : {c_df['delta_ttt_vs_crattt'].mean():+.4f}")
    print(f"  TTT fired          : {c_df['ttt_fired'].sum()}/{len(c_df)}")
    print(f"  Mean pseudo-GT     : {c_df['n_pseudo_gt'].mean():.2f}")

df_23b = pd.DataFrame(all_rows_23b)
overall_23b  = df_23b['delta_ttt_vs_crattt'].mean()
positive_23b = (df_23b['delta_ttt_vs_crattt'] > 0.001).sum()
ttt_fired_23b = df_23b['ttt_fired'].sum()

print("\n" + "="*65)
print(f"BLOCK 23b: ADAPTER TTT N=20 RESULTS")
print("="*65)
summary_23b = df_23b.groupby(["category","corruption"]).agg(
    Baseline   =("baseline_mAP",        "mean"),
    Adapter_C  =("adapter_crattt_mAP",  "mean"),
    Adapter_TTT=("adapter_ttt_mAP",     "mean"),
    TTT_gain   =("delta_ttt_vs_crattt", "mean"),
    PseudoGT   =("n_pseudo_gt",         "mean"),
    Fired      =("ttt_fired",           "sum"),
).round(4).reset_index()
display(summary_23b)

print(f"\nOverall TTT gain    : {overall_23b:+.4f}")
print(f"Images with gain>0  : {positive_23b}/{len(df_23b)}")
print(f"TTT fired           : {ttt_fired_23b}/{len(df_23b)}")

# FIX: read Block 23's N=5 result live (overall_23, still in
# memory from this session) instead of the hardcoded "+0.0043"
print(f"\n--- Scale Comparison ---")
if 'overall_23' in globals():
    print(f"Block 23  N=5  gain: {overall_23:+.4f}")
else:
    print(f"Block 23  N=5  gain: N/A (re-run Block 23 to compare)")
print(f"Block 23b N=20 gain: {overall_23b:+.4f}")

if overall_23b > 0.005:
    print("\n✅ CONFIRMED — Positive gain holds at N=20")
    print("   Proceed to Vast.ai for full-scale evaluation")
elif overall_23b > 0.001:
    print("\n✅ HOLDS — Smaller but positive at N=20")
    print("   Report as confirmed positive pilot result")
else:
    print("\n⚠️  N=5 result did not generalize to N=20")
    print("   Report N=5 as preliminary, needs more scale")

csv_23b = os.path.join(
    EVAL_PARAMS["table_dir"], "table_block23b_n20.csv"
)
df_23b.to_csv(csv_23b, index=False)
print(f"\n✅ Saved: {csv_23b}")
print("\n" + "="*50)
print("BLOCK 23b COMPLETE")
print("="*50)
