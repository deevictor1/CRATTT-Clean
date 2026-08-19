# ============================================================
# BLOCK 24e: Statistical Validation — N=50, Severities 1/3/5
# Checkpointed + resumable
# ============================================================
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import os
import gc
from scipy import stats
from tqdm.notebook import tqdm

print("=" * 65)
print("BLOCK 24e: Statistical Validation (N=50, sev 1/3/5, checkpointed)")
print("=" * 65)
print()

required = ["b24_compute_snr", "b24_apply_gate", "b24_confidence_loss",
            "b24_run_dino", "B24", "reset_lora_b", "apply_corruption_deterministic"]
missing = [f for f in required if f not in globals()]
if missing:
    raise RuntimeError(f"Missing: {missing}\nRun the restart sequence above first.")

n_images = len(image_files)
if n_images < 50:
    raise RuntimeError(f"Only {n_images} images — expand the pool first.")

reset_lora_b()
print(f"✅ Dependencies present | N images: {n_images}")
print()

SEVERITIES_TO_TEST = [1, 3, 5]
CORRUPTIONS_E = [
    ("Noise", "gaussian_noise"), ("Blur", "motion_blur"),
    ("Weather", "snow"), ("Digital", "contrast"),
]
TTT_STEPS_E = 5
TTT_LR_E    = 1e-4
N_EVAL      = 50
CLEANUP_EVERY = 20

CHECKPOINT_DIR = "/kaggle/working/results/24e_checkpoints_n50"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def checkpoint_path(severity, corruption):
    return os.path.join(CHECKPOINT_DIR, f"sev{severity}_{corruption}.csv")

print(f"N images      : {N_EVAL}")
print(f"Total runs    : {N_EVAL * len(CORRUPTIONS_E) * len(SEVERITIES_TO_TEST)}")
print()

all_rows_24e = []

for severity in SEVERITIES_TO_TEST:
    print(f"{'='*55}")
    print(f"SEVERITY {severity}")
    print(f"{'='*55}")

    for (cat, corruption) in CORRUPTIONS_E:
        ckpt_path = checkpoint_path(severity, corruption)

        if os.path.exists(ckpt_path):
            df_ckpt = pd.read_csv(ckpt_path)
            all_rows_24e.extend(df_ckpt.to_dict("records"))
            print(f"  ⏭️  {corruption} sev{severity} — checkpoint found, skipping.")
            continue

        cat_rows = []
        for idx, img_path in enumerate(tqdm(image_files[:N_EVAL], desc=f"  {corruption} sev{severity}", leave=False)):
            reset_lora_b()
            opt_24e = torch.optim.AdamW(
                [p for p in dino_model.parameters() if p.requires_grad],
                lr=TTT_LR_E, weight_decay=1e-4)

            img_id  = img_id_map[os.path.basename(img_path)]
            raw_img = loaded_images[img_path]
            c_img   = apply_corruption_deterministic(raw_img, img_id, corruption, severity)

            dino_model.eval()
            boxes_b, scores_b, labels_b = b24_run_dino(c_img)
            preds_base = [
                {"image_id": img_id, "category_id": COCO_MAP[l],
                 "bbox": [b[0].item(), b[1].item(), (b[2]-b[0]).item(), (b[3]-b[1]).item()],
                 "score": s.item()}
                for b, s, l in zip(boxes_b, scores_b, labels_b) if l in COCO_MAP
            ]
            mAP_base, _ = compute_map(preds_base, coco_gt, [img_id])

            dets = b24_compute_snr(c_img)
            v_boxes, v_labels, v_rjoints = b24_apply_gate(dets)

            mAP_ttt   = mAP_base
            ttt_fired = False

            if len(v_boxes) > 0:
                dino_model.train()
                for _ in range(TTT_STEPS_E):
                    opt_24e.zero_grad()
                    loss = b24_confidence_loss(c_img, v_boxes, device)
                    if loss is not None and loss.requires_grad:
                        loss.backward()
                        nn.utils.clip_grad_norm_(
                            [p for p in dino_model.parameters() if p.requires_grad],
                            max_norm=B24["max_norm"])
                        opt_24e.step()
                        ttt_fired = True
                dino_model.eval()

                if ttt_fired:
                    boxes_t, scores_t, labels_t = b24_run_dino(c_img)
                    preds_ttt = [
                        {"image_id": img_id, "category_id": COCO_MAP[l],
                         "bbox": [b[0].item(), b[1].item(), (b[2]-b[0]).item(), (b[3]-b[1]).item()],
                         "score": s.item()}
                        for b, s, l in zip(boxes_t, scores_t, labels_t) if l in COCO_MAP
                    ]
                    mAP_ttt, _ = compute_map(preds_ttt, coco_gt, [img_id])

            cat_rows.append({
                "severity": severity, "category": cat, "corruption": corruption,
                "img_id": img_id,
                "mAP_baseline": float(mAP_base), "mAP_ttt": float(mAP_ttt),
                "vs_baseline": float(mAP_ttt - mAP_base),
                "beats_baseline": bool(mAP_ttt > mAP_base),
                "n_verified": len(v_boxes), "ttt_fired": ttt_fired,
            })

            del opt_24e
            if (idx + 1) % CLEANUP_EVERY == 0:
                gc.collect()
                torch.cuda.empty_cache()

        df_combo = pd.DataFrame(cat_rows)
        df_combo.to_csv(ckpt_path, index=False)
        all_rows_24e.extend(cat_rows)
        print(f"  [{cat:7s} | {corruption:15s} | sev{severity}]  "
              f"vs_baseline: {df_combo['vs_baseline'].mean():+.4f}  |  "
              f"beats: {int(df_combo['beats_baseline'].sum())}/{N_EVAL}  |  "
              f"fired: {int(df_combo['ttt_fired'].sum())}/{N_EVAL}")

    print()

df_24e = pd.DataFrame(all_rows_24e)
print(f"All combinations complete. Total rows: {len(df_24e)}")
print()

def run_statistical_test(df_subset, label):
    diffs = df_subset["mAP_ttt"].values - df_subset["mAP_baseline"].values
    n = len(diffs)
    mean_diff = diffs.mean()
    std_diff  = diffs.std(ddof=1)
    se        = std_diff / np.sqrt(n) if n > 0 else np.nan
    ci_low, ci_high = mean_diff - 1.96 * se, mean_diff + 1.96 * se
    t_stat, t_p = stats.ttest_rel(df_subset["mAP_ttt"], df_subset["mAP_baseline"])
    if np.any(diffs != 0):
        w_stat, w_p = stats.wilcoxon(df_subset["mAP_ttt"], df_subset["mAP_baseline"])
    else:
        w_stat, w_p = np.nan, np.nan
    cohens_d = mean_diff / std_diff if std_diff > 0 else np.nan
    sig_t = "✅ significant" if t_p < 0.05 else "not significant"
    sig_w = "✅ significant" if (not np.isnan(w_p) and w_p < 0.05) else "not significant"
    print(f"{label}")
    print(f"  N = {n}")
    print(f"  Mean vs_baseline : {mean_diff:+.4f}  (95% CI: [{ci_low:+.4f}, {ci_high:+.4f}])")
    print(f"  Paired t-test    : t={t_stat:.3f}, p={t_p:.4f}  ({sig_t})")
    print(f"  Wilcoxon         : W={w_stat:.1f}, p={w_p:.4f}  ({sig_w})")
    print(f"  Cohen's d        : {cohens_d:.3f}")
    print()
    return {"label": label, "n": n, "mean_diff": round(mean_diff, 4),
            "ci_low": round(ci_low, 4), "ci_high": round(ci_high, 4),
            "t_stat": round(t_stat, 3), "t_p": round(t_p, 4),
            "w_stat": round(w_stat, 1) if not np.isnan(w_stat) else None,
            "w_p": round(w_p, 4) if not np.isnan(w_p) else None,
            "cohens_d": round(cohens_d, 3)}

print("=" * 65)
print("STATISTICAL RESULTS")
print("=" * 65)
print()

stat_results = []
for sev in SEVERITIES_TO_TEST:
    print(f"───── Severity {sev}: per-corruption ─────")
    for _, corr in CORRUPTIONS_E:
        df_sub = df_24e[(df_24e["severity"] == sev) & (df_24e["corruption"] == corr)]
        stat_results.append(run_statistical_test(df_sub, f"sev{sev} | {corr}"))
    print(f"───── Severity {sev}: POOLED ─────")
    df_pooled = df_24e[df_24e["severity"] == sev]
    stat_results.append(run_statistical_test(df_pooled, f"sev{sev} | POOLED"))

print(f"───── ALL SEVERITIES + CORRUPTIONS POOLED (N={len(df_24e)}) ─────")
stat_results.append(run_statistical_test(df_24e, "OVERALL POOLED"))

os.makedirs("/kaggle/working/tables", exist_ok=True)
df_24e.to_csv("/kaggle/working/tables/table_block24e_n50_statistical.csv", index=False)
with open("/kaggle/working/results/block24e_n50_statistical_results.json", "w") as f:
    json.dump(stat_results, f, indent=2)

print("✅ Saved: table_block24e_n50_statistical.csv")
print()
print("=" * 65)
print("BLOCK 24e (N=50) COMPLETE")
print("=" * 65)
