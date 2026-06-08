# ============================================================
# BLOCK V10: Main Sweep — Both Models
# Run on: Vast.ai
# Includes fog fix: uint8 conversion before corruption
# Checkpoints per severity — safe to interrupt at any point
# COCOeval runs ONCE per configuration on ALL predictions
# ============================================================

import torch
import numpy as np
import pandas as pd
import json
import os
from imagecorruptions import corrupt as ic_corrupt
from tqdm import tqdm

CLEAN_MAP = CLEAN_MAP_DINO if MODEL == "dino" else CLEAN_MAP_YOLO

all_session_results = []
config_count        = 0
total_configs_run   = len(THIS_RUN) * len(SEVERITIES)

print(f"Starting: {RUN_NAME}")
print(f"Model   : {MODEL.upper()}")
print(f"Configs : {total_configs_run}")
print(f"Images  : {len(image_files)} per config")
print()

for cat_name, corruption in THIS_RUN:
    for severity in SEVERITIES:
        config_count += 1
        label = (f"[{config_count}/{total_configs_run}] "
                 f"{MODEL.upper()} {corruption} sev{severity}")

        # Resume from valid checkpoint
        ckpt = load_ckpt(MODEL, corruption, severity)
        if ckpt is not None and ckpt.get('n_images', 0) > 0:
            print(f"↩️  {label}: "
                  f"mAP={ckpt['map']:.4f} "
                  f"CE={ckpt['ce']:.4f} "
                  f"({ckpt['n_images']} images)")
            all_session_results.append(ckpt)
            continue

        print(f"\n▶  {label}")

        all_preds   = []
        all_img_ids = []
        n_errors    = 0

        pbar = tqdm(
            image_files,
            desc=f"  {corruption}_sev{severity}",
            leave=False
        )

        for img_path in pbar:
            fname  = os.path.basename(img_path)
            img_id = img_id_map[fname]
            raw    = loaded_images[img_path]

            # FOG FIX: ensure uint8 before applying corruption
            # This fixes the imagecorruptions fog failure on
            # certain COCO image formats
            raw_uint8 = np.array(raw, dtype=np.uint8)

            try:
                c_img = ic_corrupt(
                    raw_uint8,
                    corruption_name=corruption,
                    severity=severity
                )
            except Exception:
                n_errors += 1
                continue

            try:
                if MODEL == "dino":
                    with torch.no_grad():
                        inputs = dino_processor(
                            images=c_img,
                            text=DINO_TEXT_PROMPT,
                            return_tensors="pt"
                        ).to(device)
                        outputs = dino_model(**inputs)
                        res = dino_processor\
                            .post_process_grounded_object_detection(
                                outputs,
                                inputs.input_ids,
                                target_sizes=[c_img.shape[:2]],
                                text_threshold=DINO_TEXT_THR
                            )[0]
                    preds = dino_to_coco_format(res, img_id)

                elif MODEL == "yolo":
                    import cv2
                    c_bgr = cv2.cvtColor(
                        c_img, cv2.COLOR_RGB2BGR
                    )
                    with torch.no_grad():
                        results = yolo_model.predict(
                            c_bgr,
                            conf=YOLO_CONF_THR,
                            verbose=False
                        )
                    preds = yolo_to_coco_format(
                        results, img_id
                    )

                all_preds.extend(preds)
                all_img_ids.append(img_id)

            except Exception as e:
                n_errors += 1
                continue

        # COCOeval ONCE on all predictions
        print(f"  COCOeval: {len(all_img_ids)} images, "
              f"{len(all_preds)} predictions...")

        map_val = compute_map_coco(
            all_preds, coco_gt, all_img_ids
        )

        clean_err   = 1.0 - CLEAN_MAP
        corrupt_err = 1.0 - map_val
        ce_val = corrupt_err / clean_err \
                 if clean_err > 0 else 0.0

        result = {
            "model":      MODEL,
            "run_name":   RUN_NAME,
            "category":   cat_name,
            "corruption": corruption,
            "severity":   severity,
            "n_images":   len(all_img_ids),
            "n_preds":    len(all_preds),
            "n_errors":   n_errors,
            "map":        round(map_val, 4),
            "ce":         round(ce_val,  4),
        }

        save_ckpt(MODEL, corruption, severity, result)
        all_session_results.append(result)

        print(f"  ✅ mAP={map_val:.4f} | "
              f"CE={ce_val:.4f} | "
              f"N={len(all_img_ids)} | "
              f"Errors={n_errors}")

        df_r = pd.DataFrame(all_session_results)
        df_r.to_csv(
            os.path.join(
                TABLES_DIR, f"{RUN_NAME}_running.csv"
            ),
            index=False
        )

print(f"\n{'='*50}")
print(f"SWEEP COMPLETE: {RUN_NAME}")
print(f"Configurations : {len(all_session_results)}")
print(f"{'='*50}")
