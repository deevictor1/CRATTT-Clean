# Block 23c: Larger adapter, higher lr
print("Testing larger adapter (hidden=256, lr=5e-3)...")

# Create larger adapter
adapter_large = DetectionAdapterHead(hidden_dim=256).to(device)
large_params  = sum(p.numel() for p in adapter_large.parameters())
print(f"Adapter params: {large_params:,} (was 2,821)")

# Reset
nn.init.zeros_(adapter_large.net[-1].weight)
nn.init.zeros_(adapter_large.net[-1].bias)

opt_large = torch.optim.AdamW(
    adapter_large.parameters(), lr=5e-3, weight_decay=1e-4
)

# FIX: pick the best-performing corruption live from df_23b instead
# of a typed-in assumption. Falls back to gaussian_noise (this
# session's previously observed best) only if df_23b isn't in memory.
if 'df_23b' in globals():
    corruption_means = df_23b.groupby('corruption')['delta_ttt_vs_crattt'].mean()
    corruption = corruption_means.idxmax()
    best_gain = corruption_means.max()
    print(f"Best-performing corruption from live Block 23b results: "
          f"{corruption} ({best_gain:+.4f})")
    print("Per-corruption means:")
    for corr, val in corruption_means.sort_values(ascending=False).items():
        print(f"  {corr:<20} {val:+.4f}")
else:
    corruption = 'gaussian_noise'
    print("⚠️  df_23b not in memory — defaulting to gaussian_noise "
          "(this session's previously observed best). Re-run Block 23b "
          "first to confirm this pick is still correct.")

severity = 5
gains = []
wt_changes = []

for img_path in image_files[:10]:
    img_id  = img_id_map[os.path.basename(img_path)]
    raw_img = loaded_images[img_path]
    c_img   = ic_corrupt(raw_img, corruption_name=corruption,
                          severity=severity)

    # Baseline
    with torch.no_grad():
        inp = dino_processor(
            images=c_img, text=DINO_TEXT_PROMPT,
            return_tensors="pt"
        ).to(device)
        out = dino_model(**inp)
        br  = dino_processor\
            .post_process_grounded_object_detection(
                out, inp.input_ids,
                target_sizes=[c_img.shape[:2]],
                text_threshold=CRATTT_PARAMS["dino_text_thr"]
            )[0]
    bpreds = dino_to_coco_format(br, img_id)
    bmap, _ = compute_map(bpreds, coco_gt, [img_id])

    # CRATTT no TTT
    adapter_large.eval()
    with torch.no_grad():
        cb, _ = run_crattt_with_adapter(c_img, adapter_large)
    for p in cb: p["image_id"] = img_id
    coco_b = [{"image_id":p["image_id"],"category_id":p["category_id"],
               "bbox":p["bbox"],"score":p["score"]} for p in cb]
    cmap_b, _ = compute_map(coco_b, coco_gt, [img_id])

    # Get pseudo-GT
    pg_boxes, pg_labels = get_pseudo_gt(c_img)

    # TTT
    adapter_large.train()
    if len(pg_boxes) > 0:
        for _ in range(10):
            opt_large.zero_grad()
            loss = compute_adapter_loss(
                adapter_large, dino_model, dino_processor,
                c_img, pg_boxes, pg_labels, device
            )
            if loss is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    adapter_large.parameters(), max_norm=1.0
                )
                opt_large.step()
    adapter_large.eval()

    # After TTT
    with torch.no_grad():
        cc, _ = run_crattt_with_adapter(c_img, adapter_large)
    for p in cc: p["image_id"] = img_id
    coco_c = [{"image_id":p["image_id"],"category_id":p["category_id"],
               "bbox":p["bbox"],"score":p["score"]} for p in cc]
    cmap_c, _ = compute_map(coco_c, coco_gt, [img_id])

    delta = round(float(cmap_c - cmap_b), 4)
    wt    = adapter_large.net[-1].weight.abs().max().item()
    gains.append(delta)
    wt_changes.append(wt)

mean_gain = float(np.mean(gains))
mean_wt   = float(np.mean(wt_changes))

print(f"\n{corruption} sev5 N=10:")
print(f"  Mean gain vs CRATTT  : {mean_gain:+.4f}")
print(f"  Mean weight change   : {mean_wt:.6f}")
print(f"  Per-image gains      : {gains}")
print()

if mean_gain > 0.005:
    print("✅ Larger adapter works — worth a full N=20 sweep")
elif mean_gain > 0.001:
    print("✅ Marginal improvement — confirm at N=20")
else:
    print("⚠️  No improvement even with larger adapter on this run's")
    print(f"   best-performing corruption ({corruption}).")
    print("   This is one of the later configs in Table 4.8 —")
    print("   worth checking what blocks remain before")
    print("   treating this as final.")
