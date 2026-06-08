# ============================================================
# BLOCK V6: CLIP Text Embeddings (CORRECTED VERSION)
# Run on: Vast.ai
# Uses text_model directly to avoid HuggingFace wrapper bug
# Verified against Kaggle baseline: person/car sim = 0.8450
# ============================================================

import torch
import torch.nn.functional as F

print("Computing CLIP text embeddings for 80 COCO classes...")
print("Using Transformers CLIPModel — identical to Kaggle")
print()

clip_text_features_list = []
batch_size = 20

for i in range(0, len(COCO_CLASSES), batch_size):
    batch  = COCO_CLASSES[i:i+batch_size]
    inputs = clip_processor(
        text=batch,
        return_tensors="pt",
        padding=True
    ).to(device)
    with torch.no_grad():
        # Use text_model directly — avoids wrapper output bug
        text_outputs = clip_model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        pooled = text_outputs.pooler_output
        feats  = clip_model.text_projection(pooled)
        feats  = F.normalize(feats, p=2, dim=-1)
    clip_text_features_list.append(feats)

clip_text_features = torch.cat(
    clip_text_features_list, dim=0
)
print(f"✅ Shape: {clip_text_features.shape}")

sim_person_car = F.cosine_similarity(
    clip_text_features[0:1],
    clip_text_features[2:3]
).item()

print(f"\n✅ Person/car similarity : {sim_person_car:.4f}")
print(f"   Kaggle baseline       : 0.8450")

if abs(sim_person_car - 0.845) < 0.02:
    print(f"   ✅ MATCH — implementation is consistent")
else:
    print(f"   ⚠️  MISMATCH — difference: "
          f"{abs(sim_person_car - 0.845):.4f}")
    print(f"   Proceed only if difference is < 0.02")
