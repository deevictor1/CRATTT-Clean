# LVIS Pilot Robustness Evaluation

Extends the ImageNet-C robustness sweep behind Table 4.2a/4.2b (MS-COCO)
to LVIS, a larger, long-tailed dataset with 1,203 categories, to check
whether the GroundingDINO-vs-YOLO-World robustness pattern generalises
beyond COCO specifically. Produces Table 4.2c and the LVIS Pilot decay
figure.

## Federated vs. full-vocabulary querying

LVIS's evaluation protocol is federated: ground truth only marks a
category as present or absent for images actually reviewed for it, not
every category for every image. Querying a detector with all 1,203
category names on every image would be both slow and methodologically
incorrect for this dataset. This pilot instead queries only the
categories LVIS itself annotates as relevant to a given image, the
correct protocol for this dataset, confirmed via a small N=5 smoke test
before committing to the full N=100 run.

A chunked, full-vocabulary alternative (splitting all 1,203 categories
into GroundingDINO-token-limit-sized batches) was also tested in the
smoke test specifically to confirm federated querying was the right
choice, not just the convenient one. It was found non-viable at scale,
extrapolated to over 36 hours at N=100 alone, against federated
querying's few minutes, so it was dropped entirely for the real pilot.

## What's inside

One continuous pilot investigation, in order:

1. **Setup** (Cells 0–2) — installs `lvis-api` (LVIS's own evaluation
   library, the federated-annotation counterpart to `pycocotools`),
   patches several NumPy 2.0 removals that `lvis-api` and
   `imagecorruptions` (both several years old) depend on, and downloads
   LVIS v1 validation annotations.
2. **N=5 smoke test** (Cells 3–9) — a small, deliberately rare-category-
   biased sample, used to confirm the pipeline works end-to-end and to
   directly compare federated vs. chunked full-vocabulary query timing
   before committing to the full pilot scale.
3. **N=100 pilot** (Cells 9–14) — the real evaluation: GroundingDINO-base
   (Swin-B) and YOLO-World, clean and corrupted (all 15 ImageNet-C types
   at 5 severities, 75 configurations total), checkpointed per
   corruption/severity so an interrupted session resumes rather than
   restarting.
4. **Scoring and CE conversion** (Cells 15–17) — scores every
   configuration against LVIS ground truth, then converts to CE/mCE
   using the exact formula and layout as Table 4.2a/4.2b:
   `CE_c = (1 - mean_mAP_c) / (1 - clean_mAP)`, producing Table 4.2c.
5. **Figure generation** (Cells 18–19) — the LVIS Pilot mAP-decay figure,
   in the same four-panel (Noise/Blur/Weather/Digital) style as Figure
   4.1.

## Comparing 4.2c against 4.2a/4.2b

Table 4.2c uses federated querying for both models. Table 4.2a and 4.2b
use the full, fixed 80-category COCO vocabulary for every image. This
makes the two protocols not directly comparable in absolute terms, but
the GroundingDINO-vs-YOLO-World comparison *within* 4.2c is still fair,
since both models are queried identically there.
