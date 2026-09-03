# CRATTT-Clean

Clean implementation of the CRATTT framework - MRES7015 Dissertation.

This repository is organised into five folders, each independently
runnable and independently documented. This page is a short map between
them; each folder's own README has the full breakdown down to individual
code blocks, and, where relevant, exactly which table or figure each
block produced.

| Folder | What it contains |
|---|---|
| [`RQs`](./RQs) | The complete internal-TTRV investigation: baseline robustness characterisation, external-oracle TTRV, the twelve-configuration ablation and the Representational Asymmetry Problem, internal TTRV and its formally-powered N=2,400 validation, training-free adaptation, the TTRV reliability-classifier analysis, and the phases addressing RQ2 and RQ3. |
| [`vastai`](./vastai) | The full-scale (N=5,000) baseline robustness sweep underlying Table 4.2b, run on Vast.ai rather than Kaggle due to sustained GPU requirements beyond Kaggle's session limits. |
| [`TTRV-Gate-Demo`](./TTRV-Gate-Demo) | A self-contained, interactive Gradio demo of the internal TTRV gate. Runs independently of the rest of the repository, no COCO dataset, CLIP, or YOLO-World required, accepts any user-uploaded image directly. |
| [`ttrv-gate-troubleshooting`](./ttrv-gate-troubleshooting) | The diagnostic session that preceded and justified the standalone demo's calibration values, re-validating the gate at the box-level IoU standard the dissertation's own precision figures use, not just the coarser category-presence check used earlier in the pipeline. |
| [`lvis-pilot`](./lvis-pilot) | Extends the ImageNet-C robustness sweep to LVIS (1,203 categories, long-tailed), producing Table 4.2c and the LVIS Pilot decay figure, to check whether the GroundingDINO-vs-YOLO-World robustness pattern generalises beyond COCO. |

## Where the dissertation's key numbers come from, across the whole repository

- **Table 4.1 / 4.2a (baseline robustness, pilot N=20)**: `RQs/CB6` + `CB7`
- **Table 4.2b (baseline robustness, full scale N=5,000)**: `vastai`
- **Table 4.2c (LVIS pilot)**: `lvis-pilot`
- **Tables 4.3–4.6 (external-oracle TTRV)**: `RQs/CB11`–`CB13`
- **Table 4.8 (RAP ablation, 12 configs)**: `RQs/CB16`–`CB23c`
- **Table 4.9 (methodological corrections)**: `RQs/CB24`–`CB24d`
- **Table 4.10 (formally-powered N=2,400 result)**: `RQs/CB24e`
- **Table 4.11 (detection-loss diagnostic)**: `RQs/CB24f`
- **Table 4.12 (online-memory recalibration)**: `RQs/CB25`
- **TTRV reliability classifier finding (verified vs. rejected precision)**: `RQs/CB27`, `CB28`
- **Table 4.17 (adaptation speed, LoRA vs. full fine-tune)**: `RQs/CB35`
- **Interactive demonstration of the TTRV gate**: `TTRV-Gate-Demo`
