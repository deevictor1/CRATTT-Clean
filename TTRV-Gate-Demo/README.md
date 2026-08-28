# TTRV Gate — Standalone Demo

A self-contained, interactive demonstration of the internal TTRV
verification gate (Section 4.5–4.6, formally validated N=2,400 result),
built to run independently of the full CRATTT pipeline.

## How this differs from `RQs/CB33`

`RQs/CB33_ttrv_trust_routing_demo.py` wraps the *live, fully-running*
pipeline, it depends on the entire RQs setup sequence already being in
memory (COCO dataset, CLIP, YOLO-World, category token spans built across
dozens of prior blocks). This folder's demo does not: it loads only
GroundingDINO (Swin-B) and the gate itself, accepts any user-uploaded
image directly, and needs nothing else from the rest of the repository.
It is meant for anyone who wants to see the gate's behaviour firsthand
without first reproducing the full experimental setup.

## What it does

Upload any image, optionally apply an ImageNet-C corruption at a chosen
severity, and see which detections the gate verifies (teal) versus flags
for human review (amber). The verification threshold and signal-to-noise
baseline are fixed to the exact calibration values behind the
dissertation's reported results (`snr_baseline=32.842`,
`tau_internal=0.9796`, from `cratt-rq1-new.ipynb` Block 24c, independently
validated across twelve severity-corruption conditions). Every run writes
`demo_calibration_record.json`, so any session's output can be traced
back to the calibration pass that produced it.

Corruption options deliberately exclude fog, since fog never appears in
any evaluated result in the dissertation and offering it here would
imply a validated behaviour that doesn't exist.

## Running it

This is a single, self-contained script (`ttrv_gate_demo.py`). It runs
on Kaggle (as originally developed) or in any environment with GPU
access and the listed dependencies (`transformers`, `torch`,
`torchvision`, `imagecorruptions`, `gradio`, `Pillow`). It does not
require a Kaggle account or HuggingFace token specifically, if neither
is available, the script logs a notice and continues without them,
since GroundingDINO-base is a public model.

```
pip install imagecorruptions gradio
python ttrv_gate_demo.py
```

Launches a Gradio interface with a public share link
(`demo.launch(share=True)`).

## What this demo is, and isn't

This illustrates the gate's behaviour qualitatively on individual
images. It is not a reproduction of the formal statistical test itself,
that lives in `RQs/CB24e` (the N=2,400 adaptation result) and
`RQs/CB27`–`CB28` (the N=2,624 reliability-classifier result). Detections
whose consensus score sits within roughly 1% of the gate threshold may
occasionally shift between verified and flagged across runs due to GPU
floating-point precision, a known property of GPU inference generally,
not an instability in the gate's design.
