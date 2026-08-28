# TTRV Gate — Troubleshooting and Validation

The diagnostic session that preceded and justified `../ttrv-gate-demo`.
Before building a standalone, self-contained demo of the TTRV gate, this
notebook checks whether the gate's calibration, taken directly from
`cratt-rq1-new.ipynb` Block 24c (`snr_baseline=32.842`,
`tau_internal=0.9796`), actually discriminates correct detections from
incorrect ones at the box level, not just the coarser category-presence
check used earlier in the pipeline.

## Why this exists

The gate's calibration was originally validated using a category-presence
check: does the predicted label appear anywhere in the image's ground
truth annotations. That's a fast, honest first read, but it's coarser
than the box-level IoU standard the dissertation's own precision figures
are built on: a detection whose category is present can still be
predicting the wrong location entirely. This session re-validates the
gate against that stricter standard, directly, before relying on the
calibration for a public-facing demo.

## What's inside

One continuous session, each section below was originally a separate
notebook cell, run in order:

1. **Setup** — loads GroundingDINO-base, COCO category mapping, the gate
   functions themselves (`b24_run_dino`, `b24_compute_snr`), and the image
   directory.
2. **SNR saturation diagnostic** (N=40) — checks whether the gate's
   signal-to-noise term is actually discriminating, or saturating at its
   cap.
3. **Spot-check against ground truth** — category-presence level, a
   quick first read on a small, deliberately mixed sample (high-confidence,
   low-confidence, near-threshold, and saturated cases).
4. **Box-level correctness check** — the same sample, now checked against
   real IoU ≥ 0.5, the actual standard used elsewhere in the dissertation.
5. **Scaled, randomized correctness check** (N=160) — moves from a
   hand-picked sample to a proper random one, and runs the same Fisher's
   exact test used for the dissertation's own reliability-classifier
   result.
6. **Bug found and fixed**: the initial scaled check included images with
   no matching ground-truth annotation for the predicted category at all,
   silently penalising precision. The corrected version filters to
   annotated images only, matching the original protocol's own
   image-selection rule.
7. **Condition-matched replication** (motion_blur + contrast, severity 5,
   N=160) — reproduces the dissertation's own reported scope exactly, for
   direct comparison against the published numbers.
8. **Per-corruption, per-severity replication** across all four corruption
   types at severities 1, 3, and 5 — twelve conditions in total.
9. **Second bug found and fixed**: the per-corruption loop's image sampler
   and the per-image corruption seeding were sharing the same global RNG,
   so seeding the corruption for one image could silently shift which
   images got sampled for the next corruption type. Fixed by isolating the
   sampler into its own `random.Random` instance.
10. **Focused, larger-N replications** for motion_blur (N=250) and
    gaussian_noise (N=1200) specifically, to confirm the corrected
    results hold at scale, not just in the smaller initial samples.
11. **Checkpointed severity 1 and severity 3 sweeps** (N=250 per
    corruption type each), saving per-corruption progress so an
    interrupted run doesn't need to restart from scratch.

## Relationship to the rest of the repository

This reuses the same gate implementation (`b24_run_dino`, `b24_compute_snr`)
and calibration values as `RQs/CB24c` and the formally-powered results in
`RQs/CB27`–`CB28`, but runs independently, it does not depend on the RQs
folder's setup sequence. Its purpose is narrower and more targeted: confirm
the specific calibration values used in the standalone demo hold up under
the dissertation's actual correctness standard, across the full range of
conditions the demo lets a user select.
