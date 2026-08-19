# CRATTT-Clean
Clean implementation of CRATTT framework - MRES7015 Dissertation

===================================================================================================

# RQ1 — Code Archive

This folder contains the working code behind the dissertation's investigation into
whether reasoning-aware verification (TTRV) can enable safe test-time adaptation for
open-vocabulary object detection under image corruption.

**This folder is not just "the answer to RQ1."** It contains the full research
arc that RQ1 sits inside: the baseline robustness characterisation that the whole
project builds on, the external-oracle approach that was tried first, the twelve-
configuration ablation that diagnosed why it failed, the internal-TTRV mechanism
that replaced it, an alternative training-free approach explored alongside it, and
the reliability validation that closes out the investigation. Each stage's code is
kept here as it was actually run, including the debugging and dead ends, not
cleaned up after the fact to look like a straight line.

Files are numbered `CB<n>` (Code Block n) in the order they were run. Files with a
letter suffix (`CB16b`, `CB24d`, etc.) are follow-on cells for the same numbered
block, usually a fix, a variant, or a scale-up of what came before. A few blocks
sit between numbers where a fix or pre-flight check was inserted after the fact
(e.g. `CB16d` sits between Blocks 16 and 17, `CB23d`/`CB23e` sit between Blocks 23
and 24) — these still belong in the position their name implies.

## How to use this folder

Run the files in numeric order (CB1 → CB33), sub-letters in the order shown below.
Each file is self-contained enough to read on its own, but most depend on objects
defined in earlier files (the model, the corruption helpers, the COCO ground
truth) being already in memory — this was written and run as a sequence of Kaggle
notebook cells, not as standalone scripts. If you're re-running this, expect to
run it in one continuous session (or restore the relevant state) rather than
executing files individually out of order.

---

## 1. Setup and Baseline Robustness (CB1–CB10c)

Environment setup, model loading, and the full ImageNet-C corruption sweep
establishing GroundingDINO vs YOLO-World's clean-accuracy vs. robustness
separability (Tables 4.1–4.2). Also includes BARON (region extraction) and the
CLIP Oracle (semantic verification) — the two components later combined into the
external-oracle CRATTT gate.

| File | What it does |
|---|---|
| `CB1_environment_setup.py` | Initial environment and dependency setup |
| `CB1b_numpy_patch.py` | NumPy/pandas version compatibility patch |
| `CB2_load_models.py` | Loads GroundingDINO and YOLO-World |
| `CB3_define_global_constants.py` | Shared constants (prompts, thresholds, paths) |
| `CB4_data_loading_coco_setup.py` | COCO image loading and ground truth setup |
| `CB5_patched_dino_extraction.py` | Patched category-argmax extraction (fixes the composite-phrase silent-drop bug) |
| `CB6_full_corruption_sweep.py` | Full 15-corruption × 5-severity sweep — produces Table 4.2a |
| `CB7_mCE_calculation_and_table.py` | mCE calculation and summary table/figure |
| `CB8_Baron_Region_Extractor.py` | BARON region-crop extractor, validated on a clean image |
| `CB8b_Baron_Spatial_Integrity.py` | BARON spatial-integrity evaluation under corruption |
| `CB9_compute_soracle.py` | CLIP Oracle semantic verification (Soracle) |
| `CB10_external_oracle_inference_function.py` | Canonical external-oracle CRATTT inference function |
| `CB10b_rjoint_calibration_diagnostic.py` | Rjoint distribution calibration diagnostic |
| `CB10c_empirically_calibrated_tau.py` | Empirically calibrated verification threshold τ |

## 2. Phase 2 — External-Oracle TTRV (CB11–CB13)

Tests the CLIP-Oracle-gated approach: does verification improve precision, and
does gating produce a net mAP improvement? Produces Tables 4.3–4.6.

| File | What it does |
|---|---|
| `CB11_ttrv_pilot_evaluation.py` | Small-scale pilot comparing baseline vs. gated CRATTT |
| `CB_11b_dynamic_tau_ablation.py` | Severity-adaptive threshold ablation |
| `CB11C_precision_recall_diagnostic.py` | Precision-proxy analysis — Table 4.4 |
| `CB12_crattt_baseline_comparative.py` | Full comparative sweep — Table 4.5 |
| `CB13_visual_comparison_gallery.py` | Visual gallery of baseline vs. gated detections |

## 3. Phase 3 — RAP Ablation (CB14–CB23e)

The twelve-configuration ablation diagnosing why external-oracle-gated LoRA
adaptation consistently failed to produce a positive result — the empirical basis
for the Representational Asymmetry Problem (RAP) characterisation.

| File | What it does |
|---|---|
| `CB14_lora_adapter_injection.py` | LoRA adapter injection (rank=4) into GroundingDINO |
| `CB14b_verification_cell.py` | Post-injection parameter-count verification |
| `CB15_ttt_update_loop.py` | Test-time training update loop and loss functions |
| `CB16_crattt_ttt_evaluation.py` | Config 1: rank=4, confidence loss, frozen oracle |
| `CB16b_ttt_with_alignment_loss.py` | Config 2: rank=4, alignment loss, frozen oracle |
| `CB16b2_pre_flight_for_16c.py` | Pre-flight state check before Config 3 |
| `CB16C_higher_rank_lora.py` | Config 3: rank=16, alignment loss, frozen oracle |
| `CB16d_revert_lora_rank.py` | Reverts model to rank=4 before continuing |
| `CB17_co_adaptation_pilot.py` | Config 4: rank=4, co-adapted CLIP oracle |
| `CB18_encoder_feature_gate.py` | Config 5: encoder-feature gate (untrained projection) |
| `CB19_detection_head_lora.py` | Config 6: detection-head LoRA, N=5 |
| `CB19b_detection_head_lora.py` | Config 6 continued: higher lr/steps |
| `CB19c_lora_scale_validation.py` | Config 6 scale check, N=20 |
| `CB20_self_supervised_consistency_ttt.py` | Config 7: self-supervised consistency, no oracle |
| `CB21_encoder_gate_ttt.py` | Config 8: trained cross-modal projection + encoder gate |
| `CB22_oracle_gated_multi_view-pseudo_labelling.py` | Config 9: oracle-gated multi-view pseudo-labelling |
| `CB22b_relaxed_consistency_threshold.py` | Diagnostic: relaxed-parameter pseudo-GT quality check |
| `CB23_task_specific_adapter_head.py` | Config 10: task-specific adapter head, N=5 |
| `CB23b_adapter_head_ttt.py` | Config 10 scale check, N=20 |
| `CB23c_larger_adapter.py` | Config 10 variant: larger adapter, higher lr |
| `CB23d_restore_state-before_block_24.py` | Restores clean LoRA/τ state before Phase 4 |
| `CB23e_block_24_preflight.py` | Pre-flight check before Block 24 |

## 4. Phase 4 — Internal TTRV (CB24–CB24f)

Replaces the frozen external CLIP Oracle with an internally-computed multi-view
Score-SNR signal, resolving the representational mismatch RAP identified. Includes
the debugging history behind this mechanism — four independently diagnosed
validity threats, documented as Section 4.1.5 of the dissertation — and the
formally-powered N=2,400 statistical validation (Table 4.10).

| File | What it does |
|---|---|
| `CB24_internal_ttrv.py` | Internal TTRV mechanism, first pass |
| `CB24a_pre_patched_b24_run_dino.py` | Patched extraction applied for all subsequent Phase 4+ blocks |
| `CB24b_internal_ttrv_fixed_two_pass_calibration.py` | Fixes calibration scale mismatch + episodic state leakage |
| `CB24c_internal_ttrv_fixed_calibration.py` | Adds deterministic corruption seeding |
| `CB24d_fix_for_episodic_reset.py` | Fixes the fallback-to-empty-set logic error; N=20 severity sweep |
| `CB24d2_expand_image_pool_to_50.py` | Expands image pool for the N=50 statistical run |
| `CB24e_statistical_validation.py` | Formally-powered N=200-per-condition validation (Table 4.10) |
| `CB24f_detection_loss_ttt.py` | Detection-loss TTT diagnostic (Table 4.11) |

## 5. Phase 5 — Training-Free Confidence Recalibration (CB25)

An alternative approach explored alongside TTT: rescaling detection confidence
using an online memory of attribute-level vulnerability, with no weight updates.
Produces Table 4.12.

| File | What it does |
|---|---|
| `CB25_training_free_confidence_recalibration.py` | Full streamlined pipeline: calibration, online memory, held-out validation |

## 6. Phase 6 — Reliability Classifier Validation (CB26–CB33)

Validates the TTRV gate as a standalone reliability classifier: does verification
status actually predict detection correctness? Includes the formally-powered
significance testing (Fisher's exact test, Cohen's h) underpinning the
dissertation's primary positive contribution, plus a stratified breakdown by
severity and corruption, a continuous-score discrimination analysis, and an
interactive Gradio demo of the gate itself.

| File | What it does |
|---|---|
| `CB26_preflight_for_block_27.py` | Pre-flight check + idempotent patch reapplication |
| `CB27_ttrv_reliability_classifier.py` | Precision analysis: verified vs. rejected detections, N=20 |
| `CB28_gate_relaibility_analysis_200_images.py` | Same analysis at N=200, with significance testing |
| `CB29_preflight_for_block_30.py` | Pre-flight check + backup archive before the final analysis |
| `CB30_stratified_reliability_analysis.py` | Precision gap broken down by severity × corruption |
| `CB31_continuous_reliability_analysis.py` | Rjoint as a continuous predictor — AUC-ROC, calibration curve |
| `CB32_patch_and_preflight_check.py` | Patch reapplication + Gradio demo pre-flight |
| `CB33_ttrv_trust_routing_demo.py` | Interactive Gradio demo of the validated TTRV gate |

---

## Where the key numbers come from

- **Table 4.2 (robustness)**: `CB6` + `CB7`
- **Tables 4.3–4.6 (external-oracle TTRV)**: `CB11`–`CB13`
- **Table 4.8 (RAP ablation, 12 configs)**: `CB16`–`CB23c`
- **Table 4.9 (methodological corrections)**: `CB24`–`CB24d`
- **Table 4.10 (formally-powered N=2,400 result)**: `CB24e`
- **Table 4.11 (detection-loss diagnostic)**: `CB24f`
- **Table 4.12 (online-memory recalibration)**: `CB25`
- **TTRV reliability classifier finding (verified vs. rejected precision)**: `CB27`, `CB28`
