# Experiment 07 — Calibrated Conditional Within-Stratum Alignment & Selector SFT Study

**Doc link:** <https://github.com/brando90/zip-mix/blob/main/experiments/07_calibrated_conditional_sft_zipmix/README.md>

**TLDR:** Two-stage prospective supervised fine-tuning (SFT) study on `Qwen/Qwen2.5-0.5B` (`060db6499f32faf8b98477b0a26969ef7d8b9987`) that (1) calibrates the fine-tuning learning rate on an independent training/validation partition using only the uniform pool baseline (`sample_proportional`) so SFT no longer degrades the base checkpoint, and (2) evaluates within-stratum conditional alignment (`conditional_aligned`, `stratum_uniform`, `conditional_shuffled`) and cross-stratum compression selectors (`zipmix_static`, `shuffled_zipmix`, `wrongtarget_zipmix`, `contrastive_zipmix`, `source_matched`, `direct_zipfit`) at the frozen calibrated learning rate `lr*` on fresh, previously uninspected held-out questions.

## Motivation and design sequence

In Experiments [05](../05_validation_guided_sft/README.md) and [06](../06_contrastive_validation_zipmix/README.md), all 30 fine-tuned `Qwen/Qwen2.5-0.5B` models (`lr=2e-5`, 128 steps × 16 examples = 2,048 answer tokens) degraded science target accuracy (`74.0%–77.9%`) below the unchanged base model (`404/500 = 80.80%`). Per [NEXT_STAGE_DESIGN.md](../04_validation_guided_zipmix/NEXT_STAGE_DESIGN.md), two steps are required in strict sequence:

1. **Stage 1 (`stage1_calibration` — Independent SFT Calibration, 15 cells + 1 base):**
   - Partition 500 clean, 13-gram-decontaminated unused `sciq:train` records (`cal_target`) and 500 clean, 13-gram-decontaminated unused `commonsense_qa:train` records (`cal_retention`) that have zero overlap with the 9,797 candidate pool, 256 wrong-target reserve, 242 development views, or any official validation/test split.
   - Train only `sample_proportional` (`stratum_uniform`, uniform `1/N` over the 9,797-question pool) across five prespecified learning rates (`1e-6`, `2e-6`, `5e-6`, `1e-5`, `2e-5`) × 3 seeds (`0, 1, 2`).
   - Select `lr*` via a locked deterministic rule on `(cal_target, cal_retention)` only: maximize mean `cal_target` accuracy subject to mean `cal_retention` accuracy $\ge \text{base}_{\text{cal\_retention}} - 0.010$.
2. **Stage 2a (`stage2_conditional` — Primary Within-Stratum Conditional Alignment Matrix, 18 cells + 1 base):**
   - Using the frozen `lr*` from Stage 1, train `conditional_aligned`, `stratum_uniform`, and `conditional_shuffled` (pack-rank permutation seed `1707`) across 6 independent training seeds (`0, 1, 2, 3, 4, 5`). All three methods preserve the exact population `source × compression_bin` distribution $M_{s,b}$ (`30.5298%` SciQ mass) within `1e-12`.
   - Evaluate on fresh, previously uninspected official questions: `target` ($N=500$ clean unused `sciq:validation`), `retention` ($N=500$ clean unused `commonsense_qa:validation`), and `test_holdout2` ($N=490$ clean unused `sciq:test`), plus `reused_target` ($N=500$) and `reused_retention` ($N=500$) for historical continuity.
3. **Stage 2b (`stage2_selectors` — Calibrated Cross-Stratum Selector Matrix, 18 cells + 1 base):**
   - Using the same frozen `lr*` from Stage 1, train `zipmix_static`, `shuffled_zipmix`, `wrongtarget_zipmix`, `contrastive_zipmix`, `source_matched`, and `direct_zipfit` across seeds `0, 1, 2` on the same fresh evaluation sets to test whether cross-stratum selectors improve over both the unchanged base model and their matched controls once SFT is calibrated.

## Key files

- [Protocol](expt_v1/PROTOCOL.md)
- [Frozen data manifest](expt_v1/data_manifest.json)
- [Common utilities & selector definitions](expt_v1/common.py)
- [Data preparation](expt_v1/prepare_exp07.py)
- [Two-stage trainer](expt_v1/train_exp07.py)
- [Statistical audit & cross-stage analyzer](expt_v1/analyze_exp07.py)
- [Deterministic CPU test suite](expt_v1/test_exp07.py)
