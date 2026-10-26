# Experiment 07 Version 1 — Prospective Two-Stage SFT Calibration and Conditional Alignment Protocol

**Doc link:** <https://github.com/brando90/zip-mix/blob/main/experiments/07_calibrated_conditional_sft_zipmix/expt_v1/PROTOCOL.md>

Date: 10-16-2026. Condition: `07_calibrated_conditional_sft_zipmix/expt_v1`.

## 1. Scientific objective and strict information boundary

Experiments 05 (`expt_v2`) and 06 (`expt_v1`) showed that at `learning_rate = 2e-5` (full float32 fine-tuning for 128 steps × 16 examples = 2,048 answer-label tokens), all 30 fine-tuned `Qwen/Qwen2.5-0.5B` models degraded target SciQ accuracy (`74.0%–77.9%`) below the unchanged base checkpoint (`404/500 = 80.80%`). Per [NEXT_STAGE_DESIGN.md](../../04_validation_guided_zipmix/NEXT_STAGE_DESIGN.md), two requirements govern this follow-up:

1. **Independent SFT calibration before selector comparison:** Allocate a separate training/validation split to calibrate the learning rate against the unchanged base checkpoint using only the neutral pool baseline (`sample_proportional` / `stratum_uniform`), without reading or tuning on any held-out evaluation split and without favoring any selector.
2. **Fresh held-out evaluation for conditional within-stratum and cross-stratum selection:** Freeze fresh, previously uninspected target and retention evaluation sets (`500` questions each, plus `490` remaining official `sciq:test` questions) with full exact and 13-word content n-gram decontamination against all training, development, and calibration records.

## 2. Frozen data partitions (`prepare_exp07.py`)

Reuse parent manifest `0d6dec93e7d7ccffd87f95e7ebd19e78c84ec72a546007b122888789aba5cb44` (`05_validation_guided_sft/expt_v2`) and `242d063db24573adc525b7fb523cc72bc0b6a01cb7b4f36248b823f93bccc1a8` (`06_contrastive_validation_zipmix/expt_v1`):
- **Candidate training pool (`train.npz`, $N = 9,797$):** Identical 9,797 candidate questions, 351 equal-width 4,096-byte packs, 5 compression-ratio quantile bins, and pack alignment scores against the 8 true SciQ validation packs and 8 wrong CommonsenseQA packs.
- **Stage 1 Calibration Validation Partition (used ONLY in Stage 1):**
  - `cal_target.npz` ($N = 500$): First 500 deterministic hash-sorted eligible records from `sciq:train` that were not used in `expt_v1` or `expt_v2` and share zero question IDs or 13-word content n-grams with any `expt_v1`/`expt_v2` record or any Stage 2 evaluation record.
  - `cal_retention.npz` ($N = 500$): First 500 deterministic hash-sorted eligible records from `commonsense_qa:train` that were not used in `expt_v1` or `expt_v2` and share zero question IDs or 13-word content n-grams with any `expt_v1`/`expt_v2` record, `cal_target`, or any Stage 2 evaluation record.
- **Stage 2 Fresh Held-Out Evaluation Partition (never loaded during Stage 1):**
  - `target.npz` ($N = 500$, primary fresh science holdout): First 500 deterministic hash-sorted eligible records from official `sciq:validation` outside the 242 `true_development_members` and 64 `expt_v1` alignment records, verified to have zero exact or 13-gram overlap with `train.npz` or any other split.
  - `retention.npz` ($N = 500$, primary fresh retention holdout): First 500 deterministic hash-sorted eligible records from official `commonsense_qa:validation` outside the 500 `expt_v1/v2` retention records, verified to have zero exact or 13-gram overlap with `train.npz` or any other split.
  - `test_holdout2.npz` ($N = 490$, secondary fresh official-test holdout): All 490 remaining eligible records from official `sciq:test` outside the 500 `expt_v1/v2` target records, verified to have zero exact or 13-gram overlap with `train.npz`, development views, or `target.npz`.
  - `reused_target.npz` ($N = 500$) and `reused_retention.npz` ($N = 500$): Byte-identical copies of `expt_v2`'s held-out arrays, evaluated in Stage 2 only as secondary continuity diagnostics.

## 3. Stage 1 — Independent Learning-Rate Calibration (`stage1_calibration`)

- **Matrix:** 5 learning rates (`lr_1e6` $= 10^{-6}$, `lr_2e6` $= 2\times 10^{-6}$, `lr_5e6` $= 5\times 10^{-6}$, `lr_1e5` $= 10^{-5}$, `lr_2e5` $= 2\times 10^{-5}$) × 3 seeds (`0, 1, 2`) = 15 training cells + 1 base evaluation on `(cal_target, cal_retention)`.
- **Sampling distribution:** Fixed to `stratum_uniform` (`sample_proportional`, uniform `1/9797` over the candidate pool). No selector scores are used in Stage 1.
- **Locked selection rule (`select_calibrated_lr`):**
  Among learning rates whose 3-seed mean `cal_retention` accuracy is at least $\text{base}_{\text{cal\_retention}} - 0.010$ (at most 1.0 percentage point retention drop), select the learning rate `lr*` with the highest 3-seed mean `cal_target` accuracy (ties broken by lower `cal_target` label negative log-likelihood, then smaller learning rate). Record whether `sample_proportional` at `lr*` achieves mean `cal_target` accuracy $\ge \text{base}_{\text{cal\_target}}$.

## 4. Stage 2a — Within-Stratum Conditional Alignment (`stage2_conditional`)

- **Matrix:** 3 methods × 6 independent seeds (`0, 1, 2, 3, 4, 5`) = 18 training cells + 1 base evaluation at the frozen calibrated `lr*`.
- **Methods (exact [NEXT_STAGE_DESIGN.md](../../04_validation_guided_zipmix/NEXT_STAGE_DESIGN.md) recipe):**
  1. `conditional_aligned`: For each pack $p$, signed contrast $r_p = s_p^{\text{true,max}} - s_p^{\text{wrong,max}}$. Within each `(source, compression_bin)` stratum of $m$ packs, average rank $R$, $z = (R - 0.5)/m - 0.5$, $w = \exp(\log(4) z)$, inherited by member questions and normalized inside each `(source, bin)` stratum to preserve population joint mass $M_{s,b} = n_{s,b}/N$ (`30.5298%` SciQ mass).
  2. `stratum_uniform`: Uniform sampling within each `(source, bin)` stratum with the same $M_{s,b}$ (`1/N` across all 9,797 questions).
  3. `conditional_shuffled`: Pack-rank permutation within each `(source, bin)` stratum using fixed seed `1707` in sorted `(source, bin)` order, normalized with the same $M_{s,b}$.
- **Primary comparison family:** `conditional_aligned minus stratum_uniform` and `conditional_aligned minus conditional_shuffled` on fresh `target` ($N=500$) and `test_holdout2` ($N=490$). Report both the 6-seed matrix (`seeds 0..5`, minimum two-sided sign-test p-value $0.03125$) and the prespecified 3-seed subset (`seeds 0, 1, 2`, minimum sign-test p-value $0.25$), with raw and Holm-adjusted p-values across the two-contrast family.

## 5. Stage 2b — Calibrated Cross-Stratum Selector Comparison (`stage2_selectors`)

- **Matrix:** 6 methods (`zipmix_static`, `shuffled_zipmix`, `wrongtarget_zipmix`, `contrastive_zipmix`, `source_matched`, `direct_zipfit`) × 3 seeds (`0, 1, 2`) = 18 training cells + 1 base evaluation at the same frozen `lr*`.
- **Comparisons:** Paired against `stratum_uniform` (`seeds 0, 1, 2` from Stage 2a) and against each method's matched control (`zipmix_static minus shuffled_zipmix`, `zipmix_static minus wrongtarget_zipmix`, `contrastive_zipmix minus source_matched`, `direct_zipfit minus stratum_uniform`) on fresh `target` ($N=500$), fresh `test_holdout2` ($N=490$), and fresh `retention` ($N=500$).
