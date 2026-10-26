# Zip-Mix experiment index

**Doc link:** <https://github.com/brando90/zip-mix/blob/main/experiments/README.md>

**TLDR:** All 57 training cells across Experiments 04–06 completed cleanly. Fine-tuning shows a limited development-targeting signal, but the full comparisons do not establish competitive superiority. Earlier proposals and the compression pilot remain preserved.

| Home | Setup and goal | Status |
|---|---|---|
| [00_related_work](00_related_work/README.md) | Historical literature notes | Retained; current source-verified design is in Experiment 04 |
| [01_compression_threshold_buckets](01_compression_threshold_buckets/README.md) | Compression-bucket proposal | Proposal |
| [02_alignment_prior_analysis](02_alignment_prior_analysis/README.md) | Public corpus compression distributions and priors | 20,000-document pilots committed (`results/cr_pilot_*.csv`); Phase 2 CR-distribution analysis done on CPU, [report](02_alignment_prior_analysis/results/phase2_report.md); ZIP-FIT scoring (Phase 3) not started |
| [03_zipmix_fixes_doremi_proxy_mixtures](03_zipmix_fixes_doremi_proxy_mixtures/README.md) | Earlier robust-optimization proxy-mixture proposal (grounded redesign 10-05-2026) | Designed, not run; blocked on the Experiment 02 gate |
| [04_validation_guided_zipmix](04_validation_guided_zipmix/README.md) | Nine methods × three seeds; from-scratch target/broad held-out language-model loss | 27/27 complete; prespecified gain criterion not met; [results](04_validation_guided_zipmix/results.md) |
| [05_validation_guided_sft](05_validation_guided_sft/README.md) | Seven methods × three seeds; Qwen2.5-0.5B supervised multiple-choice fine-tuning | 21/21 complete; positive shuffled/wrong-target contrasts, below unchanged base; [results](05_validation_guided_sft/results.md) |
| [06_contrastive_validation_zipmix](06_contrastive_validation_zipmix/README.md) | Three selectors × three seeds; contrastive alignment and matched-source control | 9/9 complete; planned target comparisons do not support an alignment benefit; [results](06_contrastive_validation_zipmix/results.md) |
| [07_training_calibration](07_training_calibration/README.md) | Fine-tuning recipe calibration on fresh questions (4 learning rates × 5 seeds), then the nine-cell within-stratum alignment design at 5 seeds on fresh held-out questions | 35/35 complete (FarmShare, 10-21-2026); Stage A gate passed at lr 2e-6; Stage B advance=false, +0.8 pp vs both controls under the 2 pp bar; exploratory Exp07b finds little headroom; [protocol](07_training_calibration/expt_v1/PROTOCOL.md), [stage B analysis](07_training_calibration/expt_v1/results/stage_b/analysis.json), [Exp07b](07_training_calibration/expt_v1/results/results_exp07b.md) |
| [07_calibrated_conditional_sft_zipmix](07_calibrated_conditional_sft_zipmix/README.md) | Second, never-run implementation of Experiment 07 (same pool, selectors and recipe; different evaluation sets and code) | Not run as a study; three trainer bugs fixed and 13 CPU tests added on 10-21-2026; one pre-registered diagnostic at lr 5e-6 (15 cells + base) gave +1.0 / +1.3 pp vs controls, under the 2 pp bar; [debug report](07_calibrated_conditional_sft_zipmix/debug_10-21-2026/DEBUG_REPORT.md) |

Every active experiment keeps its code, versioned protocol, data manifest, resumable checkpoint and results in its canonical folder. Raw data/model files are ignored. Updates describe actual completion separately from a successful launch. Numerical evidence and complete denominators determine claims.
