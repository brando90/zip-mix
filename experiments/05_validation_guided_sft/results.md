# Experiment 05: validation targeting shows a limited fine-tuning signal

**Doc link:** <https://github.com/brando90/zip-mix/blob/main/experiments/05_validation_guided_sft/results.md>

**TLDR:** All 21 models and the unchanged-base evaluation completed cleanly. Zip-Mix gained 1.87 [0.43, 3.30] percentage points over shuffled alignment and 3.13 [1.70, 4.57] over the wrong target (three paired seeds; exact sign-test p-val=0.25 each), but trailed direct compression selection on average and the unchanged base on every seed. This is an exploratory targeting signal, not competitive superiority.

Completed 10-04-2026 at 13:56 PDT. The active condition is version 2: Qwen2.5-0.5B, seven methods × three training seeds, and 2,048 supervised answer-label tokens per model. All choices and source identities were frozen before benchmark outcomes were read. The full 21-cell manifest, rather than an interim subset, determined this report.

## Complete measured outcomes

Every model was evaluated on the same 500 SciQ science test questions and 500 CommonsenseQA validation questions. Intervals below are descriptive 95% Student-t intervals across three training seeds, conditional on the fixed questions. They do not cover uncertainty over new questions, tasks, corpora or architectures. Standard deviations describe variability across seeds. Absolute accuracy levels have p-val=n/a; no test against an arbitrary accuracy level was run.

| Method | Target accuracy % [95% interval] | Target standard deviation, percentage points | Retention accuracy % [95% interval] | Retention standard deviation, percentage points |
|---|---:|---:|---:|---:|
| Sample proportional | 75.000 [69.038, 80.962] | 2.400 | 54.333 [47.479, 61.188] | 2.759 |
| Uniform domains | 74.800 [70.469, 79.131] | 1.744 | 56.733 [51.732, 61.735] | 2.013 |
| Compel, packed views | 76.533 [75.283, 77.784] | 0.503 | 53.600 [51.879, 55.321] | 0.693 |
| Zip-Mix | 77.200 [73.320, 81.080] | 1.562 | 54.933 [52.925, 56.941] | 0.808 |
| Direct ZIP-FIT adaptation | 77.933 [77.646, 78.220] | 0.115 | 46.000 [41.261, 50.739] | 1.908 |
| Shuffled alignment | 75.333 [70.638, 80.029] | 1.890 | 53.667 [50.756, 56.578] | 1.172 |
| Wrong development target | 74.067 [70.759, 77.375] | 1.332 | 56.267 [51.001, 61.532] | 2.120 |

The unchanged base checkpoint scored **404/500 = 80.800%** on science and **285/500 = 57.000%** on retention. These are one deterministic evaluation each; training-seed intervals and p-values do not apply. Every one of the 21 trained models scored below the base on science accuracy. All seven retention means were also lower, although individual uniform-domain and wrong-target seeds equaled or exceeded the base. These are properties of this observed run set.

| Zip-Mix minus comparator | Target difference, percentage points [95% interval] | Target p-value | Retention difference, percentage points [95% interval] | Retention p-value |
|---|---:|---:|---:|---:|
| Sample proportional | 2.200 [-4.702, 9.102] | 0.25 | 0.600 [-8.232, 9.432] | 1 |
| Uniform domains | 2.400 [-4.321, 9.121] | 1 | -1.800 [-6.216, 2.616] | 0.25 |
| Compel, packed views | 0.667 [-3.617, 4.950] | 1 | 1.333 [0.574, 2.092] | 0.25 |
| Direct ZIP-FIT adaptation | -0.733 [-4.749, 3.282] | 1 | 8.933 [4.159, 13.707] | 0.25 |
| Shuffled alignment | 1.867 [0.432, 3.301] | 0.25 | 1.267 [-3.241, 5.775] | 1 |
| Wrong development target | 3.133 [1.699, 4.568] | 0.25 | -1.333 [-4.715, 2.049] | 1 |

P-values are unadjusted, exploratory, exact two-sided sign tests of the null that either method has equal probability of higher accuracy among non-tied training seeds. With three non-tied pairs the smallest attainable p-value is 0.25. The Student-t intervals and sign tests rely on different assumptions and are not interchangeable significance decisions; intervals excluding zero do not turn this screen into a confirmed superiority result. No statistical threshold was used to select or stop training.

![All 21 models, individual seeds and unchanged-base comparison](expt_v2/results/measured/accuracy_results.png)

**The correct development target improves observed science accuracy relative to shuffled and wrong-target controls, while every trained model remains below the unchanged base.** Points show individual training seeds; diamonds and bars show means and 95% training-seed intervals. Dashed lines mark the base checkpoint. The [unaltered original analysis](expt_v2/results/measured/results_summary.md) contains all 21 correct-count pairs; [machine-readable results](expt_v2/results/measured/summary.json) and per-item predictions preserve the full denominator.

## What this says about the hypothesis

Zip-Mix outscored sample-proportional, shuffled and wrong-target training on science in all three paired seeds. The strongest mechanism evidence is the consistent difference when the development target is replaced or its alignment scores are permuted. However, these interventions also change source proportions and compression-bin exposure. They show that the chosen target can affect the resulting model in this setup; they do not isolate a benefit from bins themselves or prove semantic alignment independently of source mixture.

Zip-Mix's target difference against direct selection was −0.733 [−4.749, 3.282] percentage points, p-val=1.0. Its retention difference was +8.933 [4.159, 13.707], p-val=0.25. Direct selection puts all candidate probability on SciQ, so its higher target mean and lower retention are consistent with a specialization tradeoff. This comparison also changes both maximum-versus-mean alignment aggregation and bin sampling-versus-top-quarter selection; it cannot attribute the tradeoff to binning alone. No combined utility was declared, so neither method is an established overall winner.

The base comparison reveals a more immediate limitation: this particular full-model, answer-label fine-tuning recipe reduces observed science performance. These outcomes do not identify the cause. The next stage should first calibrate training on a separate training/validation partition with an unchanged-base control, and then use fresh uninspected evaluation questions. Learning rate, supervision format and stopping choices must not be optimized against the test values reported here.

The prospectively designed [contrastive follow-up](../06_contrastive_validation_zipmix/results.md) did not improve on matching source proportions. The completed [cross-study analysis](../06_contrastive_validation_zipmix/cross_experiment/comparison.md) audits all 30 fine-tuned models and both base evaluations, reports every comparison, and labels them exploratory. The studies share held-out questions and the same three nominal seeds; they are not independent benchmark replications. A proposed [conditional-selection study](../04_validation_guided_zipmix/NEXT_STAGE_DESIGN.md) fixes joint source-by-compression exposure to test alignment within groups, but it has not been launched.

## Frozen design and scope

Version 1 remains an [untrained calibration](expt_v1/STATUS.md). The requested single Claude Opus 5.5 maximum-effort review identified short-string length confounding; the parent made the prospective version 2 repair before training, without a second model-review round.

| Item | Verified version 2 choice |
|---|---|
| Model | Qwen/Qwen2.5-0.5B, revision `060db6499f32faf8b98477b0a26969ef7d8b9987` |
| Candidate pool | 9,797 questions in 351 disjoint source-specific packs |
| Development views | Eight true and eight wrong-target views; exactly 4,096 real bytes each |
| Exclusions | 256 wrong-target source records reserved outside training; 63 underfilled-tail examples excluded from every arm |
| Training | 128 updates × 16 questions; float32 master parameters and optimizer, bfloat16 forward computation |
| Input identity | `0d6dec93e7d7ccffd87f95e7ebd19e78c84ec72a546007b122888789aba5cb44` |
| Split audit | Zero exact or declared 13-word overlap hits against protected official splits |
| Selection support | All seven methods available; Compel retains 9,722/9,797 questions |

The scientific procedure is [expt_v2/PROTOCOL.md](expt_v2/PROTOCOL.md). Selection reads development inputs, never evaluation labels. Each of the 367 candidate/development views has exactly 4,096 real bytes without artificial padding or repetition. The true-target mixture assigns 35.8677% probability to SciQ, versus 26.9331% for the wrong target and 30.5298% for sample-proportional sampling. These are exact finite-pool probabilities, p-val=n/a; source, repetition and pack-density effects remain.

CommonsenseQA occurs in training, so its held-out evaluation measures untargeted retention within a seen family, not a new task family. Public-benchmark exposure during the backbone's pretraining is unknown. Each question supervises one answer label; the screen does not establish free-form reasoning or reinforcement-learning gains. Compel and direct ZIP-FIT are declared adaptations. This fine-tuning study does not compare against Domain Reweighting with Generalization Estimation (DoGE), Domain Reweighting with Minimax Optimization (DoReMi), low-rank gradient similarity search (LESS), or newer mixing methods; the separate pretraining study's compact learned-mixture comparisons do not establish superiority in post-training.

## Procedure, costs and reproducibility

- **21/21 training cells and 1/1 base evaluation completed**, zero failed, missing or unavailable cells, zero infrastructure resumes. Training and analysis exited successfully; all prediction audits passed.
- Frozen training source: [`611423be`](https://github.com/brando90/zip-mix/commit/611423be2e62531453a0ad4e3c496a5cd07c711c). Run identity: `199d165af58bd69d9c48349f28f96e7d83a73ba0aa429949bd12756470c850e9`.
- Prelaunch validation: **38/38 tests**, all 15 input hashes and both preparation-source hashes verified. All 351 alignment scores and 367 view hashes were independently reproduced.
- Actual supervised wall time: **6,463.014 seconds (1 hour, 47 minutes, 43 seconds)** on one A100 graphics processing unit, against a six-hour ceiling. Training plus analysis spanned 6,456.469 seconds. Exact owned process identities exited, and resource-release checks passed.
- Supervision totals: **43,008 answer-label tokens** and **2,083,587 non-padding prompt tokens**. Summed per-cell training time was **6,041.679 seconds**, including checkpoint writes. The recorded evaluation/final-save field sums to 294.127 seconds; base loading plus evaluation took 4.256 seconds. These are measured elapsed times, not pure accelerator kernel time or monetary spend. Dollar cost is unavailable.
- All **69 original exported proof files** remain byte-identical. Independent local reanalysis reproduces all means, counts, p-values and other non-interval scientific fields exactly. Numerical-library versions produce 76 confidence-interval endpoint differences, at most **1.461 × 10⁻¹²**, below the declared 1e-10 tolerance; the original remote outputs remain canonical. The [export receipt](expt_v2/results/measured/EXPORT.md) records source hashes, file hashes and comparison policy.

**TLDR-end:** [zip-mix: complete fine-tuning results] The correct development target beats shuffled and wrong-target selection in all three observed seeds, but the screen does not establish broad or competitive superiority. The unchanged base remains stronger on science, so training calibration precedes a larger selection study.

**Snapshot:**
```text
complete_training_cells=21/21; base_evaluation=complete
failed_missing_unavailable_or_resumed_cells=0
zipmix_science_accuracy_percent=77.200 [73.320, 81.080]; p-val=n/a
zipmix_minus_shuffled_pp=1.867 [0.432, 3.301]; p-val=0.25
zipmix_minus_wrong_target_pp=3.133 [1.699, 4.568]; p-val=0.25
base_science_correct=404/500
owned_training_processes_after_completion=0
```
