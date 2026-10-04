# Completed fine-tuning studies: exploratory cross-experiment comparison

**Doc link:** <https://github.com/brando90/zip-mix/blob/main/experiments/06_contrastive_validation_zipmix/cross_experiment/comparison.md>

**TLDR:** All 30 training cells and two base evaluations are verified. These are three paired seeds per method on shared held-out questions, not six independent seeds or a new benchmark replication.

95% Student-t across the same three paired training seeds, conditional on fixed items. Exact two-sided sign test: P(contrastive accuracy > comparator accuracy)=0.5 among non-tied seeds. Seven exploratory target comparisons against every Experiment 05 method; raw and Holm-adjusted p-values. Retention is an unadjusted diagnostic.

Accuracy is percent; differences and standard deviations are percentage points. Positive differences favor contrastive ZipMix. Means have p-val=n/a; no test against an arbitrary accuracy level was run.

| Study / method | Target accuracy [95% interval] | Target standard deviation | Retention accuracy [95% interval] | Retention standard deviation |
|---|---:|---:|---:|---:|
| 05/sample_proportional | 75.000 [69.038, 80.962] | 2.400 | 54.333 [47.479, 61.188] | 2.759 |
| 05/uniform_domain | 74.800 [70.469, 79.131] | 1.744 | 56.733 [51.732, 61.735] | 2.013 |
| 05/compel_filter | 76.533 [75.283, 77.784] | 0.503 | 53.600 [51.879, 55.321] | 0.693 |
| 05/zipmix_static | 77.200 [73.320, 81.080] | 1.562 | 54.933 [52.925, 56.941] | 0.808 |
| 05/direct_zipfit | 77.933 [77.646, 78.220] | 0.115 | 46.000 [41.261, 50.739] | 1.908 |
| 05/shuffled_zipmix | 75.333 [70.638, 80.029] | 1.890 | 53.667 [50.756, 56.578] | 1.172 |
| 05/wrongtarget_zipmix | 74.067 [70.759, 77.375] | 1.332 | 56.267 [51.001, 61.532] | 2.120 |
| 06/contrastive_zipmix | 75.333 [71.348, 79.318] | 1.604 | 49.400 [40.457, 58.343] | 3.600 |
| 06/source_matched | 77.067 [72.090, 82.043] | 2.003 | 53.133 [45.086, 61.180] | 3.239 |
| 06/contrastive_shuffled | 75.267 [74.980, 75.554] | 0.115 | 51.000 [47.025, 54.975] | 1.600 |

| Contrastive ZipMix minus Experiment 05 comparator | Target difference [95% interval] | Raw p-value | Holm p-value | Retention difference [95% interval] | Raw p-value |
|---|---:|---:|---:|---:|---:|
| sample_proportional | 0.333 [-5.207, 5.873] | 1 | 1 | -4.933 [-8.315, -1.551] | 0.25 |
| uniform_domain | 0.533 [-2.378, 3.444] | 1 | 1 | -7.333 [-19.915, 5.249] | 0.25 |
| compel_filter | -1.200 [-5.753, 3.353] | 1 | 1 | -4.200 [-14.669, 6.269] | 0.25 |
| zipmix_static | -1.867 [-9.472, 5.739] | 1 | 1 | -5.533 [-16.467, 5.401] | 0.25 |
| direct_zipfit | -2.600 [-6.351, 1.151] | 0.25 | 1 | 3.400 [-6.938, 13.738] | 1 |
| shuffled_zipmix | 0.000 [-8.620, 8.620] | 1 | 1 | -4.267 [-11.228, 2.695] | 0.25 |
| wrongtarget_zipmix | 1.267 [-6.007, 8.540] | 1 | 1 | -6.867 [-21.044, 7.311] | 0.25 |

The source-matched and shuffled controls within Experiment 06 remain its two separately declared primary/secondary target contrasts; see that study's report. These cross-experiment comparisons were planned as exploratory before either complete fine-tuning matrix was inspected. Three seeds give a minimum raw sign-test p-value of 0.25; intervals and sign tests address different summaries and should not be read as interchangeable significance decisions.

The model revision, candidate order, tokenizer, evaluation arrays, initialization seeds and learning settings match. Checkpoint cadence differs (16 versus 64 updates), so elapsed-time differences cannot identify selector efficiency. The same nominal seed pairs initial conditions; different mixtures intentionally lead to different sampled questions. Fine-tuning supervises one answer label per question; these results do not establish free-form reasoning or reinforcement-learning improvement.

Every individual seed, source hash and comparison is in `comparison.json`. No model fitting, additional evaluation, checkpoint selection or training rerun is performed by this analysis.
