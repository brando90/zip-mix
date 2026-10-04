# Experiment 06 results

**Doc link:** <https://github.com/brando90/zip-mix/blob/main/experiments/06_contrastive_validation_zipmix/results.md>

**TLDR:** The full nine-cell prospective contrastive-selector study launched on 10-04-2026 at 12:50:58 PDT; its separate base evaluation is complete. Final benchmark comparisons remain pending.

The verified source commit is `ab85f46074d0e479f39369cb0fe7fa0751876ef2`. All nine training cells remain in the denominator; launch verification found one running, eight pending and zero failed. Execution is bound to one admitted A100 with a durable two-hour supervisor and separate terminal artifact/release checks.

Exact development-only diagnostics are in [the frozen manifest](expt_v1/data_manifest.json). SciQ sampling mass is 58.3210% for both the primary and source-matched control, and 31.3883% for the shuffled contrast control. Total variation from the primary is 0.213562 for the source-matched control and 0.394043 for the shuffled control. These are deterministic finite-pool probabilities; confidence intervals and p-values are not applicable.

The hypothesis, primary comparison and multiple-comparison limits were specified before reading Experiment 05 outcomes. Results must distinguish procedure completion from evidence of improvement. Parent Experiment 05 outcomes and this condition's results will remain separately identified; cross-experiment comparisons are exploratory and wait for both full matrices.

Preparation verification: 35/35 deterministic checks passed in 5.46 seconds, including all three frozen selector vectors, source-mass matching, inherited checkpoint recovery and the two-comparison Holm adjustment. All 12 input-file hashes and the byte-identical parent trainer were independently verified. See [freeze_receipt.json](expt_v1/freeze_receipt.json).
