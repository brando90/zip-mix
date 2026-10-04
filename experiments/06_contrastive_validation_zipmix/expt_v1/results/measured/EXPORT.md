# Verified result export

**Doc link:** <https://github.com/brando90/zip-mix/blob/main/experiments/06_contrastive_validation_zipmix/expt_v1/results/measured/EXPORT.md>

**TLDR:** All 9 training cells and the base evaluation are verified; 33 original proof artifacts remain byte-identical after export and local reanalysis.

This directory contains the complete, frozen 9-cell 06_contrastive_validation_zipmix/expt_v1 training matrix and one base evaluation. All 9 training cells and the base passed the terminal supervisor and analysis gates before export. There are zero failed, missing or unavailable cells and 0 infrastructure resumes.

Exactly 33 source artifacts were copied: `ledger.json`, `summary.json`, `results_summary.md`, and each training cell's and the base evaluation's `result.json`, `target_predictions.npz` and `retention_predictions.npz`. Every source artifact remains byte-identical, including the original remote summary and Markdown. No model weights, checkpoints, raw questions, input-token arrays, command logs or private host packets were copied. Prediction arrays contain example identifiers, answer labels, predictions, correctness indicators and label losses.

The unchanged frozen analyzer was run locally against the export, writing its outputs privately rather than replacing the original evidence. All scientific values match exactly except 28 confidence-interval endpoints, whose maximum absolute rounding difference is 1.444372399461713e-12, below the declared 1e-10 tolerance. Creation timestamps are ignored. Remote NumPy/SciPy versions are 2.5.3/1.18.1; local versions are 2.2.6/1.16.3. The receipt records every source/export hash, frozen source identity, exact completion counts, rounding differences, recomputed hashes and measured execution totals.

The PNG and PDF figures were generated from the verified remote summary and visually inspected. Their separate provenance receipt binds the original summary and both generated images. Independent execution checks confirmed release of the owned processes. No dollar cost is inferred from compute time. These artifacts establish complete procedures and reproducible measurements; they do not establish generalization superiority.
