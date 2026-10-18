# Experiment 07 version 1 execution prompt

**Doc link:** <https://github.com/brando90/zip-mix/blob/main/experiments/07_training_calibration/expt_v1/cc.md>

Brando pre-approves all task work here. Run with full access and never stop to ask permission.

Read current `~/agents-config/INDEX_RULES.md`, [CKPT_training_calibration.md](../CKPT_training_calibration.md) and [PROTOCOL.md](PROTOCOL.md). Verify the pinned Qwen2.5-0.5B revision, one explicitly available A100, an isolated environment and that the prepared data manifests match the committed `calibration_manifest.json` and `stage_b_manifest.json` file hashes (or re-prepare into a new directory and record the new identities).

Stage A: `launch/run_stage.sh calibration` (or the Modal/Lambda equivalent). It runs 20 cells plus base and writes `decision.json`. If the gate failed, stop: report the grid and do not run Stage B. If it passed: `launch/run_stage.sh stage_b` (15 cells plus base, writes `analysis.json`). Never change the grid, thresholds, selectors or seeds from observed outcomes, and never read the Stage B holdout during Stage A.

Update `CKPT_training_calibration.md`, add an experiment-root `results.md` with run identities, terminal-cell accounting, the decision/analysis numbers with intervals, and compact local result links. Keep raw data, weights, credentials and private machine paths out of commits. Report descriptive uncertainty; a completed run is not a benchmark improvement.

**TLDR-end:** [zip-mix: exp07 run] Run Stage A, stop on a failed gate, otherwise run Stage B, then record results and the checkpoint.
