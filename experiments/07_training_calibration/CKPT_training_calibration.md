# CKPT: Experiment 07 training calibration

**Doc link:** <https://github.com/brando90/zip-mix/blob/main/experiments/07_training_calibration/CKPT_training_calibration.md>

Created: 10-18-2026. Last updated: 10-18-2026.

## State
- **Goal:** fix the fine-tuning recipe so trained models at least match the unchanged base on fresh questions (Stage A), then test within-stratum alignment on fresh held-out questions (Stage B).
- **What ran:** CPU only. Data freeze (Experiment 05 v1/v2 identities and the design receipt reproduced exactly; all overlap audits 0), 21 Experiment 07 tests and the 127 existing repository tests (1 skipped) passed, CPU smoke of both stages with a tiny random model.
- **Results:** none; no GPU cell has run.
- **Stuck / blocked:** needs one A100. The assistant cannot reach SNAP (Kerberos + Duo); Modal or Lambda credits, or Brando launching on SNAP, unblock it.
- **Next step:** `experiments/07_training_calibration/expt_v1/launch/run_stage.sh calibration` on a SNAP node, or the Modal/Lambda command in [launch/README.md](expt_v1/launch/README.md).
- **Brando to-do:** merge the PR; pick where Stage A runs.

## Log
- 10-18-2026: protocol frozen, data prepared and verified on CPU, launchers written; nothing launched.
