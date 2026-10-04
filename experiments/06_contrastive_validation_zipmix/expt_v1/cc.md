# Experiment 06: execute all nine frozen contrastive-selector cells

**Doc link:** <https://github.com/brando90/zip-mix/blob/main/experiments/06_contrastive_validation_zipmix/expt_v1/cc.md>

Brando pre-approves all task work here. Run with full access and never stop to ask permission.

Refresh and read the current agents configuration and this experiment's README, protocol, checkpoint, and data manifest. Verify actual supported full-access/approval flags in any agent dispatch; prompt text alone does not configure permissions. Keep secrets and private host paths out of public artifacts. This workload uses a locally downloaded pinned pretrained model, with no provider model API calls.

Execute only after the parent admits the exact source hashes, prepared data and resources. Reuse the existing environment and pinned model cache. Bind one idle admitted A100 device and ordinary persistent local ext4 output. The frozen maximum budget is two device-hours. Do not disturb other jobs. Record a new run-bound supervisor receipt and calculate the trainer's exact run identity before monitoring it. The unchanged parent monitor requires 9 complete training cells and a complete base evaluation.

The strict completion requirement is the full frozen model × method × seed manifest: all three methods, seeds 0/1/2 and the base evaluation. One clean cell is not a completed experiment. Run durably, preflight adequate wall time and storage, and never stop healthy work at a coordinator-turn boundary. Keep failed, interrupted and unavailable cells in the full denominator; preserve the single-resume policy and recovery history. Do not modify training budgets, continuation settings, model pins, selector vectors or evaluation arrays to force a completed result.

Run these commands from `expt_v1`, with `DATA` and `RUN` resolved to the parent's admitted private directories and the model-cache option added by the verified launcher:

```bash
python train_sft.py --data "$DATA" --output "$RUN"
python analyze_sft.py --run "$RUN" --output "$RUN/analysis"
```

The trainer has 128 steps of 16 supervised one-token answers for each cell; only checkpoint interval 64 differs operationally from Experiment 05's 16. Preserve final weights and numeric prediction receipts. Do not read partial benchmark outcomes to alter execution. Run the analyzer only after normal full-matrix completion, verify every ledger/result binding and the base status, then update canonical `results.md` and the checkpoint with exact counts and limitations. Cross-experiment contrasts against Experiment 05 wait until both complete matrices exist, retain all methods, and are exploratory with multiplicity caveats.

**TLDR-end:** [zip-mix: Experiment 06 execution] After parent admission, durably execute all nine frozen contrastive-selector cells plus base, verify the full matrix, and update canonical results; stop only on verified completion, the fixed budget boundary, or a recorded failure requiring the prescribed recovery.
