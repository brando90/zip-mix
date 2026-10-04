# Zip-Mix initialization checkpoint

**Doc link:** <https://github.com/brando90/zip-mix/blob/main/experiments/04_validation_guided_zipmix/CKPT_zipmix.md>

**TLDR:** All 27 pretraining cells completed without failures or recoveries. Zip-Mix has small gains over population/shuffled sampling, misses the prespecified gain threshold, and trails the compact DoGE comparison.

Updated 10-04-2026 12:35 PDT.

## Ownership and evidence

Base commit c672ee7. Existing project changes at task entry include staged removals/renames, paper-introduction edits and Experiment02/03 material; these are not this task's changes. Owned paths: AGENTS.md,CLAUDE.md,README.md,experiments/README.md and new Experiments04/05. No existing worker was interrupted.

Local and remote shared rules were refreshed. Local globals: /Users/brandomiranda/.codex/AGENTS.md and /Users/brandomiranda/CLAUDE.md resolve to nonempty agents-config files. The current Codex profile reports gpt-6-astra/ultra with danger-full-access/never. Existing agents do not reload automatically. Remote details remain in the private preflight receipt.

The source pilot hash and prepared artifact hashes are frozen in data_manifest.json. Preparation took 54.98 seconds;14,341 documents remain. Target and broad training outcomes are unmeasured. Training config proposes 8,388,608 final predicted tokens per cell, 27 cells; reference/proxy costs are additional and explicitly recorded. Engineering timing will verify the allocation bound, not choose settings by quality.

## Next actions

1. Preserve the completed source/data/run identity and all 27 outcomes. No more training is needed to complete this frozen matrix.
2. Use the published results and numerical proof bundle; do not lower the screening threshold after seeing results.
3. Continue the independently frozen Experiment 05 and prepare the separate prospective Experiment 06 contrastive-selector test. No Experiment 05 benchmark scores informed its design.

## Engineering verification

Twenty deterministic training tests pass, including all nine methods and bit-identical optimizer/random-state resume. The 20-step A100 engineering run completed 1/1 cell in 4.99 seconds (3.29 seconds in training); 81,920 predicted tokens, 5,289,472 parameters. Its held-out numbers are excluded from scientific comparisons. The device returned to 0 MiB used. A conservative linear extrapolation puts the simple final-training matrix near 2.5 device-hours, before explicitly additional proxy/reference overhead; the 12-hour ceiling remains ample.

Review update 10-04-2026: the requested review applied fixes to `train.py` (numerical failures are never retried; data must match the frozen preparation manifest) and Experiment 05's trainer; 23 Experiment 04 tests pass. The `train.py` hash changed, so the engineering-timing fingerprint differs from any launch fingerprint; timing remains engineering-only. See [review report](qa/opus55_max_review.md). The pinned Qwen2.5-0.5B model was downloaded to the isolated node-local cache for Experiment 05.

## Actual launch

Reviewed source revision `14c4d4f`; five transferred data-file hashes matched the frozen preparation manifest. The remote deterministic suite passed 23 tests. A real persistent terminal session runs the task-owned Python supervisor, which checks the exact 27-cell denominator and records bound progress/process/device receipts. The first cell passed step 896/2,048 with zero failed cells at the initial observation. Exact process/host/session identities are private. The requested review finished as Claude Opus 5.5 with maximum effort; no further model review was started.

## Verified completion

All 27/27 cells completed with zero recoveries; all frozen final budgets match. Whitelisted metrics, mixtures and numeric loss arrays are in `expt_v1/results/measured/`, with export hashes and documented manifest-path sanitization. Unchanged local analysis passed; point estimates match exactly, with interval-only roundoff below 9e-13 across library versions. The scalar threshold rule is retained as not met. See `results.md` for the complete quantitative interpretation.

A separate training/development-only analysis in `expt_v1/results/bin_capacity.md` shows that the fixed compression bins can assign at most 23.5496% of mass to biomedical training blocks, below all three learned DoGE mixtures. This is a source-exposure constraint, not a proof of what causes the performance gap.
