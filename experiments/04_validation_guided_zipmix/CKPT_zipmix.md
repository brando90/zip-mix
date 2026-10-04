# Zip-Mix initialization checkpoint

**Doc link:** <https://github.com/brando90/zip-mix/blob/main/experiments/04_validation_guided_zipmix/CKPT_zipmix.md>

**TLDR:** Data preparation and a single-device remote environment are ready; training and the requested Opus 5.5 maximum-effort review remain pending. Preserve existing unrelated changes and complete the frozen experiment manifest before drawing conclusions.

Created 10-04-2026 11:25 Pacific. Updated 10-04-2026 11:29 Pacific.

## Ownership and evidence

Base commit c672ee7. Existing project changes at task entry include staged removals/renames, paper-introduction edits and Experiment02/03 material; these are not this task's changes. Owned paths: AGENTS.md,CLAUDE.md,README.md,experiments/README.md and new Experiments04/05. No existing worker was interrupted.

Local and remote shared rules were refreshed. Local globals: /Users/brandomiranda/.codex/AGENTS.md and /Users/brandomiranda/CLAUDE.md resolve to nonempty agents-config files. The current Codex profile reports gpt-6-astra/ultra with danger-full-access/never. Existing agents do not reload automatically. Remote details remain in the private preflight receipt.

The source pilot hash and prepared artifact hashes are frozen in data_manifest.json. Preparation took 54.98 seconds;14,341 documents remain. Target and broad training outcomes are unmeasured. Training config proposes 8,388,608 final predicted tokens per cell, 27 cells; reference/proxy costs are additional and explicitly recorded. Engineering timing will verify the allocation bound, not choose settings by quality.

## Next actions

1. Finish training and supervised-fine-tuning implementations and deterministic checks.
2. Execute isolated single-device engineering timing, verify finite losses/checkpoint/evaluation and full-run estimate.
3. Run one requested Claude Code review with exact claude-opus-5-5 model and --effort max using existing subscription authentication; apply findings.
4. Synchronize verified source and inputs, launch the complete manifest with durable ownership and monitoring; update this checkpoint with real process identity.
5. Analyze and publish task-owned numerical evidence; never claim the still-unmeasured goal is achieved.

## Engineering verification

Twenty deterministic training tests pass, including all nine methods and bit-identical optimizer/random-state resume. The 20-step A100 engineering run completed 1/1 cell in 4.99 seconds (3.29 seconds in training); 81,920 predicted tokens, 5,289,472 parameters. Its held-out numbers are excluded from scientific comparisons. The device returned to 0 MiB used. A conservative linear extrapolation puts the simple final-training matrix near 2.5 device-hours, before explicitly additional proxy/reference overhead; the 12-hour ceiling remains ample.

Review update 10-04-2026: the requested review applied fixes to `train.py` (numerical failures are never retried; data must match the frozen preparation manifest) and Experiment 05's trainer; 23 Experiment 04 tests pass. The `train.py` hash changed, so the engineering-timing fingerprint differs from any launch fingerprint; timing remains engineering-only. See [review report](qa/opus55_max_review.md). The pinned Qwen2.5-0.5B model was downloaded to the isolated node-local cache for Experiment 05.
