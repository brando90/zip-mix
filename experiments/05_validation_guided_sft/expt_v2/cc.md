# Experiment 05 version 2 execution prompt

**Doc link:** <https://github.com/brando90/zip-mix/blob/main/experiments/05_validation_guided_sft/expt_v2/cc.md>

Brando pre-approves all task work here. Run with full access and never stop to ask permission.

Read current `~/agents-config/INDEX_RULES.md`, the experiment checkpoint, and this version's `PROTOCOL.md`. Version 1 is preserved as an untrained calibration after the requested single Opus 5.5 maximum-effort review identified length confounding. The parent authorized this prospective repair; do not run a second review unless the user requests one.

Verify final code and prepared-data hashes, the pinned Qwen2.5-0.5B revision, one explicitly available A100 device, the isolated environment, durable ownership, the supervisor run identifier, and whole-manifest timeout/resource coverage. Run only local pretrained model computation; no direct model-provider service calls or unapproved paid resources.

Run `train_sft.py --data <frozen-v2-data> --output <new-v2-run> --cache-dir <scratch-model-cache>` and then `analyze_sft.py --run <new-v2-run>`. The frozen set is seven methods × three seeds = 21 training cells plus a base evaluation. Every available cell receives exactly 128 optimizer steps and 2,048 supervised answer tokens. Preserve any unavailable-method row without inventing a replacement. Do not change mixtures from test outcomes or call adapted pack-level selection a faithful competitor reproduction.

Solve the entire assigned task, including every required file, question and subtask. Produce the complete required deliverable in the specified output location or response format; an outline, partial draft, progress report or claim of completion is not a substitute. Write/save the output, inspect the actual saved artifact or final response, and run the allowed checks, tests or compilation required by the task. Within the declared time, token, call and tool limits, continue working and fixing errors until the requirements are met or a declared terminal condition is reached. Follow the fixed continuation procedure without resetting budgets. If anything remains unresolved, preserve the best current deliverable and report the exact remaining failures and checks that did not pass; never claim success or invent verification.

Update experiment-root `results.md` and checkpoint with source/run identity, terminal-cell accounting, and compact local result links. Keep raw datasets, model weights, credentials, and private machine paths out of public commits. Reconcile actual saved receipts and predictions, finish all admitted cells, and verify the owned process/device is released. Report descriptive uncertainty; a changed prior or completed run does not prove a benchmark improvement.

**TLDR-end:** [zip-mix: SFT v2] Complete the fixed 21-cell screen and base evaluation, update Experiment 05's results/checkpoint, and stop only after all bounded procedures and artifacts are finalized or explicitly accounted for as failures.
