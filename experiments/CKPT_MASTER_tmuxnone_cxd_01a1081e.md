# Zip-Mix coordinator: full validation-guided pretraining and fine-tuning comparisons

**Doc link:** <https://github.com/brando90/zip-mix/blob/main/experiments/CKPT_MASTER_tmuxnone_cxd_01a1081e.md>

**TLDR:** The reviewed 27-cell pretraining experiment is running. A prospective 21-cell fine-tuning version fixes the short-string scoring problem and adds a wrong-target control; its results remain pending.

Updated: 10-04-2026 12:00 PDT

## Identity and recovery

Local Codex coordinator for Zip-Mix. Exact host, process, terminal-session and resume identities live in the private `zipmix-validation-guided/launch_context.json` receipt under the coordinator's Codex private store. Global local instruction files are nonempty and resolve to the current shared rules; remote defaults were corrected for future launches without restarting existing agents. The requested single quality assurance (QA) review actually used `claude-opus-5-5` and the explicit maximum-effort flag through existing subscription authentication.

## Ownership

Preserve pre-existing staged deletions, Experiment 03 rename/edits, paper edits and Experiment 02 materials. Only explicit task-owned paths are committed. Initialization commit `14c4d4f` is pushed. Raw data, model weights, checkpoints and host receipts remain ignored/private.

## Scientific state

- [Experiment 04](04_validation_guided_zipmix/results.md): frozen nine-method × three-seed from-scratch screen, 27 cells. Started 10-04-2026 11:58 PDT on one A100. Bound live process and monitor verified; first cell passed 896/2,048 steps with no failure. Deadline 12 hours, ample relative to measured pace. Five transferred data hashes and 23 remote tests passed.
- [Experiment 05](05_validation_guided_sft/results.md): retain short-string v1 as **untrained calibration**. Prospective v2 uses 351 real 4,096-byte training packs, 8 matched true-target and 8 wrong-target development views, seven methods × three seeds (21 cells), and one unchanged-base evaluation. A separate idle A100 may run this concurrently after final source/data checks. Deadline 6 hours. No settings selected from test outcomes.
- Requested review: six implementation/documentation fixes, no critical issue, one major scientific concern. The concern motivated v2; deterministic checks verify the correction. No second review is authorized or needed under the requested one-round procedure.

## Completion contract

Ordinary Python supervision survives coordinator disconnection, owns only its process group and emits timestamped, run-bound progress. A coordinator turn ending never stops healthy admitted work. No automatic reboot recovery is claimed. Analyze all declared cells, preserve missing/failing/recovered rows, verify the unchanged-base evaluation and resource release, and update numerical reports before calling either matrix complete.

Three seeds provide a descriptive mechanism screen: the smallest two-sided exact sign-test p-value is 0.25. Report means, paired intervals, retained denominators and total costs. This does not establish optimal training or state-of-the-art superiority. Larger faithful reproductions and reinforcement learning remain prospective work.

## Next actions

Finish fine-tuning source checks, commit only owned changes, transfer verified v2 data and launch its full 21 cells. Keep pretraining healthy. Analyze complete artifacts, publish compact numerical receipts/figures and update the root index. No human decision is waiting.
