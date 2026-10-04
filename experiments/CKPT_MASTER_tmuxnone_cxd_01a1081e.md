# Zip-Mix coordinator: full validation-guided pretraining and fine-tuning comparisons

**Doc link:** <https://github.com/brando90/zip-mix/blob/main/experiments/CKPT_MASTER_tmuxnone_cxd_01a1081e.md>

**TLDR:** All 27 pretraining models completed cleanly; the original Zip-Mix gain threshold was not met. The revised 21-cell fine-tuning run is active on a second A100, with fixed-byte scoring and a wrong-target control. Fine-tuning results remain pending; a separate nine-cell contrastive selector test is being frozen before inspecting those outcomes.

Updated: 10-04-2026 12:44 PDT

## Identity and recovery

Local Codex coordinator for Zip-Mix. Exact host, process, terminal-session and resume identities live in the private `zipmix-validation-guided/launch_context.json` receipt under the coordinator's Codex private store. Global local instruction files are nonempty and resolve to the current shared rules; remote defaults were corrected for future launches without restarting existing agents. The requested single quality assurance (QA) review actually used `claude-opus-5-5` and the explicit maximum-effort flag through existing subscription authentication.

## Ownership

Preserve pre-existing staged deletions, Experiment 03 rename/edits, paper edits and Experiment 02 materials. Only explicit task-owned paths are committed. Initialization commit `14c4d4f`, prospective repair commit `611423b`, live records `d96c223` and complete pretraining evidence `6212fdc` are pushed. Raw data, model weights, checkpoints and host receipts remain ignored/private.

## Scientific state

- [Experiment 04](04_validation_guided_zipmix/results.md): frozen nine-method × three-seed from-scratch screen, 27 cells. All 27 cells, six proxy stages and three reference stages completed cleanly, with no failures or recovery. The prespecified improvement criterion was not met. Compact DoGE had lower target loss than Zip-Mix. Complete small numerical evidence and the structural bin-capacity diagnosis are published; original analysis was independently reproduced from saved loss arrays.
- [Experiment 05](05_validation_guided_sft/results.md): retain short-string v1 as **untrained calibration**. Prospective v2 uses 351 real 4,096-byte training packs, 8 matched true-target and 8 wrong-target development views, seven methods × three seeds (21 cells), and one unchanged-base evaluation. Started at 12:08 PDT on a separately verified idle A100 after 38 remote tests and all 15 data hashes passed. The unchanged-base evaluation completed. Deadline 6 hours. No settings selected from test outcomes.
- Requested review: six implementation/documentation fixes, no critical issue, one major scientific concern. The concern motivated v2; deterministic checks verify the correction. No second review is authorized or needed under the requested one-round procedure.

## Completion contract

Ordinary Python supervision survives coordinator disconnection, owns only its process group and emits timestamped, run-bound progress. A coordinator turn ending never stops healthy admitted work. No automatic reboot recovery is claimed. Analyze all declared cells, preserve missing/failing/recovered rows, verify the unchanged-base evaluation and resource release, and update numerical reports before calling either matrix complete.

Experiment 06 is not yet admitted: three selector methods × three seeds plus base; bounded at two device-hours after resource preflight. It includes a source-mass-matched control and shuffled contrast scores. Its design uses training/development diagnostics and the completed pretraining outcome, with no Experiment 05 benchmark inspection.

Three seeds provide a descriptive mechanism screen: the smallest two-sided exact sign-test p-value is 0.25. Report means, paired intervals, retained denominators and total costs. This does not establish optimal training or state-of-the-art superiority. Larger faithful reproductions and reinforcement learning remain prospective work.

## Next actions

Experiment 04 is complete and its point estimates, uncertainty and structural bin-capacity diagnosis are published. Experiment 05 remains healthy and its benchmark outcomes are uninspected. Freeze and verify the separately authorized nine-cell Experiment 06 contrastive selector study, using the same pool/model/updates with checkpoint interval 64 as a documented operational difference. Keep admitted work healthy. Analyze complete artifacts, publish compact numerical receipts/figures and update the root index. No human decision is waiting.
