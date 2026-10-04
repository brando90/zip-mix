# Zip-Mix coordinator: initialized pretraining and supervised fine-tuning

**Doc link:** <https://github.com/brando90/zip-mix/blob/main/experiments/CKPT_MASTER_tmuxnone_cxd_01a1081e.md>

**TLDR:** Both executable experiments and their frozen data are prepared. The requested Claude Opus 5.5 maximum-effort review finished: 50 tests pass, no critical issue, one fine-tuning length confound. Pretraining is ready; fine-tuning is being redesigned prospectively before admission.

Created: 10-04-2026 11:44 PDT
Last updated: 10-04-2026 11:50 PDT

## Identity and recovery

Local Codex coordinator for Zip-Mix; full thread/profile/host identities and exact private resume/deployment commands are in the private `zipmix-validation-guided/launch_context.json` receipt under the coordinator's Codex private store. No private host packet or credential belongs in this public repository. The local configured default is GPT-6 Astra with ultra effort and full access/no routine approvals; remote defaults were verified and corrected for future launches, without restarting existing work.

## Hazards

- Shared working tree has pre-existing staged deletions, Experiment03 rename/edits and paper-introduction edits. Preserve them; commit only explicitly named task-owned files.
- Source has two earlier commits ahead of origin/main. They were inspected for potential secrets; this work has not rewritten them.
- Three training seeds are a descriptive screen: an exact two-sided sign test cannot reach p<0.25. No claim of optimal training or state-of-the-art superiority.
- Raw data/models/checkpoints and host receipts remain private/ignored. Shared remote storage is tight; use the verified node-local runtime.

## Owned work

- [Experiment04](04_validation_guided_zipmix/README.md): 27 from-scratch cells; [checkpoint](04_validation_guided_zipmix/CKPT_zipmix.md), [live results](04_validation_guided_zipmix/results.md). Training harness/data ready, 20 tests passed; engineering GPU timing complete.
- [Experiment05](05_validation_guided_sft/README.md): 18 fine-tuning cells plus one unchanged-base evaluation; [checkpoint](05_validation_guided_sft/CKPT_validation_guided_sft.md), [live results](05_validation_guided_sft/results.md). Data ready, 25 tests passed; disposable full-precision-master GPU optimizer check complete.
- Implementation/research/compute helper agents finished and froze source. The requested reviewer has authority for minimal scoped corrections and deterministic verification; no second review is authorized.

## Watches and completion

The ordinary Python supervisor survives coordinator disconnection, owns one process group and emits fresh process/progress/device receipts. It admits 27 or18 expected training cells respectively and checks the separate fine-tuning baseline. Run sequentially on one device. Deadline ceilings:12hours pretraining,6hours fine-tuning; reboot recovery is not implemented. No measured job has been launched yet, so no running-job monitoring is claimed.

## Decisions and next actions

No user decision is waiting. Finish the exact requested review, apply its findings, run deterministic checks, inspect staged diffs and publish only owned files. Synchronize the verified revision and19 transferred input-file hashes. Launch and supervise the complete frozen manifests, analyze all cells, preserve failures, update live numerical reports, and verify resource release. Modern larger-scale comparisons and reinforcement learning remain a research roadmap, not completed results.

## Landing

Initial base c672ee7; task commit/publication pending the requested review. Private runtime launch identities must be added to the private receipt after actual launch, and public status updated here.

Reviewer decision: keep Experiment05 v1 as untrained calibration. The prospective v2 packs scoring text to exactly4,096 bytes and adds wrong-target development controls. Do not launch the old short-string v1 fine-tuning run. Execute reviewed Experiment04 independently; completion of the underlying scientific goal remains unresolved.
