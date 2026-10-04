# Zip-Mix: 57 completed training runs; limited fine-tuning targeting signal, competitive superiority unresolved

**Doc link:** <https://github.com/brando90/zip-mix/blob/main/experiments/CKPT_MASTER_tmuxnone_cxd_01a1081e.md>

**TLDR:** All 27 pretraining, 21 main fine-tuning and nine contrastive models completed cleanly, plus nine pretraining reference/proxy stages and two base evaluations. Zip-Mix shows exploratory gains over shuffled and wrong-target fine-tuning controls, but its compact pretraining comparator DoGE and the unchanged fine-tuning backbone remain stronger in the relevant comparisons. All owned training processes have exited; complete numerical evidence is exported and independently audited.

Updated: 10-04-2026 after complete result verification.

## Identity and recovery

Local Codex coordinator for Zip-Mix. Exact host, process, terminal-session and resume identities remain in the private launch and terminal-verification receipts. Global local instruction files are nonempty and resolve to the current shared rules; remote defaults were corrected for future launches without restarting existing agents. The requested single quality assurance review actually used `claude-opus-5-5` and the explicit maximum-effort flag through existing subscription authentication. Its original report and a later coordinator disposition are preserved [here](04_validation_guided_zipmix/qa/opus55_max_review.md).

## Ownership

Preserve pre-existing staged deletions, Experiment 03 rename/edits, paper edits and Experiment 02 materials. Only explicit task-owned paths are committed. The frozen execution sources are `14c4d4f` for Experiment 04, `611423b` for Experiment 05 version 2, and `ab85f46` for Experiment 06. Raw data, model weights, checkpoints and host receipts remain ignored/private; no private host material is published.

## Completed scientific state

- [Experiment 04](04_validation_guided_zipmix/results.md): 27/27 final models, six proxy stages and three reference stages completed, zero failures or resumes. The prespecified improvement criterion was not met. Zip-Mix minus compact Domain Reweighting with Generalization Estimation (DoGE) has target negative-log-likelihood difference +0.41265 [0.39285, 0.43246] nats/token, exact sign-test p-val=0.25; lower is better. The full arrays, perplexity report and bin-capacity diagnosis are preserved. All 23 deterministic tests passed.
- [Experiment 05](05_validation_guided_sft/results.md): 21/21 version 2 models plus base completed, zero failures or resumes. Zip-Mix minus shuffled alignment is +1.867 [0.432, 3.301] percentage points on science accuracy; minus wrong-target alignment is +3.133 [1.699, 4.568]; both p-val=0.25. Every trained model scored below the unchanged base on science. Version 1 remains untrained; version 2 prospectively repaired unequal-byte compression scoring. All 38 tests passed; 69 original proof files are byte-identical after export.
- [Experiment 06](06_contrastive_validation_zipmix/results.md): 9/9 models plus base completed, zero failures or resumes. Contrastive Zip-Mix minus the source-matched control is −1.733 [−10.625, 7.159] percentage points, raw and Holm-adjusted p-val=1.0. The follow-up was designed before any Experiment 05 outcome was inspected. All 35 tests passed; 33 original proof files are byte-identical after export.

Intervals are descriptive 95% Student-t intervals over three paired training seeds, conditional on fixed evaluation data. Tests are exact two-sided sign tests of equal winning probability among non-tied seeds. Three pairs cannot yield p<0.25. These screens do not establish confirmatory significance, optimality or state-of-the-art superiority. Fine-tuning comparisons reuse 500 target and 500 retention questions; they do not represent independent benchmark replications.

The [cross-study report](06_contrastive_validation_zipmix/cross_experiment/comparison.md) independently audits every saved prediction, shared data identity and training budget for all 30 fine-tuned models and both base evaluations. All seven cross-study target contrasts are explicitly exploratory, with raw and Holm-adjusted p-values. Its five deterministic corruption/completeness tests passed. No model was fitted or evaluated again during reanalysis.

## Execution completion and costs

All three full frozen manifests completed under their original admitted settings, without numerical failures, infrastructure interruptions or recovery. Training and analysis exited successfully, canonical run-bound status rows matched, and independent exact-process checks found no owned remnants. Resource release does not require other users to leave a formerly used device empty.

Experiment 04's measured invocation was 1,321.160 seconds including its reference/proxy stages. Experiment 05's supervised allocation lasted 6,463.014 seconds; Experiment 06's lasted 1,348.092 seconds. These elapsed measurements include checkpoint writes and other waits; their boundaries differ and they do not isolate pure compute or monetary spend. Fine-tuning used 43,008 and 18,432 answer-label tokens respectively. Full stage accounting, limitations and source identities are in the individual reports. Dollar costs are unavailable.

## Next research decision

The authorized initialization and full pilot campaign are complete. The result supports a limited development-targeting signal, not the user's hoped-for broad superiority claim. The current fine-tuning recipe first needs calibration on a separate training/validation partition with an unchanged-base control. Any follow-up must preserve these results and use fresh uninspected evaluation questions rather than tune against the reported test scores.

The [next-stage proposal](04_validation_guided_zipmix/NEXT_STAGE_DESIGN.md) fixes joint source-by-compression exposure to test alignment within groups; it is documented but not launched. Establishing added value from bins also requires a fair direct-score comparison. Larger faithful learned-mixture reproductions, modern competitors and reinforcement learning remain prospective work. No healthy admitted work remains running, and no human-only completion step is waiting.

**TLDR-end:** [zip-mix: completed pilot campaign] All 57 planned training models and supporting stages completed and were audited. The fine-tuning targeting signal is exploratory; competitive superiority is unresolved, and training calibration precedes the next selector study.

**Snapshot:**
```text
pretraining_final_models=27/27
pretraining_reference_proxy_stages=9/9
main_finetuning_models=21/21
contrastive_finetuning_models=9/9
unchanged_base_evaluations=2/2
failed_missing_or_resumed_cells=0
owned_training_process_remnants=0
cross_study_report=complete
```
