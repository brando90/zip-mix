# Experiment 05 completion checkpoint

**Doc link:** <https://github.com/brando90/zip-mix/blob/main/experiments/05_validation_guided_sft/CKPT_validation_guided_sft.md>

Experiment 05: Qwen2.5-0.5B supervised fine-tuning across seven mixtures and three seeds completed all 21 cells plus the base evaluation; artifact export, independent reanalysis and benchmark reporting are complete.

Updated 10-04-2026 at 13:57 Pacific Daylight Time (PDT) after independent terminal verification. The run started at 12:08:58 PDT and reached verified procedure completion at 13:56:41 PDT: 21/21 training cells and 1/1 base evaluation complete, with zero failed, missing, or interrupted cells and zero infrastructure resumes.

The parent coordinator authorized the initial public-data Qwen2.5-0.5B multiple-choice screen. One requested Opus 5.5 maximum-effort review identified a major short-string length confound before any training/evaluation. Version 1 remains preserved and untrained. The parent then authorized version 2's fixed-byte pack design and true/wrong-target control, with no additional model-review round.

Completed version: `expt_v2/`. Seven methods × seeds 0, 1, 2 = 21 cells plus one base evaluation. Each cell uses 128 optimizer steps × 16 examples, one answer token per example, float32 master parameters, and bfloat16 forward autocasting. Model/tokenizer/data revisions and the old evaluation arrays remained pinned, and the checkpoint interval remained 16 steps throughout execution.

Prepared-data identity: `0d6dec93e7d7ccffd87f95e7ebd19e78c84ec72a546007b122888789aba5cb44`. Prepared inputs are ignored under `expt_v2/data/`; the compact public copy is `expt_v2/data_manifest.json`. There are 9,797 training examples in 351 packs, eight true SciQ and eight reserved CommonsenseQA development views, and 500 examples in each evaluation split. All scoring views contain exactly 4,096 real bytes without padding/repetition. The 256 wrong-target source records and 63 trailing underfilled-pack records are excluded from training.

Independent deterministic checks reproduced all 367 view hashes and all 351 candidate compression/alignment scores, verified the parent/file manifests and both preparation-source hashes, and confirmed all seven methods have support. The final test suite includes training-resume/precision checks, analysis auditing, and new pack/target-control invariants. Final deterministic suite: 38/38 tests passed in 4.60 seconds; compilation and whitespace checks passed. Frozen source commit `611423b` is pushed and deployed; 38 remote tests also passed.

Exact finite-pool prior diagnostics: true-vs-wrong total variation 0.1395318171, true-vs-shuffled 0.0650869769; true-target SciQ mass 0.3586768058 versus wrong-target 0.2693312755. These demonstrate a changed training intervention, not improved accuracy. Source and pack-density effects remain.

Terminal identity: source commit `611423be2e62531453a0ad4e3c496a5cd07c711c`, run `199d165af58bd69d9c48349f28f96e7d83a73ba0aa429949bd12756470c850e9`. The exact source and run identity matched at completion. Training and analysis each exited with code 0; the analyzer reported full procedure completion with zero audit errors, and the supervisor verified clean completion. The canonical board showed 21/21 complete with matching run receipts and exited coordinator/driver identities.

Measured allocation: one NVIDIA A100 80 GB device for 6,463.014 seconds of supervised wall time (1 hour 47 minutes 43.0 seconds), equivalent to 1.7953 device-hours. Training plus analysis spanned 6,456.469 seconds. This allocation measurement includes checkpoint writes and other waits; it is not a measurement of active compute time. Independent release verification found zero task-owned process remnants, and the assigned device showed zero memory in use and 0% utilization. Exact host/process identities remain private.

Sanitized export and independent reanalysis are complete: all 69 original proof files match their source hashes. The [export receipt](expt_v2/results/measured/EXPORT_RECEIPT.json) records completion, source identity, local reanalysis checks, and resource release; the [export note](expt_v2/results/measured/EXPORT.md) documents its numerical tolerances. The [complete report](results.md) and [cross-study comparison](../06_contrastive_validation_zipmix/cross_experiment/comparison.md) are finalized. Zip-Mix minus shuffled alignment is +1.867 [0.432, 3.301] percentage points on science accuracy; minus the wrong target is +3.133 [1.699, 4.568]. These are descriptive 95% Student-t intervals over three paired seeds; both exact two-sided sign tests against equal winning probability have p-val=0.25. All trained science scores remain below the unchanged base. No comparative benchmark superiority is established. Version 1 remains preserved and untrained.

**TLDR-end:** [zip-mix: fine-tuning v2 checkpoint] Version 2 completed all 21 training cells and the base evaluation without failures or resumes; analysis and resource-release checks passed. Sanitized export, independent reanalysis and full benchmark reporting are complete; the targeting signal remains exploratory.

**Snapshot:**
```text
version=expt_v2
model=Qwen/Qwen2.5-0.5B
training_cells=21/21_complete
base_evaluation=1/1_complete
failed_missing_interrupted_cells=0
infrastructure_resumes=0
supervised_wall_seconds=6463.014
training_analysis_exit_codes=0,0
owned_process_remnants=0
artifact_export_and_reanalysis=complete
benchmark_reporting=complete
```
