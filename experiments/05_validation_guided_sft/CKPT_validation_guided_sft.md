# Experiment 05 resumable checkpoint

**Doc link:** <https://github.com/brando90/zip-mix/blob/main/experiments/05_validation_guided_sft/CKPT_validation_guided_sft.md>

Updated 10-04-2026 after actual launch at 12:08 PDT. The full 21-cell run is active and its unchanged-base evaluation is complete; this is not yet full-matrix completion.

The parent coordinator authorized the initial public-data Qwen2.5-0.5B multiple-choice screen. One requested Opus 5.5 maximum-effort review identified a major short-string length confound before any training/evaluation. Version 1 remains preserved and untrained. The parent then authorized version 2's fixed-byte pack design and true/wrong-target control, with no additional model-review round.

Active version: `expt_v2/`. Seven methods × seeds 0, 1, 2 = 21 cells plus one base evaluation. Each cell uses 128 optimizer steps × 16 examples, one answer token per example, float32 master parameters, and bfloat16 forward autocasting. Model/tokenizer/data revisions and the old evaluation arrays remain pinned. The implementation worker must not launch measured training; the parent owns deployment and durable execution.

Prepared-data identity: `0d6dec93e7d7ccffd87f95e7ebd19e78c84ec72a546007b122888789aba5cb44`. Output is ignored under `expt_v2/data/`; the compact public copy is `expt_v2/data_manifest.json`. There are 9,797 training examples in 351 packs, eight true SciQ and eight reserved CommonsenseQA development views, and 500 examples in each evaluation split. All scoring views contain exactly 4,096 real bytes without padding/repetition. The 256 wrong-target source records and 63 trailing underfilled-pack records are excluded from training.

Independent deterministic checks reproduced all 367 view hashes and all 351 candidate compression/alignment scores, verified the parent/file manifests and both preparation-source hashes, and confirmed all seven methods have support. The final test suite includes training-resume/precision checks, analysis auditing, and new pack/target-control invariants. Final deterministic suite: 38/38 tests passed in 4.60 seconds; compilation and whitespace checks passed. Frozen source commit `611423b` is pushed and deployed; 38 remote tests also passed.

Exact finite-pool prior diagnostics: true-vs-wrong total variation 0.1395318171, true-vs-shuffled 0.0650869769; true-target SciQ mass 0.3586768058 versus wrong-target 0.2693312755. These demonstrate a changed training intervention, not improved accuracy. Source and pack-density effects remain.

Next resume steps: inspect the private launch receipt and fresh supervisor status, allow every healthy admitted cell to finish, verify terminal process/resource release, analyze all saved predictions and publish compact proof artifacts. All source/input hashes and run bindings were verified at launch; exact host/process identities are in the private deployment receipt. Never launch version 1's obsolete training plan.

**TLDR-end:** [zip-mix: SFT v2 checkpoint] Version 2 repairs the identified scoring-length confound and provides distinct true/wrong-target mixtures; the coordinator now owns complete 21-cell execution and measured result reporting.

**Snapshot:**
```text
active_version=expt_v2
model=Qwen/Qwen2.5-0.5B
train_examples=9797
scoring_bytes=4096
available_training_cells=21
coordinator_launch=10-04-2026 12:08 PDT
base_evaluation=complete
full_matrix=running
```
