# Experiment 05 resumable checkpoint

Created 10-04-2026; updated after implementation and preparation. This file is a work checkpoint, not evidence of completed training.

The parent coordinator authorized the public-data Qwen2.5-0.5B multiple-choice screen with six methods, three seeds, 128 steps × 16 examples per cell, and one base evaluation. The implementation worker owns only Experiment 05 and must not launch measured training. The parent owns deployment and the single requested Opus 5.5 maximum-effort review across Experiments 04 and 05.

Source revisions and the exact configuration are fixed in `expt_v1/common.py`. Local preparation is invoked with `python3 experiments/05_validation_guided_sft/expt_v1/prepare_sft.py --workers 4`; output is ignored under `expt_v1/data/`. Tokenizer verification found distinct one-token answer continuations: `[362, 425, 356, 422, 468]` for space-prefixed A–E. Preparation asserts the full prompt boundary for every item.

Prepared data identity: `d133f1242f8edb9827ddcd1c3ad404bac8b56f688943c0d943ff3063ce5e097a`; 10,116 training examples, 64 alignment views, 500 target and 500 retention examples. Preparation completed in 34.46 seconds, with no residual exact-question or declared 13-word overlap hits. The public manifest is `expt_v1/data_manifest.json`; raw data remain ignored. Twelve trainer/selection tests and twelve analysis tests passed independently. A final precision-load assertion is being checked with the combined suite.

The trainer uses float32 master parameters and optimizer states, with bfloat16 forward autocasting. The remote preflight worker verified a disposable synthetic optimizer step: finite loss/gradient norm and all 1,024 sampled weights changed; no benchmark evaluation or training checkpoint was created. This is engineering verification only.

Next resume steps: obtain the parent's recorded review outcome; synchronize all final hashes and prepared data; launch only through the coordinator's durable remote owner; update live records. Record exact runtime timestamps and private run paths at launch. No measured model result currently exists.

**TLDR-end:** [zip-mix: SFT checkpoint] Implement and verify the frozen screen, then hand its complete source and prepared inputs to the parent for review and launch; training and benchmark gains remain pending.

**Snapshot:**
```text
expected_training_cells=18
base_evaluation_required=true
training_launched_by_implementation_agent=false
answer_label_tokens=362,425,356,422,468
```
