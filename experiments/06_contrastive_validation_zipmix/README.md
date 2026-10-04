# Experiment 06: contrastive validation-guided ZipMix

**Doc link:** <https://github.com/brando90/zip-mix/blob/main/experiments/06_contrastive_validation_zipmix/README.md>

**TLDR:** The nine-cell contrastive-selector study launched at 12:50 PDT on 10-04-2026 after 35 remote checks passed. The base evaluation is complete; benchmark results remain pending.

The pretraining [bin-capacity diagnosis](../04_validation_guided_zipmix/expt_v1/results/bin_capacity.md) found that fixed compression bins restrict target-source exposure. In the supervised pool, a positive true-minus-wrong development contrast yields 58.3210% SciQ sampling mass. This experiment asks whether that contrast helps beyond merely changing source proportions.

| Frozen selector | Definition | SciQ mass |
|---|---|---:|
| `contrastive_zipmix` | Positive pack-level true-minus-wrong maximum score, inherited per question, summed within existing bins | 58.3210% |
| `source_matched` | Exactly the same four source masses, uniform sampling inside each source | 58.3210% |
| `contrastive_shuffled` | Fixed permutation of positive pack contrasts before inheritance and bin aggregation | 31.3883% |

These are exact finite-pool probabilities, not benchmark estimates. The primary contrast with `source_matched` isolates selection within sources; the shuffled control tests score assignment while also changing source proportions. Total variation distances from the primary are 0.213562 and 0.394043 respectively.

The experiment reuses the frozen [Experiment 05 version 2 data](../05_validation_guided_sft/expt_v2/data_manifest.json): 9,797 questions, 351 equal-width compression packs, eight true and eight wrong development views, and the same 500-question SciQ target and 500-question CommonsenseQA retention sets. Necessary arrays and tokenizer files are copied byte-for-byte into the ignored local data snapshot; original data and results remain unchanged. Reuse and hashes are explicit in the new [manifest](expt_v1/data_manifest.json).

Training uses the same pinned Qwen2.5-0.5B base model, float32 master parameters with bfloat16 forward computation, optimizer, seeds 0/1/2, and 128 steps × 16 one-label questions. The reviewed trainer is byte-identical to Experiment 05 version 2. Checkpoints occur every 64 steps instead of 16, an operational change that reduces disk writes and makes cross-experiment wall-time comparisons imperfect. The nine cells plus one base evaluation have a separate identity and a maximum two-device-hour budget on the admitted single A100 graphics processing unit.

| Artifact | Purpose |
|---|---|
| [Protocol](expt_v1/PROTOCOL.md) | Frozen hypothesis, comparisons, accounting and interpretation |
| [Manifest](expt_v1/data_manifest.json) | Selector vectors' identity, source masses, inherited input hashes |
| [Execution prompt](expt_v1/cc.md) | Full nine-cell execution contract |
| [Checkpoint](CKPT_contrastive_validation.md) | Current handoff state |
| [Results](results.md) | Canonical pending/result record |

The prospective timestamp records that the selector design used no Experiment 05 benchmark outcomes. Comparisons against its 21 cells are deferred until both complete matrices exist and remain exploratory. This experiment shares the public held-out pools with its parent and is not an independent benchmark replication.

The [freeze receipt at the launch commit](https://github.com/brando90/zip-mix/blob/ab85f46074d0e479f39369cb0fe7fa0751876ef2/experiments/06_contrastive_validation_zipmix/expt_v1/freeze_receipt.json) binds the prospective source and initial documents. Live README, checkpoint and result documents subsequently evolve; training and analysis source identities stay frozen.
