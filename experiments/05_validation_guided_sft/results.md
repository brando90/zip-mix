# Experiment 05 live results

Updated 10-04-2026. Status: implementation checked and data frozen; measured training pending.

Scientific outcome: unmeasured. No claim of improved benchmark accuracy, broad generalization, optimality, or current-method superiority is supported yet.

| Required stage | Status |
|---|---|
| Pinned source/model revisions | Resolved before downloads |
| Data preparation and leakage audit | Complete: 10,116 train, 64 alignment, 500 target, 500 retention; zero residual exact-question or declared 13-word overlap hits |
| Deterministic implementation checks | 15 trainer/selection/resume tests and 12 analysis tests pass after review fixes |
| Requested Opus 5.5 maximum-effort review | Done: resume fix applied, length confound documented ([report](../04_validation_guided_zipmix/qa/opus55_max_review.md)) |
| Base checkpoint evaluation | Not run |
| Six methods × three seeds | 0/18 measured training cells completed |
| Final paired-seed analysis | Pending |

The canonical procedure is [expt_v1/PROTOCOL.md](expt_v1/PROTOCOL.md). Large datasets and checkpoints stay ignored in the version's data/run storage or a coordinator-documented scratch location.

Preparation took 34.46 seconds. The frozen public [manifest](expt_v1/data_manifest.json) has identity `d133f1242f8edb9827ddcd1c3ad404bac8b56f688943c0d943ff3063ce5e097a`. Compel selects 150 of 10,116 candidates; direct ZIP-FIT selects 2,529. Candidate compression views range from 43 to 850 bytes, so they are bounded views rather than equal-length strings. The target-source mixture mass barely changes under historical maximum-score ZipMix; a beneficial effect is not assumed.

**TLDR-end:** [zip-mix: SFT screen] Source, split, selection, and analysis artifacts are ready for the coordinator's review and launch; performance remains unmeasured.

**Snapshot:**
```text
model=Qwen/Qwen2.5-0.5B
methods=6
seeds=0,1,2
supervised_answer_tokens_per_cell=2048
expected_training_cells=18
measured_training_cells=0
```
