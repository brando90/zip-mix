# Experiment 06 checkpoint

**Doc link:** <https://github.com/brando90/zip-mix/blob/main/experiments/06_contrastive_validation_zipmix/CKPT_contrastive_validation.md>

**TLDR:** The full nine-cell study and base evaluation completed cleanly on 10-04-2026 at 13:13 PDT, with zero failed/missing cells or infrastructure resumes. The planned comparison does not support an alignment benefit: −1.73 [−10.63, 7.16] target-accuracy percentage points versus matching source proportions, exact sign-test p-val=1.0.

Execution source: `ab85f46074d0e479f39369cb0fe7fa0751876ef2`. Run identity: `b67673d75e6d5e9ac6bf962cfe65202d095e43c02262b35063717f994aaf35a5`. Private execution receipts bind the actual device, process, terminal session, monitor and canonical board row. All 35 remote checks, 12 input files, two preparation sources and 14 freeze-receipt hashes passed.

- Scope: three fixed selectors × seeds 0/1/2 = 9 training cells, plus one base evaluation; same model and frozen pool as Experiment 05 version 2.
- Data identity: `242d063db24573adc525b7fb523cc72bc0b6a01cb7b4f36248b823f93bccc1a8`.
- Parent identity: `0d6dec93e7d7ccffd87f95e7ebd19e78c84ec72a546007b122888789aba5cb44`.
- Trainer source SHA-256: `d5ec9f607bc5f5d4d7c8f480e9893056532f73b30f689975df8cece5223f2d1e`, byte-identical to Experiment 05 version 2.
- Operational change: checkpoint every 64 instead of 16 steps, ordinary persistent local ext4 output. All learning/optimizer settings and 128 updates are unchanged. Maximum budget: two device-hours.
- Information boundary: no Experiment 05 benchmark metric or prediction was inspected while designing/preparing this condition. Preparation timestamp is in the manifest; held-out arrays were copied by bytes only.
- Validation: all 35 selector, artifact-accounting, analyzer, precision, padding and checkpoint-recovery tests passed in 5.46 seconds. Compilation and whitespace checks also passed before handoff.
- Terminal evidence: training and analysis exited 0; supervisor and sequencer verified 9/9 cells plus base. Wall time 1,348.092 seconds. Exact owned process identities and all remnants exited; independent device inspection found 0 MiB used.
- Evidence: all 33 remote proof files preserved byte-for-byte; local reanalysis agrees except for at most 1.45 × 10⁻¹² interval-endpoint drift. Figure visually inspected. Canonical results and export provenance are under this home.
- Next: keep Experiment 05 healthy until all 21 cells complete, then run the already-published exploratory cross-study analyzer. No new training or parameter choice follows from these test results.

Preserve the old matrices and inputs. Do not stop healthy admitted work at a coordinator boundary, tune using held-out scores, or omit failed/interrupted cells. Update this checkpoint and canonical results when verified run evidence changes.
