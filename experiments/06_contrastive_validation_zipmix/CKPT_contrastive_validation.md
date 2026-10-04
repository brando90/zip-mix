# Experiment 06 checkpoint

**Doc link:** <https://github.com/brando90/zip-mix/blob/main/experiments/06_contrastive_validation_zipmix/CKPT_contrastive_validation.md>

**TLDR:** The complete nine-cell study launched at 12:50:58 PDT on 10-04-2026, with base evaluation complete and zero failures at launch verification. Preserve the healthy frozen run through its full manifest.

Execution source: `ab85f46074d0e479f39369cb0fe7fa0751876ef2`. Run identity: `b67673d75e6d5e9ac6bf962cfe65202d095e43c02262b35063717f994aaf35a5`. Private execution receipts bind the actual device, process, terminal session, monitor and canonical board row. All 35 remote checks, 12 input files, two preparation sources and 14 freeze-receipt hashes passed.

- Scope: three fixed selectors × seeds 0/1/2 = 9 training cells, plus one base evaluation; same model and frozen pool as Experiment 05 version 2.
- Data identity: `242d063db24573adc525b7fb523cc72bc0b6a01cb7b4f36248b823f93bccc1a8`.
- Parent identity: `0d6dec93e7d7ccffd87f95e7ebd19e78c84ec72a546007b122888789aba5cb44`.
- Trainer source SHA-256: `d5ec9f607bc5f5d4d7c8f480e9893056532f73b30f689975df8cece5223f2d1e`, byte-identical to Experiment 05 version 2.
- Operational change: checkpoint every 64 instead of 16 steps, ordinary persistent local ext4 output. All learning/optimizer settings and 128 updates are unchanged. Maximum budget: two device-hours.
- Information boundary: no Experiment 05 benchmark metric or prediction was inspected while designing/preparing this condition. Preparation timestamp is in the manifest; held-out arrays were copied by bytes only.
- Validation: all 35 selector, artifact-accounting, analyzer, precision, padding and checkpoint-recovery tests passed in 5.46 seconds. Compilation and whitespace checks also passed before handoff.
- Next: keep all nine admitted cells healthy, require the terminal training/analysis gates and base completion, export only small numerical evidence, and verify release of the owned device processes. A separate completion-gated exporter and release observer are active.

Preserve the old matrices and inputs. Do not stop healthy admitted work at a coordinator boundary, tune using held-out scores, or omit failed/interrupted cells. Update this checkpoint and canonical results when verified run evidence changes.
