# Experiment 06 checkpoint

**Doc link:** <https://github.com/brando90/zip-mix/blob/main/experiments/06_contrastive_validation_zipmix/CKPT_contrastive_validation.md>

Updated 10-04-2026. State: prepared; awaiting parent resource/source verification and admission. No Experiment 06 GPU process was launched by the preparation worker.

- Scope: three fixed selectors × seeds 0/1/2 = 9 training cells, plus one base evaluation; same model and frozen pool as Experiment 05 version 2.
- Data identity: `242d063db24573adc525b7fb523cc72bc0b6a01cb7b4f36248b823f93bccc1a8`.
- Parent identity: `0d6dec93e7d7ccffd87f95e7ebd19e78c84ec72a546007b122888789aba5cb44`.
- Trainer source SHA-256: `d5ec9f607bc5f5d4d7c8f480e9893056532f73b30f689975df8cece5223f2d1e`, byte-identical to Experiment 05 version 2.
- Operational change: checkpoint every 64 instead of 16 steps, ordinary persistent local ext4 output. All learning/optimizer settings and 128 updates are unchanged. Maximum budget: two device-hours.
- Information boundary: no Experiment 05 benchmark metric or prediction was inspected while designing/preparing this condition. Preparation timestamp is in the manifest; held-out arrays were copied by bytes only.
- Validation: all 35 selector, artifact-accounting, analyzer, precision, padding and checkpoint-recovery tests passed in 5.46 seconds. Compilation and whitespace checks also passed before handoff.
- Next: parent verifies exact source/data hashes and resource headroom, then admits all nine cells under its durable supervisor. The preparation worker does not launch training.

Preserve the old matrices and inputs. Do not stop healthy admitted work at a coordinator boundary, tune using held-out scores, or omit failed/interrupted cells. Update this checkpoint and canonical results when verified run evidence changes.
