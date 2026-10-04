# Version 1 status: preserved untrained calibration

**Doc link:** <https://github.com/brando90/zip-mix/blob/main/experiments/05_validation_guided_sft/expt_v1/STATUS.md>

Updated 10-04-2026. No measured training or held-out benchmark evaluation was launched for this version.

The user-requested single Opus 5.5 maximum-effort review found a major scientific limitation: short raw question strings yielded compression ratios and alignment scores dominated by byte length, with the ZipMix prior nearly reproduced by the length-matched permutation control. The recorded data and protocol remain useful calibration evidence, but they do not provide a meaningful validation-target mechanism test.

The coordinator authorized a separately identified prospective repair in [version 2](../expt_v2/PROTOCOL.md), which uses equal 4,096-byte real-content packs and a reserved wrong-target development control. Preserve this version's inputs, manifest, and source; do not run its training manifest as though the concern were fixed here.

**TLDR-end:** [zip-mix: SFT v1] This version is an untrained calibration preserved after the requested review; version 2 owns the prospective repaired experiment.
