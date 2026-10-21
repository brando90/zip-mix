# Experiment 07: training calibration before selector admission

**Doc link:** <https://github.com/brando90/zip-mix/blob/main/experiments/07_training_calibration/README.md>

**TLDR:** Prepared 10-18-2026, not run. Fresh calibration and held-out questions are frozen and verified; Stage A must find a fine-tuning recipe that at least matches the unchanged base before Stage B tests target alignment within fixed source × compression strata.

- Protocol: [expt_v1/PROTOCOL.md](expt_v1/PROTOCOL.md) (frozen rules, gates, seeds, resources).
- Data manifests: [calibration](expt_v1/calibration_manifest.json), [Stage B](expt_v1/stage_b_manifest.json). Prepared arrays stay private under `expt_v1/data/` or the launcher's work directory.
- Code: `prepare_calibration.py` (CPU data freeze), `train_calibration.py` (both stages, `--smoke` on CPU), `analyze_calibration.py` (predeclared decision and analysis), `test_calibration.py` (21 tests).
- Launch: [expt_v1/launch/README.md](expt_v1/launch/README.md) for SNAP, a 1× A100 cloud box or Modal.
- Checkpoint: [CKPT_training_calibration.md](CKPT_training_calibration.md).

Reproduce the CPU checks from this directory's `expt_v1/`:

```bash
python prepare_calibration.py --output data       # about 2 minutes; needs the Hugging Face Hub
python -m pytest -q test_calibration.py           # 21 passed on CPU
python train_calibration.py --smoke --data data/calibration --output runs/smoke_a
python analyze_calibration.py --run runs/smoke_a --stage calibration
```

**TLDR-end:** [zip-mix: exp07] Calibration and held-out data are frozen and tested on CPU; GPU stages wait for a launch.
