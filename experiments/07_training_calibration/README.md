# Experiment 07: training calibration before selector admission

**Doc link:** <https://github.com/brando90/zip-mix/blob/main/experiments/07_training_calibration/README.md>

**TLDR:** Both stages ran on FarmShare (10-21-2026) and the result is a null. Stage A (job 1861877, 20 cells + base) passed its gate and selected lr 2e-6; Stage B (job 1862429, 15 cells + base) gave `advance=false`: the aligned selector beat uniform by +0.76 pp [−0.70, +2.22] and the permuted control by +0.80 pp [−0.32, +1.92] on 500 fresh held-out target questions (5 paired seeds), under the pre-registered +2 pp bar. A post hoc, not pre-registered sweep (Exp07b, job 1863035, 40 cells) finds no gain from unstratified or even oracle SciQ selection at this budget either; see [expt_v1/results/results_exp07b.md](expt_v1/results/results_exp07b.md). Fresh calibration and held-out questions were frozen and verified before any training.

- Results: [stage A decision](expt_v1/results/stage_a/decision.json), [stage B analysis](expt_v1/results/stage_b/analysis.json), per-cell ledgers and prediction archives under `expt_v1/results/`; the paper's Appendix A (`06_appendix_exp07_null.tex` in both drafts) summarises the null.

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

**TLDR-end:** [zip-mix: exp07] Stage A gate passed (lr 2e-6), Stage B complete with advance=false (+0.8 pp vs both controls, under the 2 pp bar); exploratory Exp07b shows little headroom for selection at this model and step budget.
