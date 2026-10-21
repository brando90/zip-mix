# Exp 07 debug: live results

**Status:** COMPLETE (started 10-21-2026 15:19Z; diagnostic finished 15:47Z)
**Last updated:** 10-21-2026 08:49 PDT

| Phase | State |
|---|---|
| Locate original Exp 07 run and read it end to end | DONE (branch `calib/exp07-fresh-calibration`, 35 cells + 2 base evaluations, all complete) |
| Deterministic checks (a) labels/splits, (d) arms distinct | DONE: local frozen data (18/18 hashes, 0 overlaps, selectors reproduce) and original per-item predictions (decision.json and analysis.json reproduce exactly) |
| Fix genuine bugs on this branch + tests | DONE: 3 blocking bugs in the never-run local trainer fixed; 13/13 CPU tests pass; no correctness bug found in the original run's code |
| Smallest prospective diagnostic (≤30 GPU-min) | DONE: 15/15 cells + base at lr 5e-6, 471 s wall on skampere2 GPU 1 (377 s summed training), 13.5 GB; receipts in `experiments/07_calibrated_conditional_sft_zipmix/debug_10-21-2026/diagnostic_lr5e-6/` |
| DEBUG_REPORT.md | DONE: `experiments/07_calibrated_conditional_sft_zipmix/debug_10-21-2026/DEBUG_REPORT.md` (copy in `dispatch/DEBUG_REPORT.md`) |

## Original run (frozen evidence, not modified)
Stage A (fresh calibration split, 4 learning rates × 5 seeds, uniform sampling), base cal_target 400/500 = 80.0%, cal_retention 260/500 = 52.0%:

| lr | cal_target mean % (5 seeds) | sd pp | cal_retention mean % | predictions changed vs base |
|---|---:|---:|---:|---:|
| 2e-5 | 73.52 | 2.27 | 55.24 | 23.2% |
| 1e-5 | 77.80 | 2.05 | 59.80 | 15.4% |
| 5e-6 | 79.88 | 0.91 | 60.28 | 11.8% |
| 2e-6 (selected) | 80.72 | 0.56 | 57.24 | 6.7% |

Stage B at lr 2e-6 (3 selectors × 5 seeds), holdout_target base 394/500 = 78.8%: conditional_signed_rank 80.76%, uniform_stratum 80.00%, permuted_stratum_rank 79.96%. Contrasts aligned−uniform +0.76 pp [−0.70, 2.22] (sign p=1.0, 3/5 positive), aligned−permuted +0.80 pp [−0.32, 1.92] (sign p=0.625, 3/5 positive, 1 tie). Advance rule (≥2 pp and ≥4/5 positive) not met → advance=false.

## Original run, per-item evidence (holdout_target, N=500, 5 seeds)
- Predictions changed vs base: 6.5–6.9% per arm. Same-arm cross-seed disagreement 5.3–5.6%; same-seed cross-arm disagreement 5.1–5.6%: the selector moves predictions no more than the sampling seed does.
- Pooled discordant items, aligned vs uniform 61 wins / 42 losses (binomial p=0.076); aligned vs permuted 69/49 (p=0.080). Item-paired SE 0.34–0.44 pp.
- Aligned fixes 17.2% of the 106 base-wrong items and breaks 2.1% of the 394 base-right items; controls 15.3–15.8% and 2.6–2.7%.

## Diagnostic (pre-registered in CKPT_e07_debug.md before launch; lr 5e-6, 3 selectors × 5 seeds, local frozen pool, eval on cal_target/cal_retention only)
First launch 15:38Z aborted at once on local-trainer bug 3 (no training); relaunched 15:45Z after the fix; complete 15:47Z.

| cal_target (N=500; base 399/500 = 79.80%) | mean % (5 seeds) | sd pp | predictions changed vs base |
|---|---:|---:|---:|
| conditional_aligned | 81.16 | 0.85 | 11.9% |
| stratum_uniform | 80.12 | 1.41 | 11.0% |
| conditional_shuffled | 79.88 | 2.22 | 11.4% |

| aligned minus control, cal_target | mean pp [95% t-interval, 5 paired seeds] | positive | exact sign p | Holm p | discordant items win/loss (pooled) |
|---|---|---:|---:|---:|---|
| uniform | +1.04 [−0.94, +3.02] | 4/5 | 0.375 | 0.75 | 132/106 (binomial p=0.105) |
| shuffled | +1.28 [−1.67, +4.23] | 4/5 | 0.375 | 0.75 | 126/94 (binomial p=0.036) |

Retention (cal_retention, base 274/500 = 54.80%): aligned 57.20%, uniform 57.20%, shuffled 55.88%; aligned − shuffled +1.32 pp [+0.50, +2.14], 5/5 positive, sign p=0.0625 (secondary, unadjusted). Original advance rule (≥2 pp and ≥4/5 positive against both controls) on cal_target: **not met**. Pre-registered expectation was +1.0 to +1.6 pp: observed +1.04 and +1.28. Verdict by the recorded rules: inconclusive band; the contrast scales with model movement (6.7% → 11–12% changed predictions, +0.8 → +1.0/+1.3 pp) but does not reach the bar at a non-degrading rate. Exploratory; no paper claim changes.

## Bugs fixed on this branch (none in the code that produced the null)
1. `train_exp07.py` hashed the never-shipped `analyze_exp07.py` at startup (crash in every mode). 2. Stage 2 modes read a `method_status` key the frozen manifest does not have (KeyError). 3. Metadata arrays in `selection.npz` were validated as probability vectors (`Invalid frozen weights: domain`, observed live). Plus diagnostic mode and `expt_v1/test_exp07.py`: 13 passed.

**TLDR-end:** [zip-mix: exp07 null debug] The Exp 07 null is a +0.8 pp within-stratum alignment contrast at the calibrated rate 2e-6, explained by the pre-registered rule choosing the least-moving rate, with no bug in the code that produced it; the 5e-6 diagnostic (15 cells, 8 GPU-min) doubles movement and contrast to +1.0/+1.3 pp, still under the 2 pp bar, so the null stands.
