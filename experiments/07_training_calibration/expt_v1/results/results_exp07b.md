# Exp07b: why was the Exp07 Stage B result null? (EXPLORATORY)

Post hoc and not pre-registered. FarmShare job 1863035 (L40S, `debug_exp07b.py`, commit 447e8a5), data/stage_b of Exp07, 4 arms x LR {2e-6, 2e-5} x seeds 0-4 = 40 cells plus base. Every cell has a result.json (`results/exp07b/<arm>__lr<lr>__seed<s>/`). Analysis: `analyze_exp07b.py`.

## Takeaways

1. **Recipe sensitivity: yes on the target metric.** oracle_sciq beats anti_sciq on holdout_target by +0.031 [+0.017, +0.045] at LR 2e-6 and +0.059 [+0.040, +0.079] at LR 2e-5 (paired seeds, n=5). The training data matters.
2. **Unstratified global_tilt does not beat uniform.** Target: +0.008 [-0.011, +0.028] at 2e-6 and +0.001 [-0.047, +0.050] at 2e-5, both include 0. Retention is worse at 2e-6 (-0.014 [-0.022, -0.006]) and includes 0 at 2e-5 (-0.037 [-0.082, +0.007]). The tilt does move the data (SciQ share drawn 0.73 vs 0.305 for uniform) but not the outcome.
3. **Headroom is small, which is the likely reason for the null.** Even the upper bound oracle_sciq (100% SciQ) is only +0.017 [-0.002, +0.037] over uniform on target at 2e-6 and +0.000 at 2e-5. The oracle vs anti gap is about half anti_sciq being worse than uniform at 2e-6 (-0.014 vs oracle +0.017) and all of it at 2e-5 (anti -0.059, oracle +0.000). Same-arm seed disagreement is 0.056 (diagnose_null), so effects under about 0.012 are not resolvable here.
4. **LR 2e-5 is too high.** Uniform falls below the base model (target 0.764 vs 0.788, retention 0.507 vs 0.546); at 2e-6 uniform is 0.800 / 0.568. Any tilt effect at 2e-5 is swamped by damage and seed noise.
5. Oracle_sciq trades retention for target at both LRs (retention -0.013 at 2e-6, -0.086 at 2e-5 vs uniform).

Reading: Exp07's null is not a pipeline bug and not a missing-stratification artefact. The target-domain data does not offer much beyond uniform at this model and step budget, so a selection method has little room. Next step if pursued: a larger target gap (harder holdout, or a model further from SciQ) before more selector work.

40 cells; base model: holdout_target 0.7880, holdout_retention 0.5460

Per-arm mean accuracy over seeds (n = seeds) and paired-seed 95% t interval of the difference vs uniform (same seed).

### LR 2e-06

| arm | n | target mean | target diff vs uniform [95% CI] | retention mean | retention diff vs uniform [95% CI] | SciQ share drawn | train loss |
|---|---|---|---|---|---|---|---|
| uniform | 5 | 0.8000 | - | 0.5680 | - | 0.305 | 0.831 |
| global_tilt | 5 | 0.8084 | +0.0084 [-0.0114, +0.0282] | 0.5540 | -0.0140 [-0.0224, -0.0056] | 0.730 | 0.571 |
| oracle_sciq | 5 | 0.8172 | +0.0172 [-0.0024, +0.0368] | 0.5548 | -0.0132 [-0.0219, -0.0045] | 1.000 | 0.404 |
| anti_sciq | 5 | 0.7864 | -0.0136 [-0.0290, +0.0018] | 0.5712 | +0.0032 [-0.0146, +0.0210] | 0.000 | 0.968 |

Paired contrasts:

- oracle_sciq - anti_sciq (recipe sensitivity), holdout_target: +0.0308 [+0.0171, +0.0445] (excludes 0, n=5)
- oracle_sciq - anti_sciq (recipe sensitivity), holdout_retention: -0.0164 [-0.0407, +0.0079] (includes 0, n=5)
- global_tilt - uniform, holdout_target: +0.0084 [-0.0114, +0.0282] (includes 0, n=5)
- global_tilt - uniform, holdout_retention: -0.0140 [-0.0224, -0.0056] (excludes 0, n=5)

### LR 2e-05

| arm | n | target mean | target diff vs uniform [95% CI] | retention mean | retention diff vs uniform [95% CI] | SciQ share drawn | train loss |
|---|---|---|---|---|---|---|---|
| uniform | 5 | 0.7644 | - | 0.5068 | - | 0.305 | 1.086 |
| global_tilt | 5 | 0.7656 | +0.0012 [-0.0472, +0.0496] | 0.4696 | -0.0372 [-0.0817, +0.0073] | 0.730 | 0.821 |
| oracle_sciq | 5 | 0.7644 | +0.0000 [-0.0504, +0.0504] | 0.4208 | -0.0860 [-0.1270, -0.0450] | 1.000 | 0.665 |
| anti_sciq | 5 | 0.7052 | -0.0592 [-0.0965, -0.0219] | 0.5016 | -0.0052 [-0.0576, +0.0472] | 0.000 | 1.143 |

Paired contrasts:

- oracle_sciq - anti_sciq (recipe sensitivity), holdout_target: +0.0592 [+0.0395, +0.0789] (excludes 0, n=5)
- oracle_sciq - anti_sciq (recipe sensitivity), holdout_retention: -0.0808 [-0.1240, -0.0376] (excludes 0, n=5)
- global_tilt - uniform, holdout_target: +0.0012 [-0.0472, +0.0496] (includes 0, n=5)
- global_tilt - uniform, holdout_retention: -0.0372 [-0.0817, +0.0073] (includes 0, n=5)

