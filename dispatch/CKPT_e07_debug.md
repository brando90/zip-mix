# CKPT: Exp 07 null-result debug (worker cc-zipmix-e07-debug-10-21-2026)

Created: 10-21-2026 08:36 PDT
Last updated: 10-21-2026 08:49 PDT

## Goal
- Why does Experiment 07 (calibrated conditional SFT Zip-Mix) give a null result; fix genuine bugs on branch `codex-zipmix-e07-debug-10-21-2026`; run one smallest prospective diagnostic (≤30 GPU-min); preserve the original run and main.

## What ran
- 10-21-2026 15:19Z attempt 3 started (Claude `claude-fable-5-1`, effort max, `--dangerously-skip-permissions`); ACK.md written 15:21Z.
- Discovered the actual Exp 07 run on GitHub branch `calib/exp07-fresh-calibration` (`experiments/07_training_calibration`, commits 965bb9e..447e8a5, FarmShare L40S jobs 1861877/1862429). Extracted read-only copy to `../orig_branch_ro/` (outside the repo). The locally transferred folder `experiments/07_calibrated_conditional_sft_zipmix` is a different, never-run implementation.
- Deterministic checks on local frozen data: 18/18 file hashes, manifest id, selection vectors reproduce (max abs diff 2.7e-20); zero id/question overlaps across 8 splits.
- Patched local trainer: (1) missing-file hash crash, (2) missing `method_status` KeyError, (3) selection.npz metadata arrays validated as weights → abort; added diagnostic mode and `test_exp07.py` (13 tests pass on CPU).

## Results
- Original run (frozen, branch): Stage A selected lr=2e-6 (cal_target 0.8072 vs base 0.800; prediction flips 6.7%); Stage B aligned−uniform +0.76 pp [−0.70, 2.22] sign p=1.0, aligned−permuted +0.80 pp [−0.32, 1.92] sign p=0.625 on holdout_target (n=500, 5 seeds); advance=false.
- Diagnostic (lr 5e-6, 3 selectors × 5 seeds, local cal_target N=500): aligned − uniform +1.04 pp [−0.94, +3.02] 4/5 sign p=0.375; aligned − shuffled +1.28 pp [−1.67, +4.23] 4/5 sign p=0.375; predictions changed 11–12% vs base; advance rule not met; inside the pre-registered +1.0 to +1.6 pp expectation → inconclusive band, movement scaling supported. 15/15 cells, 471 s wall, 377 s summed training on GPU 1.

## Pre-registered diagnostic (recorded before launch)
- Config: local frozen data (manifest c01070d7…), `train_exp07.py --mode diagnostic --learning-rate 5e-6`, methods conditional_aligned / stratum_uniform / conditional_shuffled × seeds 0–4 = 15 cells + base; eval on cal_target + cal_retention only (fresh official-split holdouts target/retention/test_holdout2 are never loaded). Device: skampere2 GPU 1 (0% util, ~60 GB free), `CUDA_VISIBLE_DEVICES=1`, hard `timeout 1800`.
- Learning-rate rule (fixed before measuring): the largest Stage A grid point whose 5-seed mean cal_target was within the protocol's 0.2 pp tie band of the base (5e-6: 0.7988 vs 0.800), i.e. the most model movement without degradation; not chosen from Stage B outcomes.
- Readouts: paired aligned−control on cal_target (mean, Student-t 95% CI over 5 seeds, exact sign test, Holm over 2 contrasts); prediction-flip rate vs base per cell; cal_target vs base.
- Quantitative expectation (recorded before any diagnostic number): Exp 05 selector contrasts of +1.9 to +3.1 pp occurred at lr 2e-5 (23.2% predictions changed vs base in Stage A); the original Stage B at 2e-6 changed 6.7% and showed +0.8 pp. If the contrast scales with movement, 5e-6 (11.8% changed) should give roughly +1.0 to +1.6 pp, i.e. still under the 2 pp advance bar.
- Interpretation: (1) both contrasts ≥ +2 pp with ≥4/5 positive seeds → selector effect is expressible at a non-degrading rate, so the original null is plausibly a consequence of the calibration rule picking the least-moving rate; (2) contrasts < +1 pp or mixed sign → the null is substantive at every non-degrading grid rate; (3) otherwise inconclusive. Exploratory either way; no paper claim changes.

## Stuck / blocked
- nothing

## Next steps
1. Commit the branch (`git add experiments/07_calibrated_conditional_sft_zipmix dispatch && git commit`), push `codex-zipmix-e07-debug-10-21-2026` only; write `dispatch/COMPLETE.md`.
2. Coordinator: review `DEBUG_REPORT.md`; decide whether a separately frozen prospective version with a calibration rule that trades a bounded target drop for more movement is worth running (not authorized here).
3. Cleanup owner: coordinator (parent chat) removes `/lfs/skampere2/0/brando9/snap_jobs/codex-zipmix-e07-debug-10-21-2026/` after the branch is verified on GitHub; `orig_branch_ro/` and `expt_v1/runs_diag/` are regenerable.

## Brando to-do
- none

## Log
- 10-21-2026 08:36 PDT -- checkpoint created; original run located on branch calib/exp07-fresh-calibration; trainer patched; diagnostic pre-registered, launching next.
- 10-21-2026 08:40 PDT -- bug 3 found by the first diagnostic launch (no GPU time consumed beyond model load); fixed + regression test; diagnostic relaunched (15 cells + base, lr 5e-6, GPU 1, timeout 1800 s). Per-item analysis of original run: arms differ (5.1–5.6% same-seed disagreement) about as much as seeds within an arm (5.3–5.6%); pooled discordant items favour aligned 61/42 vs uniform, 69/49 vs permuted (binomial p≈0.08).
- 10-21-2026 08:49 PDT -- diagnostic complete 15/15 (exit 0, GPU released); results, report §7, results.md updated; next: commit + push branch, completion marker.
