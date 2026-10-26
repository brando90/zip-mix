# Debug report: why Experiment 07 (calibrated conditional SFT Zip-Mix) returned a null result

**Doc link:** <https://github.com/brando90/zip-mix/blob/codex-zipmix-e07-debug-10-21-2026/experiments/07_calibrated_conditional_sft_zipmix/debug_10-21-2026/DEBUG_REPORT.md>

**TLDR:** The null is real and expected, not a bug. Stage A's pre-registered rule (maximise fresh calibration-target accuracy subject to a retention floor) selects the smallest learning rate in the grid (2e-6), the one that moves the base model least (6.7% of predictions change). At that rate the three selectors change predictions from each other (5.1–5.6% of items) no more than the sampling seed does within one selector (5.3–5.6%), so the within-stratum alignment contrast comes out at +0.8 percentage points (pp), a third of the 2 pp advance bar. Three blocking bugs were found and fixed, all in the never-run local implementation, none in the code that produced the null. The pre-registered diagnostic at the next grid rate (5e-6) doubles the model movement and roughly doubles the contrast (+1.0 and +1.3 pp, 4/5 seeds positive each) but still does not reach the 2 pp advance bar (§7).

Date: 10-21-2026. Worker: Claude `claude-fable-5-1` (effort max) on SNAP skampere2, branch `codex-zipmix-e07-debug-10-21-2026`. Dispatch records: `dispatch/ACK.md`, `dispatch/CKPT_e07_debug.md`, `dispatch/results.md`.

## 1. What "Experiment 07" is, and which copy produced the null

Two implementations exist. They share the pool, selectors and training recipe but differ in evaluation sets and code.

| Copy | Location | State |
|---|---|---|
| **Original run (the null result)** | GitHub branch `calib/exp07-fresh-calibration` at `447e8a5`, folder `experiments/07_training_calibration` (git tree `923a3b26`), FarmShare Slurm jobs 1861877 (Stage A) and 1862429 (Stage B), one NVIDIA L40S, torch 2.13.0+cu130, transformers 4.57.6 | 20 + 15 training cells and 2 base evaluations, all complete, 0 failures, 0 resumes; `decision.json` (gate passed, lr 2e-6) and `analysis.json` (advance=false) committed with every per-item prediction file (140 `.npz`) |
| Local transferred source | `experiments/07_calibrated_conditional_sft_zipmix` (untracked on this branch; from the Mac, dated 10-16/10-17-2026) | Never run anywhere I could find (no ledger under `/dfs/scratch0/brando9` or `/lfs/skampere2/0/brando9`); could not run as shipped (§6) |

The original run's artifacts were read from a read-only `git archive` extraction outside the repo (`../orig_branch_ro/`, not committed) and were not modified. SHA-256 of the frozen originals:

```text
9eb9e63a33df38c508774a69a31d398bcb45baedc3d7557166e020602cf140b1  results/stage_a/ledger.json
886051a2a25e0bd4af61e73b69bc5fb69b4b9e5771045108292b2c3ff0ccd963  results/stage_a/decision.json
82859e94889962645f99c2db45bd261e7c55e4757e025b16a2d5f2890b7ecf7a  results/stage_b/ledger.json
72a85a37d09f53f85dbe3114aa1d047cb4fdde828e5f72db0b86fbd19eb42ef2  results/stage_b/analysis.json
456df929f93cda8f5ef63f4c0ff2bc49585200a801d96d680c48fd3f8b8755e0  results/exp07_calibration_results.tgz
a4dfed973d6d7a93022c58550cecd6c787e8ee2c4ecec8cac667a23d043fa552  results/exp07_stage_b_results.tgz
531718aec5829d3129dc537f4160d9cc6e35ccc0dbb6956c2c41deb72e60b63b  train_calibration.py
d7152909ee98a4ff792296508995e40c3afdd7a60df5b9c4a05fe9e966ec42b0  analyze_calibration.py
faaed08a9b62f55a6ad1767b969a934d03e7257899514352744e212fd2e49433  calib_common.py
```

The code hashes recorded inside both run ledgers equal the committed files (6/6), so the committed code is the code that ran. The Stage B training pool `train.npz` has the same hash in the original run manifest and in the local data (`e1aef754…`): both are Experiment 05 version 2's 9,797-question pool.

## 2. The original result, restated with denominators

Stage A (fresh calibration split, uniform sampling over 8,121 fresh records, 4 learning rates × 5 seeds). Base: `calibration_target` 400/500 = 80.0%, `calibration_retention` 260/500 = 52.0%.

| Learning rate | cal_target mean % (5 seeds) | sd pp | seeds ≥ base | cal_retention mean % | predictions changed vs base |
|---|---:|---:|---:|---:|---:|
| 2e-5 (Exp 05 recipe) | 73.52 | 2.27 | 0/5 | 55.24 | 23.2% |
| 1e-5 | 77.80 | 2.05 | 1/5 | 59.80 | 15.4% |
| 5e-6 | 79.88 | 0.91 | 2/5 | 60.28 | 11.8% |
| **2e-6 (selected)** | **80.72** | 0.56 | 4/5 | 57.24 | 6.7% |

Stage B at 2e-6 (3 selectors × 5 seeds on the Experiment 05 v2 pool). Base: `holdout_target` 394/500 = 78.8%, `holdout_retention` 273/500 = 54.6%.

| Selector | holdout_target mean % | by seed % | holdout_retention mean % |
|---|---:|---|---:|
| conditional_signed_rank (aligned) | 80.76 | 81.2, 81.2, 80.8, 81.0, 79.6 | 56.48 |
| uniform_stratum | 80.00 | 81.8, 80.0, 79.0, 79.2, 80.0 | 56.80 |
| permuted_stratum_rank | 79.96 | 79.2, 81.2, 79.6, 81.2, 78.6 | 57.24 |

| Contrast (aligned minus control), holdout_target | mean pp [95% Student-t over 5 paired seeds] | positive pairs | exact sign p | Holm p |
|---|---|---:|---:|---:|
| minus uniform_stratum | +0.76 [−0.70, +2.22] | 3/5 | 1.0 | 1.0 |
| minus permuted_stratum_rank | +0.80 [−0.32, +1.92] | 3/5 (1 tie) | 0.625 | 1.0 |

Advance rule (protocol): mean gain ≥ 2 pp and ≥ 4/5 positive pairs against both controls, retention change ≥ −1 pp. Not met → `advance=false`, "stop scaling this exact formulation". Recipe transfer check passed (uniform_stratum on `calibration_target` 80.08% vs base 80.0%). I recomputed every mean, interval and sign-test p-value in `decision.json` and `analysis.json` from the committed per-item predictions: all reproduce exactly.

## 3. Check (a): labels and data splits

Evidence from the original run:
- Leakage audit in both manifests: all pairwise role/role and role/prior hit counts are 0; fresh evaluation sets vs Stage B training pool: 0 exact and 0 13-gram hits for all four sets.
- Per-item files: `prediction == gold` agrees with the stored `correct` flag everywhere; gold labels are balanced (cal_target 122/127/121/130 over A–D); the base never predicts a label beyond each item's choice count (masking works).
- Both evaluation families are what the protocol says: SciQ and CommonsenseQA *training-split remainders* never used by Experiments 05/06. (The original protocol notes only ~491 unused official SciQ test questions remained, so the fresh holdout is a train remainder; the local never-run copy instead froze official validation/test remainders.)

Evidence from the local frozen data (`expt_v1/data`, manifest `c01070d7…`): 18/18 file hashes and the manifest id verify; label tokens decode to ` A`–` E`; `answer_token == label_tokens[answer]` and `0 ≤ answer < nchoices` for all 13,287 rows across the 8 arrays; zero identity or normalised-question overlap between any two of train, cal_target, cal_retention, target, retention, test_holdout2; padding beyond each length is the pad token.

**Verdict: no label or split bug.** (Tests: `test_every_split_has_consistent_labels_and_unique_items`, `test_splits_are_pairwise_disjoint_by_identity_and_question`.)

## 4. Check (b): learning rate and training length

- The selection rule is "highest mean cal_target among rates whose retention is within 1 pp of base; ties within 0.2 pp go to the larger rate". The retention floor never binds (base 52.0%; every rate ≥ 55.2%), and cal_target decreases monotonically with the rate (80.72 → 79.88 → 77.80 → 73.52), so the rule returns the smallest rate in the grid. The 0.2 pp tie band does not reach 5e-6 (gap 0.84 pp).
- The gate's "moves predictions ≥ 2%" check passes at 6.7%, but 6.7% is also the ceiling on how much any selector can change the result. On holdout_target the selectors disagree with each other on 5.1–5.6% of items at the same seed, while one selector disagrees with itself across seeds on 5.3–5.6%: the data-selection signal is no larger than the sampling-seed noise at this rate.
- Training length is the frozen Exp 05 recipe (128 updates × 16 one-token labels = 2,048 answer tokens; 1,793–1,867 unique questions, about 19% of the pool). Mean training loss is 0.78–0.87 at 2e-6 and 0.79–0.84 at 5e-6; the last-step values (0.4–1.8) are single-batch noise. The cosine schedule reaches zero at the final step in every Exp 05/06/07 trainer, so 127 of 128 updates are effective; this is identical across arms and immaterial to paired contrasts (documented in `test_cosine_schedule_matches_frozen_recipe`, not changed).
- Experiment 05's selector contrasts (+1.9 and +3.1 pp) were measured at 2e-5, where 23.2% of predictions change and every model loses 4–7 pp of target accuracy. Scaling the contrast by movement (6.7/23.2 ≈ 0.29) predicts +0.5 to +0.9 pp at 2e-6, which is what Stage B measured (+0.8 pp).

**Verdict: yes, the calibrated rate is too small for a 2 pp selector effect to appear, but by construction of the pre-registered rule, not by a coding error.** Any rate in the grid that moves the model more also degrades cal_target below the base (−0.12 pp at 5e-6, −2.2 pp at 1e-5, −6.5 pp at 2e-5), which is the tension the experiment was designed to expose.

## 5. Check (c): metric mismatches

- Training objective: cross-entropy of the gold label token at the final (left-padded) position. Evaluation: argmax over the valid label tokens at the same position, plus the label negative log-likelihood. Preparation verifies the prompt/label tokenization boundary. Same code path in Exp 05 v2, the original run and the local copy (`test_last_logits_equals_final_position_of_full_logits`, `test_left_padding_positions_start_at_zero_on_first_real_token`).
- Selector signal: pack-level maximum gzip alignment to eight SciQ *validation* views (Exp 05 v2) minus the wrong-target (CommonsenseQA) alignment, ranked inside 13 source × compression strata. Evaluation target: SciQ questions. Same family; no proxy/target mismatch beyond the known pack granularity (351 packs, 28 questions per pack on average, so items inherit their pack's weight).
- Stage A calibrated the rate on a *different* pool (8,121 fresh records) than Stage B trains on (the 9,797 Exp 05 pool). The protocol anticipated this with the transfer check, which passed (80.08% vs 80.0%).
- Analysis: paired seeds are the unit; exact two-sided sign test (minimum p 0.0625 at 5 seeds), Student-t intervals, Holm over the two target contrasts. The advance rule is a resource heuristic, not a significance test. `analysis.json` reproduces from the per-item files.

**Verdict: no metric mismatch found.**

## 6. Check (d): seeds or configuration shared across arms

- The design is paired on purpose: arm × seed cells share the pretrained initialisation (no dropout, no random init) and the sampler seed, and differ only through the frozen probability vector. The frozen vectors are distinct (total variation 0.169 aligned vs uniform, 0.225 aligned vs permuted; effective sample fractions 0.866 / 1.000 / 0.866; joint source × bin masses equal to 5.6e-17); the local `selection.npz` reproduces these vectors to 2.7e-20 (`test_frozen_conditional_selectors_reproduce_and_pass_gates`).
- The realised training sets differ across arms: unique questions drawn at seed 0 are 1,824 (aligned), 1,859 (uniform), 1,809 (permuted); mean training losses differ; same-seed predictions differ on 5.1–5.6% of holdout items. Because `numpy.Generator.choice` uses common random numbers, same-seed draws are correlated across arms, which lowers paired variance; it does not make arms identical.
- Learning rate, steps, batch, optimizer, precision and evaluation sets are shared across arms by design (that is the controlled comparison).

**Verdict: arms are not identical; no shared-seed collapse.**

## 7. Check (e): expected effect against noise, and the prospective diagnostic

Noise in the original Stage B (holdout_target, N=500, 5 paired seeds): per-selector sd 0.6–1.1 pp; paired 95% half-widths 1.1–1.5 pp; item-paired standard error 0.34–0.44 pp, giving an 80%-power minimum detectable contrast of about 1.0–1.2 pp (2.8 × SE). The design could therefore detect the 2 pp bar; it observed +0.8 pp. Pooled over seeds, the aligned selector wins 61 and loses 42 discordant items against uniform (binomial p=0.076) and 69 against 49 versus permuted (p=0.080). Aligned fixes 17.2% of the 106 base-wrong items and breaks 2.1% of the 394 base-right ones, versus 15.3–15.8% and 2.6–2.7% for the controls. Retention moves the other way (aligned − permuted −0.76 pp, 0/5 positive, sign p=0.125). Reading: a small, consistently signed target effect well under the bar, with 95% intervals that include zero; not proof of a zero effect and not proof of a bug.

**Pre-registered diagnostic** (recorded in `dispatch/CKPT_e07_debug.md` before any number was read): the same three selectors × seeds 0–4 at the next grid rate 5e-6, chosen by a fixed rule (the largest Stage A grid rate whose mean cal_target lies within the protocol's 0.2 pp tie band of the base: 79.88% vs 80.0%), on the local frozen pool, evaluated only on the local `cal_target` and `cal_retention` arrays (500 each; the fresh official-split holdouts `target`, `retention`, `test_holdout2` were never loaded). Prediction before measuring: +1.0 to +1.6 pp if the contrast scales with movement (11.8% at 5e-6), i.e. still below 2 pp. Interpretation rules: both contrasts ≥ +2 pp with ≥ 4/5 positive → effect expressible at a non-degrading rate, so the null follows from the calibration rule; contrasts < +1 pp or mixed sign → null substantive at every non-degrading rate; otherwise inconclusive. Exploratory either way; no paper claim changes.

### 7.1 Diagnostic result (measured 10-21-2026 15:39–15:47Z, run id `a4add617336742bd…`)

Receipts: [`diagnostic_lr5e-6/`](diagnostic_lr5e-6/) (ledger, every `result.json`, every per-item prediction file, `train.log`, `launch.sh`, `SHA256SUMS`); analysis by [`analyze_diagnostic.py`](analyze_diagnostic.py) → [`analysis.json`](diagnostic_lr5e-6/analysis.json), [`analysis.md`](diagnostic_lr5e-6/analysis.md). 15/15 cells and the base evaluation complete, 0 failures, 0 resumes. Compute: one shared H200 (skampere2 GPU 1, `CUDA_VISIBLE_DEVICES=1`, torch 2.5.1+cu124, transformers 4.51.1), wall 471 s from launch to exit, 377 s of summed training time, 13.5 GB resident; under the 30 GPU-minute cap. Every cell drew exactly the same unique-example counts as the corresponding original Stage B cell (e.g. seed 0: 1,824 / 1,859 / 1,809), so the local pool, selectors and sampler reproduce the original run's draws.

Primary split `cal_target` (N=500, base 399/500 = 79.80%) and secondary `cal_retention` (N=500, base 274/500 = 54.80%); intervals are Student-t over 5 paired seeds, sign tests exact two-sided, Holm across the two target contrasts; pooled discordant-item counts are descriptive.

## cal_target (N=500; base 399/500 = 79.80%)

| method | seeds | mean % | sd pp | by seed % | flip vs base | fixes base-wrong % | breaks base-right % |
|---|---:|---:|---:|---|---:|---:|---:|
| conditional_aligned | 5 | 81.16 | 0.85 | 80.4, 82.0, 82.0, 81.2, 80.2 | 11.9% | 27.7 | 5.3 |
| stratum_uniform | 5 | 80.12 | 1.41 | 79.2, 81.2, 80.4, 78.2, 81.6 | 11.0% | 23.8 | 5.6 |
| conditional_shuffled | 5 | 79.88 | 2.22 | 78.2, 77.8, 80.2, 83.4, 79.8 | 11.4% | 23.0 | 5.7 |

| contrast (aligned minus control) | paired seeds | mean pp [95% t-interval] | positive / ties | exact sign p | Holm p (2 target contrasts) | item-paired SE pp | discordant items win/loss (pooled seeds) | binomial p |
|---|---:|---|---|---:|---:|---:|---|---:|
| conditional_aligned − stratum_uniform | 5 | +1.04 [-0.94, +3.02] | 4 / 0 | 0.3750 | 0.7500 | 0.697 | 132/106 | 0.105 |
| conditional_aligned − conditional_shuffled | 5 | +1.28 [-1.67, +4.23] | 4 / 0 | 0.3750 | 0.7500 | 0.643 | 126/94 | 0.036 |

## cal_retention (N=500; base 274/500 = 54.80%)

| method | seeds | mean % | sd pp | by seed % | flip vs base | fixes base-wrong % | breaks base-right % |
|---|---:|---:|---:|---|---:|---:|---:|
| conditional_aligned | 5 | 57.20 | 1.36 | 56.0, 55.6, 58.4, 58.6, 57.4 | 24.3% | 20.6 | 12.6 |
| stratum_uniform | 5 | 57.20 | 1.24 | 57.4, 58.0, 58.4, 57.0, 55.2 | 25.5% | 21.6 | 13.4 |
| conditional_shuffled | 5 | 55.88 | 0.76 | 55.4, 54.8, 56.4, 56.6, 56.2 | 25.7% | 20.4 | 14.9 |

| contrast (aligned minus control) | paired seeds | mean pp [95% t-interval] | positive / ties | exact sign p | Holm p (2 target contrasts) | item-paired SE pp | discordant items win/loss (pooled seeds) | binomial p |
|---|---:|---|---|---:|---:|---:|---|---:|
| conditional_aligned − stratum_uniform | 5 | +0.00 [-2.41, +2.41] | 2 / 1 | 1.0000 | 1.0000 | 0.776 | 203/203 | 1.000 |
| conditional_aligned − conditional_shuffled | 5 | +1.32 [+0.50, +2.14] | 5 / 0 | 0.0625 | 0.1250 | 0.792 | 222/189 | 0.114 |



### 7.2 Reading against the pre-registered rules

- Measured target contrasts: aligned − uniform **+1.04 pp** [-0.94, +3.02], 4/5 positive, sign p=0.375 (Holm 0.750); aligned − shuffled **+1.28 pp** [-1.67, +4.23], 4/5 positive, sign p=0.375 (Holm 0.750). Both land inside the expectation recorded before launch (+1.0 to +1.6 pp) and both fall short of the 2 pp advance bar, so by the recorded rules the outcome is **"otherwise: inconclusive"**, not the "expressible at a non-degrading rate" branch and not the "substantive null" branch.
- Movement scaling is supported: predictions changed vs base rose from 6.5–6.9% (original, 2e-6) to 11.0–11.9% (5e-6), and the aligned-minus-control contrast rose from +0.76/+0.80 pp to +1.04/+1.28 pp, roughly in proportion. The pooled discordant-item counts lean the same way (132/106 and 126/94; binomial p=0.105 and 0.036, descriptive). With 5 seeds the smallest attainable sign-test p is 0.0625, and all 95% intervals on the target contrasts include zero.
- At 5e-6 on this draw none of the three selectors fell below the base on cal_target (+1.36, +0.32, +0.08 pp for aligned, uniform, shuffled), whereas the original Stage A sample showed −0.12 pp for uniform at this rate: the "non-degrading" boundary of the grid is itself within sampling noise of 5e-6.
- Retention: aligned equals uniform (0.00 pp) and beats shuffled by +1.32 pp [+0.50, +2.14] (5/5, sign p=0.0625, unadjusted secondary readout); no retention cost of the aligned selector is visible here.

**Conclusion of check (e):** the original null is what a ~1 pp-per-10%-movement effect looks like under a rule that permits only 6.7% movement and a 2 pp bar; the diagnostic at the next rung doubles the movement and the contrast but still does not reach the bar. The alignment effect appears small and consistently signed, not zero, and not large enough to clear the pre-registered advance criterion at any rate in the grid that keeps the base accuracy. This is exploratory evidence on one evaluation draw and must not be used to pick a rate for a confirmatory claim or to change the paper's statement of the null.

## 8. Bugs found and fixed (branch `codex-zipmix-e07-debug-10-21-2026` only)

All three are in the local, never-run implementation `experiments/07_calibrated_conditional_sft_zipmix/expt_v1/train_exp07.py`. None exists in the original run's `train_calibration.py`, and none can have affected the null result.

| # | Bug | Effect | Fix | Regression test |
|---|---|---|---|---|
| 1 | Startup hashed `analyze_exp07.py`, a file that was never shipped (README also lists it and `test_exp07.py`) | `FileNotFoundError` before any training in every mode | `code_identity()` records absent files as `missing` | `test_code_identity_records_missing_files_instead_of_crashing` |
| 2 | Stage 2 modes read `manifest["method_status"][method]`, a key the frozen manifest never contains | `KeyError` in `conditional` and `selectors` modes | `plan_cells()` defaults to `available` (every frozen vector is valid) | `test_plan_cells_without_method_status_table_marks_every_cell_pending` |
| 3 | Every array in `selection.npz` was validated as a probability vector, including the metadata arrays `domain`, `bucket`, `pack_index` | `ValueError: Invalid frozen weights: domain` in every mode (observed on the first diagnostic launch) | `frozen_weights()` skips the metadata keys | `test_frozen_weights_skip_metadata_arrays_and_validate_vectors`, `test_frozen_selection_file_loads_as_nine_valid_selectors` |

Also added: a `--mode diagnostic` with explicit `--learning-rate`, `--methods`, `--seeds`, `--eval-splits` (defaults to the calibration splits so fresh holdouts stay unread; the run identity includes these arguments), an unknown-selector check before training, and `expt_v1/test_exp07.py` (13 tests, CPU only, no weights or network):

```text
python -m pytest -q -p no:langsmith expt_v1/test_exp07.py   →   13 passed in 5.79s
```

Not changed: frozen data, manifests, selectors, `common.py`, training hyper-parameters, the original branch, `main`, and all paper sources. `analyze_exp07.py` for the full local protocol remains unwritten; the diagnostic analysis lives in `analyze_diagnostic.py` here.

## 9. Limitations

- The diagnostic evaluates on calibration-type sets (SciQ / CommonsenseQA train remainders, N=500 each, one draw), with 5 seeds on one 0.5B backbone; the smallest two-sided sign-test p-value is 0.0625 and intervals cover sampling seeds only, not new questions or corpora.
- It changes one setting (the learning rate) relative to the frozen protocol and is therefore exploratory; it must not be used to pick a rate for a confirmatory claim, and no claim in the paper was changed.
- Compute receipts: per cell 25–30 s of training plus 2–5 s of evaluation on a shared H200 (GPU 1, about 13.5 GB resident); a 17 s checkpoint write at step 64 is in the elapsed figures.

**TLDR-end:** [zip-mix: exp07 null debug] The Exp 07 null is a +0.8 pp within-stratum alignment contrast measured at the calibrated rate 2e-6, where selectors move predictions no more than seeds do; it follows from the pre-registered rule, and no bug was found in the code that produced it. Three blocking bugs in the separate never-run local trainer were fixed with tests; the 5e-6 diagnostic (15 cells, 8 GPU-minutes) shows the contrast scaling with movement to +1.0/+1.3 pp, still under the 2 pp bar, so the null stands as reported.
