# Experiments 04 and 05: requested Opus 5.5 maximum-effort review

**Doc link:** <https://github.com/brando90/zip-mix/blob/main/experiments/04_validation_guided_zipmix/qa/opus55_max_review.md>

**TLDR:** This is the single review the user asked for. It found and fixed one major recovery bug in the fine-tuning trainer and four robustness gaps, and added five tests. All 50 deterministic tests pass. No critical issue remains. One major design issue remains in Experiment 05: compression bins and maximum alignment mostly track string length, so the Zip-Mix-versus-shuffled contrast cannot separate validation alignment from length. That needs a prospective decision from the parent. No training was launched and no result is claimed.

Reviewed 10-04-2026. Scope: base commit `c672ee7` to the current working tree, limited to `AGENTS.md`, `CLAUDE.md`, `README.md`, `experiments/README.md`, `experiments/04_validation_guided_zipmix/` and `experiments/05_validation_guided_sft/`, including untracked files. The review touched no paper edits, no Experiment 02/03 material, no staged deletions or renames, no global configuration and no Git state.

## Reviewer identity

- Model: `claude-opus-5-5` (Claude Opus 5.5). This comes from the runtime system context of this Claude Code session.
- Effort: the dispatcher requested maximum effort. The reviewer cannot observe the command-line effort flag from inside the session, so this is the requested setting, not an independently verified receipt.
- No other reviewer model, model-provider application programming interface (API) call, training run, public message or download of model weights was used.

## Fixes applied

| # | Severity | File | Defect | Fix and evidence |
|---|---|---|---|---|
| 1 | Major (bug) | `05_validation_guided_sft/expt_v1/train_sft.py:173` | Suppose a cell is interrupted after its final-step (step 128) checkpoint, for example during evaluation. On resume the training loop runs zero iterations, so `del model, optimizer, logits, loss` raised `UnboundLocalError`. That happened *after* the ledger row was marked complete, so `run()` overwrote the verified complete row as `failed`. | `logits = loss = None` before the loop. New test `test_interrupted_cell_resumes_bitwise_and_final_step_resume_completes` checks that an interrupted-then-resumed cell is bit-identical to an uninterrupted one, and that a final-step resume finishes `complete` with 0 replayed steps. A mutation check confirmed this test fails with `UnboundLocalError` without the fix. |
| 2 | Major (protocol guard) | `04_validation_guided_zipmix/expt_v1/train.py:307,629,643` | `--retry-failed` re-ran *any* failed cell, including nonfinite-loss and nonfinite-gradient failures. That contradicts the runbook ("do not retry numerical failures to hunt for wins"). | Each failure is now classified as `numerical`, `interrupted` or `runtime` and stored as `failure_kind` in the manifest and the failure receipt. Numerical failures are never retried. Tests: `test_failure_kind_separates_numerical_from_infrastructure` and `test_retry_skips_numerical_failures_but_resumes_runtime_failures`; the second also exercises a real checkpoint resume through the command line. |
| 3 | Moderate (input integrity) | `04_validation_guided_zipmix/expt_v1/train.py:594` | The trainer fingerprinted the data but never compared it with the frozen preparation manifest, so a corrupted or mismatched transfer would only be caught by hand. | Training now refuses to start if `train.npz`, `dev.npz` or `eval.npz` differ from `data/manifest.json` `files`. The real frozen hashes match (all three `True`). Test: `test_data_must_match_frozen_preparation_manifest`. |
| 4 | Moderate (durability) | `05_validation_guided_sft/expt_v1/train_sft.py:42` | Checkpoints were renamed into place without `fsync`, so a host crash could leave a checkpoint that cannot be loaded. | The checkpoint is now flushed and fsynced before the atomic rename, matching Experiment 04. |
| 5 | Moderate (completion integrity) | `05_validation_guided_sft/expt_v1/train_sft.py:341` | The trainer exited 0 even with failed cells or a failed base evaluation. An external monitor that checks the exit code plus training-cell counts could therefore report a run as complete when its base evaluation had failed. | The trainer now exits nonzero unless all 18 cells and the base evaluation are complete, the same contract as Experiment 04's `train.py`. |
| 6 | Documentation | `05_validation_guided_sft/README.md`, `results.md`; `04_validation_guided_zipmix/README.md`, `CKPT_zipmix.md` | The length confound below was undocumented, and the review status was stale. | Added the measured confound and its interpretation limit, and linked this report. |

Also added `test_left_padded_last_logits_match_unpadded_qwen2_architecture`. It uses a tiny randomly initialized Qwen2 model, so it goes through the same Transformers masking, position and `logits_to_keep` code as the real run, without weights or network access. A left-padded batch matches individually scored prompts within 1e-5. The pinned Qwen2.5-0.5B weights are not cached on the review machine (only its tokenizer is), so this checks the code path, not the pretrained model.

Editing `train.py` and `train_sft.py` changes their run fingerprints. That is expected because no scientific run has started. The engineering-timing receipt keeps its old fingerprint and stays engineering-only. `common.py` and `prepare_sft.py` were deliberately left unchanged so the frozen manifest's `preparation_code` hashes still match.

## Verified without change

- **Fine-tuning precision (specifically requested).** Weights load with `torch_dtype=torch.float32` (`train_sft.py:116`), and the code asserts that every parameter is float32 and trainable. AdamW therefore keeps float32 moment buffers. Only the forward pass runs under CUDA bfloat16 autocast (`:73`). The loss uses float32 logits (`:77`) and gradients are float32. At learning rate `2e-5`, the float32 master update avoids bfloat16 rounding swamping the step; no low-precision weight update is performed. The existing test `test_model_loading_uses_float32_master_parameters` covers the load contract.
- **Labels, causal shifts and last-token scoring.** Experiment 04 predicts `tokens[1:]` from `tokens[:-1]` with causal attention; tests check the shift and that future tokens cannot change earlier logits. Experiment 05 uses left padding, attention masks and cumulative position ids (`:61`) and scores position −1. One-position cross-entropy equals masked causal loss (tested), and invalid letter options are masked. Answer continuations are asserted to be single tokens at the prompt boundary.
- **Split and leakage boundaries.** Experiment 04: exact normalized deduplication, then a hash-ordered split per domain (32 development / 64 evaluation / rest training), then removal of 623 training documents that share any 13-word sequence with held-out text. The tokenizer is fit on training text only, alignment uses development documents only, and evaluation data is read only after final training. Experiment 05: full official held-out splits are protected by exact-question and 13-word content rules, with 0 residual hits recorded. Cross-role duplicates are removed in the order alignment, target, retention. Selection never reads answer labels.
- **Maximum versus mean ZIP-FIT.** `zipmix_static` uses the historical maximum score in both experiments. `direct_zipfit` uses the published mean, and refuses to run without `mean_score` in Experiment 04. In Experiment 04 direct ZIP-FIT is soft sampling; in Experiment 05 it is top-25% selection. Each is labeled as an adaptation.
- **Exact weight normalization.** Bucket mass is the sum of clipped scores, plus `tau` in Experiment 04 for nonempty buckets only, then uniform within each bucket. Probabilities sum to 1, and the empty-bucket formula is tested exactly. The recomputed Experiment 04 total variation (TV) from token-proportional is 0.0207, matching the manifest.
- **DoReMi.** The token-proportional reference is trained and charged separately. Per-token excess loss is clipped *before* averaging over each domain (`train.py:399`). The update is exponentiated, normalized and smoothed (`c=1e-3`). The proxy gradient uses the unclipped weighted negative log-likelihood (NLL), with balanced per-domain batches and time-averaged weights. Reference, proxy and final models are distinct, the final model starts from a fresh initialization, and costs are kept per stage.
- **DoGE.** Alignment is the inner product of the full-parameter development gradient with each domain's gradient. Each vector is norm-clipped, the score is divided by the mean training-gradient norm, and the step is the current learning rate divided by `mu`. The pooled held-out target and same-size proxy are declared adaptations. That the normalization matches the official trainer was **not** checked against upstream code in this review.
- **Budgets and seeds.** Every final model gets 2,048 × 32 × 128 = 8,388,608 tokens, and the analysis rejects any mismatch. Experiment 05 gives each cell 128 × 16 = 2,048 answer tokens. Each final stage reseeds initialization and sampling from the seed, so arms with the same seed share initialization and random draws. Reference, proxy and probe tokens are counted separately; the cost proxy states what it excludes.
- **Statistics and denominators.** Both analyses use paired seeds as the independent unit, Student-t intervals only when n ≥ 3, and exact sign tests (minimum p = 0.25 at three pairs, stated). They never impute values, keep pending, failed and missing rows in the denominator, and re-audit every aggregate against saved per-example arrays.
- **Locks and ownership.** Each output directory has an exclusive `flock`. The Experiment 04 monitor checks process-group liveness through `/proc`, enforces the deadline with SIGTERM followed by SIGKILL, refuses to reuse a runtime directory, and requires fresh progress plus exact counts to report completion. A SIGTERM inside `train.py` marks the cell interrupted.

## Remaining issues and limits

**Major, needs a parent decision (not fixed, because frozen data was preserved):**

1. **Experiment 05 cannot separate validation alignment from length.** On the frozen pool of 10,116 candidates:
   - Spearman correlation of LZ4 ratio with byte length is −0.92, and of maximum alignment with byte length is −0.88.
   - 87.4% of candidates have an LZ4 ratio above 1, because framing overhead exceeds compression savings on short strings.
   - Median view bytes fall from 210 in bin 0 to 91 in bin 4, while mean maximum score rises from 0.369 to 0.495.
   - Zipmix_static differs from sample-proportional by TV 0.0431, but the source-by-length permutation control reproduces almost all of that (TV 0.0426 from sample-proportional). Zipmix_static and its shuffled control differ by only TV 0.0025.

   The prespecified Zip-Mix-versus-shuffled contrast is therefore close to null by construction. Any Zip-Mix gain over sample-proportional would mainly reflect a preference for short questions. These are exact descriptive properties of the frozen pool, not performance estimates (p-val=n/a). The parent can run the frozen screen as a length-confounded mechanism screen. Testing the user's alignment hypothesis needs a separately versioned condition, for example equal-length or length-normalized compression scoring, compression bins that are not dominated by framing overhead, or a length-matched mismatched-target control.

**Scope limits (already documented, not counted as defects):** Experiment 05 includes no DoReMi, DoGE, LESS (low-rank gradient similarity search), DSIR (data selection via importance resampling) or newer baseline. Experiment 04's DoReMi and DoGE are compact same-size adaptations, and DoReMi is target-agnostic by design. The Experiment 04 Zip-Mix prior moves only 2.07% of mass (TV 0.0207) while the shuffled controls move 0.04–0.08%, so a small effect may be undetectable with three seeds. Total compute is not matched.

**Minor (not fixed):**

- Experiment 04 `analyze.py` reports NLL but no perplexity column, and does not compute the protocol's agent-suggested ≥0.02-nat screening criterion; both can be derived from the reported values.
- `uniform_bucket` gives about 84 top-bucket blocks 1/7 of the mass (3,942 effective blocks), and Experiment 05's Compel arm has 150 examples (75% ARC), so repetition confounds both controls.
- A recovery attempt under the monitor needs the original `--run-id`; otherwise a correct completion is reported as unverified.
- Experiment 05 `common.atomic_json` has no fsync; it was left unchanged to keep the frozen preparation-code hash.
- The final cosine step in Experiment 05 has learning rate 0. This is the same for every arm.
- "One interrupted cell may resume" in the protocol is implemented as one recovery per cell in both experiments.

## Checks run

```text
Experiment 04: python -m pytest -q test_train.py                -> 23 passed (20 existing + 3 new)
Experiment 05: python -m pytest -q test_sft.py test_analysis.py -> 27 passed (25 existing + 2 new)
Mutation check: Experiment 05 resume test without fix 1         -> UnboundLocalError (test detects the bug)
Frozen Experiment 04 data vs manifest files                     -> train/dev/eval sha256 match: True/True/True
py_compile on all owned Python sources                          -> OK
git diff --check on owned experiment paths                      -> clean
AGENTS.md vs CLAUDE.md bodies after the doc-link header         -> identical
```

Local test runtime: Python 3.11, torch 2.7.0, NumPy 2.2.6, SciPy 1.16.3, Transformers 4.57.2. These tests use only synthetic arrays and tiny random models; none of their values is a research result.

VERDICT: FIXED
CRITICAL_ISSUES: 0
MAJOR_ISSUES: 1
FIXES_APPLIED: 6
STRUCTURAL: IMPROVED
SUMMARY: Fixed a fine-tuning resume bug that turned recovered complete cells into failures, blocked retries of numerical failures, and added frozen-data, durability and exit-code guards with tests (50/50 pass). Experiment 04 is ready to launch as a declared compact screen. Experiment 05's compression scores are mostly string length, so its Zip-Mix-versus-shuffled contrast cannot test validation alignment without a prospective redesign.
