# Runbook prompt: Experiment 03, does Compel-ZipMix fix DoReMi?

Paste this whole file as the prompt for the executing agent (Antigravity `agy`, on SNAP).

---

Brando pre-approves all task work here (zip-mix project). Run with full access and never stop to ask permission. Budgets, secrets and shared-node safety still apply.

## Read first

1. `~/zip-mix/experiments/03_zipmix_fixes_doremi_proxy_mixtures/README.md`: the protocol. It is binding: arms, metrics, statistics and thresholds. Do not change them silently. Record any deviation in `CKPT_exp03.md` with the reason.
2. `~/zip-mix/experiments/02_alignment_prior_analysis/README.md` and its latest `results/results_summary_*.md`: the gate and the priors.
3. `~/agents-config/INDEX_RULES.md` Hard Rule 9 (no LLM provider API calls), Hard Rule 11 (statistics), Trigger Rule 61 (uninterrupted full-set evaluation).

## Phase 0: gate and setup (stop if the gate fails)

- If Exp 02's decision is not "proceed" (alpha^(0) concentrated on mid-CR buckets) and no redesign is documented, write the reason to `CKPT_exp03.md` and stop. Do not train.
- Pick free GPUs with `nvidia-smi` on `skampere1`/`skampere2`; never use a GPU another user has memory on.
- Set up Marin (Stanford) or, if Marin cannot run a ~150M model within a day of effort, the DoReMi reference code `sangmichaelxie/doremi`. Record which and why.
- Copy DoReMi's DRO step size, smoothing and reference-model recipe from the paper or the reference code into `config/doremi.yaml`, with a citation comment for each value.
- Tokenize `monology/pile-uncopyrighted` once; store the Pile domain (`pile_set_name`) and the LZ4 CR bucket (P7 boundaries) per document. Hold out a per-domain validation split.

Checkpoint: commit `config/` and `scripts/`; update `CKPT_exp03.md`.

## Phase 1: mixture weights

- A1 TP, A2 uniform, A8 Compel band: compute directly.
- alpha^(0) over buckets (A6, A7) and over domains (A4): take them from Exp 02's priors file if it covers the full Pile split; otherwise recompute with Exp 02's scripts at full scale.
- A3–A6: run the DoReMi procedure (reference model, then the DRO proxy) for each cell of the 2×2. Save the averaged final weights.

Checkpoint: `results/weights.json` with all 8 weight vectors, plus the highest-CR-bucket mass for each. Commit.

## Phase 2: one-run timing pilot

Train one ~150M arm (A1, seed 0) for 5% of the token budget. Extrapolate wall-clock and confirm the 24-run Stage A manifest fits the free GPUs. Write the estimate to `CKPT_exp03.md`.

## Phase 3: Stage A, the full manifest

Train 8 arms × 3 seeds at ~150M on ~3B tokens. Use durable execution (tmux/nohup with logs on disk) and nested timeouts. Never stop healthy runs at a coordinator boundary. A failed run is retried once, then kept in the manifest as failed.

Evaluate every finished run: per-domain held-out log-perplexity; lm-eval-harness on HellaSwag, ARC-E, ARC-C, PIQA, LAMBADA (held out from V); MMLU test and GSM8K test labeled "aligned-to-prior"; 13-gram contamination check.

Checkpoint after every 4 finished runs: append per-run rows to `results/runs.csv`; update `CKPT_exp03.md`; commit.

## Phase 4: analysis and report

For each pre-declared comparison (H0, H_prior, H_part, H_fix, H_static) report `difference [95% CI] p-val=Z (paired permutation test)`. Resample domains × seeds for perplexity and items (stratified by task) × seeds for accuracy. Give across-seed SD and n. Read verdicts only against the README's threshold, and label it agent-suggested.

Write `results/results_summary_<MM-DD-YYYY__HH-MM-SS>.md`: TL;DR first, config table, the 8-arm results table, the 2×2 table, weight plots, the contamination table, and a verdict for each hypothesis. Keep missing cells as "not measured / failed", never invented.

## Phase 5: Stage B (only if H_fix or H_static is supported)

Retrain A1, A3 and the best of {A6, A7} at ~1.4B with Marin, using the Stage A weights unchanged. Evaluate on the Compel benchmark suite and VeriBench. Report as in Phase 4.

## Rules

- No direct LLM-provider API calls. Model work goes through `agy`, the Antigravity CLI.
- Never stall on your own limits. If you hit one, write a checkpoint and hand off.
- Dates are MM-DD-YYYY. Commit with explicit pathspecs; never commit secrets or data shards.
- Final message: copy-ready one-line status, then `TLDR-end` and `Snapshot`.
