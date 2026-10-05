# Experiment 03: Does Compel-ZipMix fix DoReMi? Proxy-scale mixture comparison

**Status:** Designed, not run. Blocked on the Experiment 02 go/no-go gate (below).
**Started:** 10-04-2026
**Owner:** Brando Miranda (brando9@stanford.edu)
**Execution runbook:** [`prompt.md`](prompt.md)

## TL;DR

The project's Goal 1 is to "fix DoReMi's issues" (research journal slide 1).
The concrete, falsifiable version of that goal:

> Can a mixture chosen with a *validation-aligned compression-bucket prior* (Compel-ZipMix) beat **token-proportional sampling**, the baseline DoReMi reportedly loses to, at matched compute?

The design is a 2×2 that separates DoReMi's two suspected failure causes, plus the baselines David Hall (dlwh) called crucial.

## Where every design choice comes from

| Choice | Source |
|---|---|
| Goal = fix DoReMi | Research journal slide: "Goal 1: fix DoReMi's issues" |
| Token-proportional sampling is the crucial baseline | dlwh quote in journal: "It's worse than token proportional sampling (crucial baseline)" |
| Two failure hypotheses: uniform prior (H_prior) and over-pessimistic min-max (H_max) | Root `README.md` and paper `02_method.tex` |
| CR buckets, alpha^(0) formula, tau = 1e-3, P7 boundaries | `experiments/01_compression_threshold_buckets/README.md` |
| Validation suite V_agi, NCD with gzip, LZ4 for CR | `experiments/02_alignment_prior_analysis/README.md` |
| Proxy ~150M–280M, target ×10 (~1.4B), Marin framework | Exp 01 "Datasets and Models"; journal "Stanford's Marin – 1.4B model" |
| VeriBench is the most important downstream eval | Journal Experiments 1(e): "veribench (most important on)" |
| Exp 02 must pass before any GPU training | Exp 02 "Decision Criteria" and "Successor: Experiment 03" |

Anything not in this table that sets a number is marked **(agent-suggested, Brando to confirm)**.

## Gate: Experiment 02 must say "proceed"

Do not start GPU training unless Exp 02's decision is "Strong validation" (alpha^(0) concentrates on mid-CR buckets) or a documented redesign.
If alpha^(0) is nearly uniform, ZipMix-Static is close to uniform sampling and this experiment cannot distinguish H_prior.
A pilot of Exp 02 is being run on 10-04-2026; its result is linked here once available.

## Hypotheses

Let `TP` = token-proportional sampling (each source sampled in proportion to its token count).

- **H0 (replicate the failure).** Standard DoReMi on Pile domains does **not** beat TP on the primary metric. If DoReMi beats TP in our setup, there is nothing to fix, and the paper's framing must change.
- **H_prior (informative prior fixes it).** DRO initialized from alpha^(0) beats DRO initialized uniformly, holding the partition fixed.
- **H_part (compression buckets fix it).** DRO over CR buckets beats DRO over hand-defined domains, holding the initialization fixed.
- **H_fix (headline).** ZipMix-DRO (buckets + alpha^(0) init) beats TP on the primary metric.
- **H_static.** ZipMix-Static (sample directly from alpha^(0), no DRO) is within the noise of ZipMix-DRO; if true, the DRO step is unnecessary, which is a cheaper and still publishable result.

## Arms (all trained at the same model size, tokens and seeds)

| # | Arm | Partition | Weights |
|---|---|---|---|
| A1 | TP (crucial baseline) | Pile domains | token-proportional |
| A2 | Uniform | Pile domains | uniform over domains |
| A3 | DoReMi (standard) | Pile domains | DRO, uniform init |
| A4 | DoReMi + prior | Pile domains | DRO, alpha^(0) init (domain-level alignment prior) |
| A5 | Bucket-DRO, uniform init | CR buckets (P7) | DRO, uniform init |
| A6 | ZipMix-DRO | CR buckets (P7) | DRO, alpha^(0) init |
| A7 | ZipMix-Static | CR buckets (P7) | alpha^(0), no DRO |
| A8 | Compel hard filter | CR in [0.65, 0.80] | token-proportional within band |

A3/A4/A5/A6 form the 2×2 {domains, buckets} × {uniform init, alpha^(0) init}.
Comparisons: H0 = A3 vs A1; H_prior = A4 vs A3 and A6 vs A5; H_part = A5 vs A3 and A6 vs A4; H_fix = A6 vs A1; H_static = A7 vs A6.

## Experimental setting

- **Corpus:** `monology/pile-uncopyrighted` (DoReMi's original setting is The Pile; this mirror drops copyrighted subsets such as Books3, so the domain set is smaller than DoReMi's 22, report the exact list).
FineWeb-EDU is a second corpus in Stage B only, to connect to Compel.
- **Validation suite for alpha^(0):** V_agi from Exp 02 (MMLU validation split, 200 GSM8K train items with seed 42, MiniF2F informal statements). These items are excluded from all evaluation.
- **Proxy (DRO reference + DRO run):** follows DoReMi's recipe: train a reference model on the initial weights, then the proxy with Group-DRO weight updates; final weights are the average over DRO steps.
Size ~150M **(agent-suggested within Exp 01's 150M–280M range)**.
DRO step size and smoothing: use the DoReMi paper defaults; copy the exact values from the paper or the reference implementation `sangmichaelxie/doremi` into `config/` and cite the line, do not invent them.
- **Stage A (all 8 arms):** ~150M model, fixed token budget ~20 tokens/parameter (~3B tokens) **(agent-suggested)**, 3 training seeds per arm **(agent-suggested)**.
- **Stage B (transfer):** A1, best of {A6, A7}, and A3 retrained at ~1.4B with Marin (journal), same weights as Stage A (no re-tuning).
- **Compute:** SNAP `skampere1` (4× A100 80GB) / `skampere2` (H200); verify free GPUs before launch. Rough Stage A cost ~5 A100-hours per run × 24 runs ≈ 120 A100-hours **(estimate, verify with a 1-run timing pilot)**.

## Metrics

**Primary (Stage A):** mean held-out log-perplexity over Pile domains (DoReMi's metric), plus worst-domain log-perplexity.
Perplexity is primary at 150M because downstream accuracy on MMLU/GSM8K is near chance at that size.

**Secondary (Stage A):** downstream accuracy on tasks **not** in V: HellaSwag, ARC-Easy, ARC-Challenge, PIQA, LAMBADA (lm-eval-harness).
Report the in-V tasks (MMLU test, GSM8K test) separately and label them "aligned-to-prior", because the prior saw their validation splits.

**Stage B:** the Compel benchmark suite (copy the exact list from the Compel paper) and VeriBench.
VeriBench is near zero for models without Lean data, so a meaningful VeriBench comparison needs the Lean-augmented corpus from journal Experiment 1; record it, but do not read a verdict from it at Stage B unless scores are off the floor.

**Mixture diagnostics:** final weight vector per arm; mass on the highest-CR bucket (noise proxy); entropy of the weights (descriptive only, not a quality metric).

**Contamination check:** 13-gram overlap between sampled training data and every eval set, reported per arm.

## Statistics (Hard Rule 11)

- Every headline difference is reported as `x [lo, hi] p-val=Z (test)`.
- Perplexity differences: paired over (domain × seed); 95% CI by bootstrap over domains and seeds; p-value from a paired permutation test against zero difference.
- Downstream accuracy: paired bootstrap over evaluation items, stratified by task, with seeds as an outer level; p-value from a paired permutation test against zero difference.
- Report across-seed SD and n for every arm. Three seeds give weak power; say so when an interval includes zero.
- **Decision threshold (agent-suggested, Brando to confirm):** H_fix is supported if A6 − A1 mean log-perplexity is < 0 with the 95% CI excluding 0, **and** the held-out downstream macro-accuracy difference is ≥ 0 (no regression beyond its CI).

## Results

Not run. Results go to `results/results_summary_<MM-DD-YYYY__HH-MM-SS>.md` with TL;DR, config table and results table.

## Known limitations

- 150M models can only show perplexity-level effects; the downstream "fix" claim needs Stage B.
- The alignment prior uses validation splits of MMLU/GSM8K, so in-V gains are partly by construction; the held-out task block is what supports a general claim.
- Pile-uncopyrighted is not DoReMi's exact domain set; failure or success of A3 may not match the original paper.
- CR is a byte-level proxy and may behave differently on code and math-heavy domains.

## Verification checklist

- [ ] Exp 02 gate decision recorded and linked.
- [ ] DoReMi hyperparameters copied from the paper or reference code with citations.
- [ ] One-run timing pilot done before launching the 24-run manifest.
- [ ] Full manifest (8 arms × 3 seeds) completed; failed or missing cells kept in the denominator (Trigger Rule 61).
- [ ] V items excluded from eval; 13-gram contamination report produced.
- [ ] All headline numbers follow Hard Rule 11.

## Reproduction

Commands are added by the executing agent as scripts land in `scripts/`.

## Experimental details

- Dates, hardware, framework version, exact model configs and seeds: recorded by the executing agent at run time. Nothing here is filled in before it happens.
