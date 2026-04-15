# Experiment 02: Alignment Prior Computation & Bucket Structure Analysis

## Motivation

The paper (Compel-ZipMix) has placeholder results throughout — every claim about gains (+Y pp, X× faster) is unsubstantiated.
Before committing GPU hours to proxy model training, we need to **validate the core premise on real data**: that compression-ratio buckets create meaningful partitions, and that ZIP-FIT alignment scores concentrate weight in the "Goldilocks zone" rather than in noise.

This experiment is purely CPU-based (no model training), making it the cheapest and fastest way to:
1. Validate the scientific hypothesis before expensive training
2. Produce actual figures and tables for the paper (CR distributions, alignment weight heatmaps)
3. Answer multiple open questions from the proposal simultaneously
4. Generate the alignment priors α^(0) that all downstream training experiments consume

**If this experiment shows that alignment priors are flat (uniform across buckets) or concentrate on noise buckets, we know ZipMix has a fundamental problem before spending GPU time.**

---

## Research Questions

| ID | Question | Why It Matters |
|----|----------|----------------|
| Q1 | What do CR distributions look like across FineWeb-EDU, DCLM, and The Pile? | Validates Compel's "Goldilocks zone" premise; determines if CR is a useful partition axis at all |
| Q2 | Does the ZIP-FIT alignment prior α^(0) concentrate weight in the mid-CR buckets (B4–B5 in the proposal)? | Core ZipMix hypothesis — if it doesn't, the method is broken |
| Q3 | How sensitive is α^(0) to the number of buckets M ∈ {3, 5, 7, 10}? | Open Question #1 from proposal — determines final bucket count |
| Q4 | Uniform vs. quantile-based bucket boundaries — which gives more stable priors? | Open Question #2 from proposal — affects bucket design |
| Q5 | How sensitive is α^(0) to the choice of validation suite V? | Open Question #4 from proposal — determines how "retargetable" ZipMix really is |
| Q6 | Do CR distributions differ enough across corpora to warrant corpus-specific bucket thresholds? | Open Question #5 from proposal — determines if method is corpus-universal |
| Q7 | What is the per-document compute cost of ZIP-FIT scoring at scale (docs/sec)? | Practical feasibility claim: "single CPU-only ZIP-FIT pass" |

---

## Design

### Phase 1: Corpus Acquisition & CR Computation

**Corpora** (use streaming subsets — we don't need full corpora for distribution analysis):
- **FineWeb-EDU** — 500K documents sampled uniformly (HuggingFace: `HuggingFaceFW/fineweb-edu`)
- **DCLM-Baseline** — 500K documents sampled uniformly (HuggingFace: `mlfoundations/dclm-baseline-1.0`)
- **The Pile** — 500K documents sampled uniformly (HuggingFace: `EleutherAI/the_pile_deduplicated` or `monology/pile-uncopyrighted`; note: the original `EleutherAI/pile` was removed from HuggingFace — use a publicly available mirror)

**Compression ratio computation:**
```python
import lz4.frame

def compression_ratio(text: str) -> float:
    raw = text.encode('utf-8')
    compressed = lz4.frame.compress(raw)  # default level (LZ4 HC level 0 = fast mode)
    return len(compressed) / len(raw)
```

Use LZ4 (matching Compel) — NOT gzip. LZ4 is ~10x faster and was used in the original paper.

**Preprocessing:** Filter out documents with < 512 bytes raw text. Short documents produce unreliable CR values and noisy NCD scores (gzip/LZ4 framing overhead dominates for small inputs).

**Reproducibility:** All random operations (corpus subsampling, stratified NCD subsample, validation suite subsampling, bootstrap resamples) use fixed seeds. Default seed = 42; report results for seeds {42, 123, 456} on a 10K-doc pilot to confirm seed insensitivity.

**Output:** Per-document CSV: `(doc_id, corpus, cr, byte_length)`

### Phase 2: CR Distribution Analysis (answers Q1, Q6)

**Analyses:**
1. **Histograms** — CR distribution per corpus (100 bins, [0, 1.1])
2. **CDFs** — Overlay all 3 corpora on one CDF plot
3. **Summary statistics** — Mean, median, Q1, Q3, IQR, skewness per corpus
4. **Quartile comparison table** — Compare against Compel's reported quartiles (FineWeb-EDU: Q1=0.67, Median=0.73, Q3=0.78; DCLM: Q1=0.69, Median=0.75, Q3=0.82)
5. **KS test + effect size** — Pairwise Kolmogorov-Smirnov tests between corpora to quantify distributional differences (Q6). **Important:** At n=500K, KS will reject virtually any non-identical pair (the test has excessive power at large n). Always report the KS statistic D alongside the p-value; D is the maximum CDF gap and serves as the effect size. Supplement with Cohen's d on the CR means and the Earth Mover's Distance (Wasserstein-1) to give a scale-invariant measure of how much the distributions actually differ. Decision rule: KS p<0.01 *and* D>0.05 together indicate a practically meaningful distributional difference.
6. **Bucket population counts** — For each bucket configuration, how many docs land in each bucket per corpus?

**Key figure for paper:** Three-panel histogram with Compel's [0.65, 0.80] band shaded, bucket boundaries overlaid.

### Phase 3: ZIP-FIT Alignment Scoring (answers Q2, Q5, Q7)

**Validation suites (V):**
| Suite | Components | Rationale |
|-------|-----------|-----------|
| V_agi | MMLU-val + GSM8K-dev + MiniF2F-dev | The "AGI suite" from the proposal |
| V_mmlu | MMLU-val only | Pure knowledge/reasoning |
| V_math | GSM8K-dev + MiniF2F-dev + MATH-dev | Math-focused |
| V_code | HumanEval + MBPP (docstrings+signatures) | Code-focused |
| V_broad | MMLU-val + GSM8K-dev + MiniF2F-dev + HellaSwag-dev + ARC-Challenge-dev | Broadest coverage |

**Validation suite details:**
- **MMLU-val:** Use the MMLU *validation* split (sometimes called "val"), which has ~1.5K items across all 57 subjects. Do NOT use the 5-shot "dev" split (only ~285 items). Concatenate all 57 subjects. When referencing this in code, use `split="validation"` in the HuggingFace `datasets` loader (e.g., `load_dataset("cais/mmlu", "all", split="validation")`). Note: the proposal calls this "MMLU-dev" but we use "MMLU-val" to avoid confusion with the 5-shot dev split.
- **GSM8K-dev:** GSM8K does not have an official dev split. Use a fixed random subsample of 200 items from the train split (seed=42). Alternatively, use the first 200 items sorted by index.
- **MiniF2F-dev:** MiniF2F contains formal math statements (Lean/Isabelle). For NCD computation, use the *natural-language problem statements* (not the formal proofs). These are available in the `informal_statement` field of the dataset. If using the HuggingFace version, extract the informal statement; if using the GitHub repo, parse from the docstrings.
- **MATH-dev:** The MATH dataset (Hendrycks et al.) has train and test splits but no standard dev split. Create one by taking a fixed random subsample of 200 items from the train split (seed=42), stratified across the 7 difficulty levels.
- **HumanEval + MBPP (docstrings only):** Use only the docstring/prompt field (not the canonical solution) to avoid leaking solution code into the NCD computation. Include the function signature together with the docstring, since the signature carries type and naming information relevant to code-task alignment.

**Validation suite subsampling:** For suites with > 200 items (e.g., MMLU-val has ~1.5K), randomly subsample to 200 representative items, stratified by subject/category where applicable. This keeps the per-document NCD compute bounded while preserving coverage. Report sensitivity to this choice by comparing scores at 100, 200, and 500 items on a 1K-doc pilot.

**NCD computation:**
```python
import gzip

def ncd_gzip(x: bytes, y: bytes, level: int = 9) -> float:
    cx = len(gzip.compress(x, compresslevel=level))
    cy = len(gzip.compress(y, compresslevel=level))
    cxy = len(gzip.compress(x + y, compresslevel=level))
    return (cxy - min(cx, cy)) / max(cx, cy)

def zipfit_score(doc: str, val_suite: list[str]) -> float:
    d = doc.encode('utf-8')
    scores = [1 - ncd_gzip(d, v.encode('utf-8')) for v in val_suite]
    return max(scores)  # max-similarity (following ZIP-FIT paper)
```

**Note:** NCD uses gzip (not LZ4) — this is deliberate and matches the ZIP-FIT paper. LZ4 is for CR computation; gzip's higher compression gives more discriminative NCD scores.

**Subsample for NCD:** NCD is O(n) per pair, so use a **stratified subsample** of 50K docs per corpus (stratified by CR quintile to preserve distribution shape). Each call to `zipfit_score` computes NCD against every item in the validation suite and returns the max. Total NCD calls = 50K docs × ≤200 val items × 5 suites × 3 corpora = up to 150M NCD pair evaluations (each requiring 3 gzip calls). To keep this tractable: (1) subsample each validation suite to at most 200 representative items, and (2) parallelize aggressively with `multiprocessing`.

**Output:** Per-document CSV: `(doc_id, corpus, cr, s_agi, s_mmlu, s_math, s_code, s_broad)`

**Timing:** Measure wall-clock time for 1K, 10K, 50K docs to extrapolate to full-corpus cost (Q7).

### Phase 4: Alignment Prior Analysis (answers Q2, Q3, Q4, Q5)

**Bucket configurations to test:**

| Config | M | Boundary Strategy | Boundaries |
|--------|---|-------------------|------------|
| U3 | 3 | Uniform CR | [0, 0.33, 0.67, 1.0+] |
| U5 | 5 | Uniform CR | [0, 0.20, 0.40, 0.60, 0.80, 1.0+] |
| U7 | 7 | Uniform CR | [0, 0.143, 0.286, 0.429, 0.571, 0.714, 0.857, 1.0+] |
| P7 | 7 | Proposal (non-uniform) | [0, 0.50, 0.60, 0.67, 0.73, 0.80, 0.90, 1.0+] (from Experiment 01) |
| U10 | 10 | Uniform CR | [0, 0.10, 0.20, ..., 0.90, 1.0+] |
| QT3 | 3 | Quantile | Corpus-specific terciles |
| QT5 | 5 | Quantile | Corpus-specific quintiles |
| QT7 | 7 | Quantile | Corpus-specific septiles |
| QT10 | 10 | Quantile | Corpus-specific deciles |

**Scope note:** The full grid is 9 configs × 5 suites × 3 corpora = 135 triples. To keep the analysis interpretable, designate a **primary slice** for the main paper figures: P7 config × V_agi suite × all 3 corpora (3 triples). The remaining 132 triples are sensitivity analyses reported in tables/appendix. All scripts should compute the full grid, but narrative and figures should lead with the primary slice.

**For each (corpus × bucket_config × validation_suite) triple, compute:**

```
α^(0)_m = (Σ_{d ∈ B_m} s(d) + τ) / (Σ_j Σ_{d ∈ B_j} s(d) + M · τ),  τ = 1e-3
```
where M = number of buckets, s(d) = zipfit_score(d, V) as defined in Phase 3, and τ is the Laplace smoothing constant ensuring no bucket has zero weight. Note: with ~50K docs and scores in [0,1], the total score sum is O(10^3–10^4), so τ=1e-3 adds negligible mass (M·τ < 0.01) — it only prevents division-by-zero for empty buckets. If a bucket is empty, its prior weight will be approximately τ / (total_score + M·τ) ≈ 0, which is the desired behavior.

**Bucket boundaries note:** The last bucket in every configuration is open-ended (no upper CR bound). Notation "1.0+" means [c_{M-1}, ∞). LZ4 can produce CR > 1.0 on high-entropy inputs due to framing overhead, so the last bucket must catch these documents rather than dropping them.

**Analyses:**
1. **Alignment prior bar charts** — α^(0) per bucket for each config (key paper figure)
2. **Prior concentration metric** — Entropy of α^(0) vs. entropy of uniform (lower = more concentrated = more opinionated prior)
3. **Bucket-weight heatmap** — (bucket × validation_suite) matrix showing how different V's shift weight
4. **Stability analysis** — Variance of α^(0) across 1000 bootstrap resamples of the 50K subsample (report 95% confidence intervals; bootstrap is cheap since it only resamples pre-computed scores)
5. **Goldilocks validation** — Does the prior peak in buckets containing the [0.65, 0.80] CR range? Formally: test whether the summed weight of buckets overlapping [0.65, 0.80] exceeds the uniform expectation 1/M × (number of overlapping buckets) using a one-sided permutation test (permute bucket labels 10K times). This is the key test: if ZipMix's alignment prior independently recovers Compel's manually-tuned band, that's powerful evidence for the method.
6. **NCD score distributions** — Before aggregating into priors, plot the raw per-document NCD score distributions: (a) histogram of s(d) across all docs, (b) scatter plot of s(d) vs. CR(d) to visualize their relationship, (c) Spearman rank correlation between s(d) and CR(d). This reveals whether NCD scores have enough variance to produce meaningful priors.
7. **Length-conditioned analysis** — NCD is sensitive to document length (longer documents tend to have lower NCD values because the validation item's contribution to joint compression is proportionally smaller). Bin documents by byte length (e.g., quartiles) and recompute α^(0) within each length bin to check whether the Goldilocks pattern persists after controlling for length. Also compute partial correlation of s(d) with CR(d) after controlling for byte_length.
8. **LZ4 vs. gzip CR rank correlation** — Since CR is computed with LZ4 but NCD uses gzip, compute gzip-CR on the 50K subsample and report Spearman rank correlation between LZ4-CR and gzip-CR. If the correlation is high (>0.95), the compressor mismatch is benign; if low, it suggests documents may be bucketed differently under gzip, which could confound the alignment prior.

---

## Expected Outcomes & Decision Criteria

| Outcome | Interpretation | Next Step |
|---------|---------------|-----------|
| α^(0) concentrates on mid-CR buckets (B4–B5) across all corpora and suites | **Strong validation** — ZipMix alignment prior independently recovers Compel's Goldilocks zone | Proceed to Experiment 03 (proxy training) |
| α^(0) is nearly uniform | **Weak signal** — ZIP-FIT NCD doesn't discriminate well at the bucket level | Investigate per-document scores; consider finer buckets or alternative NCD variants |
| α^(0) concentrates on HIGH-CR buckets (B6–B7) | **Surprising** — high-entropy data is most "aligned" | Investigate: could indicate NCD bug, length confound (long high-CR docs dominating), or that the validation suite itself is high-entropy (e.g., math/code suites). Check per-suite breakdown before concluding failure. |
| CR distributions are nearly identical across corpora | **Simplifies design** — universal bucket boundaries work | Use one boundary set for all corpora |
| CR distributions differ significantly across corpora (KS p < 0.01 AND D > 0.05) | **Complicates design** — need corpus-specific thresholds | Report this in the paper; use quantile-based boundaries |
| Changing V dramatically shifts α^(0) | **Retargetability confirmed** — a key selling point of ZipMix | Emphasize in paper; plot the shift as a figure |
| Changing V barely shifts α^(0) | **Retargetability is weaker than claimed** | Temper claims in paper; the prior is mostly driven by CR structure, not task alignment |
| Goldilocks pattern holds for some corpora but not others | **Method is corpus-dependent** | Report which corpora succeed/fail; use quantile-based boundaries for the failing corpus and test whether that recovers the pattern. If it does, corpus-specific thresholds are mandatory. |
| α^(0) differs across corpora but NOT across validation suites | **CR structure dominates task signal** | The prior is mostly a CR artifact; ZIP-FIT scoring adds little. Investigate whether bucket population imbalance (rather than NCD scores) is driving the prior shape. |

---

## Deliverables

1. **`scripts/01_compute_cr.py`** — Streams corpora from HuggingFace, computes LZ4 CR, saves CSV
2. **`scripts/02_compute_ncd.py`** — Computes ZIP-FIT NCD scores against validation suites
3. **`scripts/03_compute_priors.py`** — Computes α^(0) for all (corpus × config × suite) triples
4. **`scripts/04_analysis.py`** — Generates all figures, tables, and statistical tests
5. **`results/figures/`** — Publication-ready figures (CR distributions, alignment prior bar charts, heatmaps)
6. **`results/tables/`** — LaTeX-formatted tables (summary statistics, alignment priors, KS tests)
7. **`results/alignment_priors.json`** — Machine-readable priors for downstream experiments to consume
8. **`report.md`** — Summary of findings with interpretation and decision on next steps

---

## Compute Requirements

| Step | Resource | Estimate |
|------|----------|----------|
| CR computation (1.5M docs) | CPU, 16 cores | ~30 min (LZ4 is fast: 40K+ docs/sec per Compel) |
| NCD scoring (150K docs × 5 suites × ≤200 val items each) | CPU, 16 cores | ~8–16 hours (gzip NCD is slow; parallelizable across docs and suites) |
| Prior computation + analysis | CPU, single core | ~5 min |
| **Total** | **CPU only — no GPU** | **~9–17 hours wall clock** |

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| NCD scoring too slow at 50K scale | Start with 10K pilot; extrapolate timing; parallelize with `multiprocessing` |
| HuggingFace streaming fails or is rate-limited | Cache downloaded subsets to local disk; use `datasets` library with retry logic |
| gzip NCD is too noisy for short documents | Filter to docs with ≥512 bytes raw; report length-conditioned analysis |
| Validation suite data not freely available | MMLU/GSM8K/HellaSwag/ARC are all publicly available; MiniF2F may need special handling (formal math — use text descriptions) |
| Val-suite subsampling distorts scores | Run pilot at 100/200/500 items to measure score stability; if max-similarity saturates early, 200 items suffices |

---

## Connection to Paper

This experiment directly fills the following paper sections:
- **Section 3 (Experiments):** CR distribution figures, bucket population tables
- **Section 2.2 (Step 1: Compression buckets):** Empirical justification for bucket boundaries
- **Section 2.3 (Step 2: Alignment prior):** The actual α^(0) values and their interpretation
- **Section 5 (Discussion):** Sensitivity analyses, limitations of validation suite dependence

**Critical paper figure:** A 3-panel figure showing (a) CR histogram with Compel band, (b) alignment prior bar chart showing weight concentration in Goldilocks zone, (c) heatmap showing how different V's shift the prior. This single figure tells the entire ZipMix story.

---

## Dependencies

- **Python packages:** `datasets`, `lz4`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`
- **Data:** HuggingFace Hub access (public datasets)
- **Predecessor:** Experiment 01 (proposal — defines bucket boundaries and alignment prior formula)
- **Successor:** Experiment 03 (proxy model training — consumes the α^(0) priors computed here)
