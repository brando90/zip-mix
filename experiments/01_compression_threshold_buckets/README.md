# Experiment 01: Compression Threshold Buckets — From Hard Filtering to Soft Weighting

## The Idea

Compel (Miranda et al., under review) showed that filtering pre-training data by compression ratio (CR) improves downstream performance: keeping documents with CR in [0.65, 0.80] yields +0.5-1.1% across FineWeb, FineWeb-EDU, and DCLM at 1.4B and 8B scale. David Hall (Stanford/Hazy Research) reviewed the idea and said "even just the thresholding idea makes a lot of sense" and expected Percy Liang to endorse it.

**The key evolution from Compel to ZipMix:** Instead of hard-filtering (binary keep/discard), partition the corpus into compression-ratio buckets and assign each bucket a soft weight derived from ZIP-FIT alignment scores to a validation suite. This preserves the full corpus while concentrating sampling on the most task-relevant information-dense regions.

## Background: Compel's Contribution

- **Method:** Compute LZ4 compression ratio CR(d) = bytes_zip(d) / bytes_raw(d) for each document. Retain only documents with CR in [0.65, 0.80].
- **Insight:** Low CR = repetitive/boilerplate (e.g., keyword stuffing), High CR = noisy/unnatural (e.g., HTML spam, mixed-language metadata). The "Goldilocks zone" contains information-rich, well-edited text.
- **Results at 1.4B:** FineWeb +1.1pp, FineWeb-EDU +0.7pp, DCLM +0.4pp (macro-avg accuracy across 13 benchmarks)
- **Results at 8B:** FineWeb +0.2pp macro / +0.6pp micro avg
- **Limitation:** Hard thresholds are manually tuned, global (ignore domain variability), and discard potentially useful data at the margins.

## What This Experiment Tests

### Step 1: Compression Buckets (extends Compel)
Instead of a single band [0.65, 0.80], partition the full CR range into M buckets:
- B1: CR in [0, 0.50) — highly compressible (very repetitive)
- B2: CR in [0.50, 0.60) — moderately repetitive
- B3: CR in [0.60, 0.67) — low-information but structured
- B4: CR in [0.67, 0.73) — core quality band (lower half)
- B5: CR in [0.73, 0.80) — core quality band (upper half)
- B6: CR in [0.80, 0.90) — high-entropy, possibly noisy
- B7: CR in [0.90, 1.0+] — very high entropy (noise/spam)

Thresholds above are motivated by FineWeb-EDU quartiles (Q1=0.67, Median=0.73, Q3=0.78) and DCLM quartiles (Q1=0.69, Median=0.75, Q3=0.82) from the research journal.

### Step 2: Alignment Prior (the ZipMix innovation)
For each bucket B_m, compute alignment weight:

```
alpha^(0)_m = (sum_{d in B_m} s(d) + tau) / (sum_j sum_{d in B_j} s(d) + M * tau)
```

where `s(d) = 1 - NCD_zip(d, V)` is the ZIP-FIT similarity to validation suite V, and tau = 1e-3 is a smoothing constant.

### Step 3: Training Variants
1. **ZipMix-Static:** Sample directly from alpha^(0) — hierarchical sampling: pick bucket m ~ Categorical(alpha), then x ~ Uniform(B_m)
2. **Loss-Reweighted AWPT:** Uniform dataloader but scale loss by alpha^(0)_{m(x)}
3. **ZipMix-DRO:** Refine alpha^(0) with one Group-DRO pass on a proxy model

## Baselines

1. **Uniform / Stratified sampling** — the default (surprisingly hard to beat per Aioli)
2. **Token proportional sampling** — crucial baseline per dlwh ("it's worse than token proportional sampling")
3. **Compel hard-filter** — the [0.65, 0.80] band from the Compel paper
4. **DoReMi** — the established method (known to fail in some settings)
5. **No filtering** — raw corpus

## Datasets and Models

- **Datasets:** FineWeb-EDU, DCLM, The Pile (use at least 2 to show generality)
- **Validation suite (V):** MMLU-val, GSM8K-dev, MiniF2F-dev (the "AGI suite") — note: "MMLU-val" refers to the MMLU validation split (~1.5K items), not the 5-shot dev split (~285 items); see Experiment 02 plan for details
- **Proxy model:** ~150M-280M parameters (following DoReMi convention)
- **Target models:** proxy x10 (~1.4B) and proxy x100 (~14B), staying below 10^3 dark-matter limit
- **Training framework:** Stanford Marin (per conversation with Elyas and David Hall)
- **Evaluation:** lm-eval-harness, same 13 benchmarks as Compel + VeriBench for domain-specific

## Key Hypotheses

**H_prior:** ZipMix-Static (with alignment prior) will outperform both uniform mixing and Compel hard-filtering because it uses task-relevant soft weighting instead of binary keep/discard.

**H_max:** ZipMix-DRO will outperform ZipMix-Static because the DRO refinement tempers the over-pessimistic worst-case objective of vanilla Group-DRO by starting from an informed prior (not uniform).

**H_transfer:** Mixture weights learned at proxy scale will transfer to 10x and 100x models (staying below 10^3 scaling limit).

## Expected Outcomes

- ZipMix-Static beats uniform by +Y pp (where Y > Compel's 0.5-1.1pp gains)
- ZipMix-DRO reaches baseline accuracy X times faster
- Both transfer unchanged to larger models
- Compel hard-filter is a strict subset of ZipMix (recoverable by setting extreme bucket weights to 0)

## Open Questions

1. How many buckets? Too few = coarse, too many = noisy estimates. Ablate M in {3, 5, 7, 10}.
2. Should bucket boundaries be uniform in CR space, or quantile-based (equal documents per bucket)?
3. Where does Compel/ZipMix fit in the FineWeb pipeline? (Before or after existing filters?)
4. How sensitive is the alignment prior to the choice of validation suite V?
5. Does the CR distribution differ enough across corpora to warrant corpus-specific thresholds?

## Connection to Related Work

- **Compel** (this experiment's foundation): Hard-filter by CR band
- **BETR** (Mizrahi et al., 2025): Neural embedding hard-filter; ZipMix is model-free + soft-weighted
- **DoReMi** (Xie et al., 2023): Group-DRO on hand-defined domains; ZipMix replaces domains with CR buckets + adds alignment prior
- **Aioli** (Chen et al., 2025): Online estimation of mixing-law parameters; complementary to ZipMix's offline prior
- **Training on Test Task** (Dominguez-Olmedo et al., 2025): Must address contamination objection
- **Contamination scaling laws** (Schaeffer, Miranda et al.): Upper bound on alignment-based selection
