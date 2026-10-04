# Compel-ZipMix: Compression-Aligned Mixtures for Efficient Task-Aware Language-Model Pre-training

Mixture composition is a pivotal yet under-explored axis of large-language-model pre-training. **Compel-ZipMix** partitions the corpus into compression-ratio buckets and assigns each bucket a weight derived from a ZIP-based similarity to a small validation suite, enabling *practically free, domain-conditioned* foundation checkpoints.

## Overview

DoReMi learns domain weights via Group-DRO, but empirical replications report accuracy losses: its coarse, hand-defined domains blur signal and noise, and its worst-case objective begins from an uninformative uniform prior. We trace this to two hypotheses:

- **H_prior (No task prior):** DoReMi starts from uniform weighting of coarse domains. By No-Free-Lunch, a learner with no prior preference is free to prefer entropy.
- **H_max (Over-pessimistic objective):** The inner maximisation conflates "hard because useful" with "hard because irreducible."

**ZipMix** addresses both:
1. **Compression buckets** partition the corpus by `CR(d) = bytes_zip(d) / bytes_raw(d)`, a model-free proxy for information density.
2. **Alignment prior** assigns each bucket a weight proportional to ZIP-FIT similarity to a validation suite (MMLU, GSM8K, MiniF2F, ...).

Two variants:
- **ZipMix-Static** samples directly from the alignment prior (addresses H_prior)
- **ZipMix-DRO** refines with one Group-DRO pass (tempers H_max)

## Key Links

| Resource | Link |
|---|---|
| Paper (Overleaf) | https://www.overleaf.com/project/6861da6f7c236a3466d7e749 |
| Slides | https://docs.google.com/presentation/d/1XoQ24_KofQUOeSxLE_lNVjYAw0lMtsm9spF6ertwL7s/edit |
| Research Journal | https://docs.google.com/presentation/d/1cI569bcyQpxB66MUnflvBBXJKxe26GDodLig1xQ1iEI/edit |
| Compel (predecessor, ICLR 2026 submission) | https://openreview.net/forum?id=KFafeqE5fe |
| ZIP-FIT (compression-based alignment) | https://arxiv.org/abs/2410.18194 |
| Alignment Coefficient | https://arxiv.org/abs/2501.08496 |

## Repository Structure

```
zip-mix/
  paper_latex_and_notes/
    ICLR_2025_CompelZipMix/  # ICLR 2025 submission (Compel-ZipMix paper)
    DMLR_2026_CompelZipMix/  # DMLR 2026 submission
  experiments/
    00_related_work/         # Comprehensive literature review
    01_compression_threshold_buckets/  # Core experiment: CR buckets + alignment prior
    02_alignment_prior_analysis/       # CPU-only: CR distributions, priors, bucket sensitivity
    03_zipmix_doremi_fix/              # Core experiment: ZipMix fixing DoReMi via validation priors
  src/                       # Source code (scripts and tools)
```

## Method Summary

### Step 1: Compression Buckets
For each document `d` in corpus D, compute:
```
CR(d) = bytes_zip(d) / bytes_raw(d)
```
Partition into buckets `B_m = {d : CR(d) in [c_{m-1}, c_m)}` using thresholds calibrated from high-quality reference datasets (FineWeb-EDU quartiles: Q1=0.67, Median=0.73, Q3=0.78).

### Step 2: Alignment Prior
Compute ZIP-FIT similarity `s(d) = 1 - NCD_zip(d, V)` to validation suite V. Bucket weight:
```
alpha^(0)_m = (sum_{d in B_m} s(d) + tau) / (sum_j sum_{d in B_j} s(d) + M * tau)
```

### Step 3: Training
- **ZipMix-Static:** Sample `m ~ Categorical(alpha^(0))`, then `x ~ Uniform(B_m)`
- **ZipMix-DRO:** Refine `alpha^(0)` with one Group-DRO pass, then sample as above

## Related Work

Key papers and positioning:

| Method | Type | Key Limitation (vs ZipMix) |
|---|---|---|
| DoReMi (Xie et al., NeurIPS 2023) | Group-DRO on hand-defined domains | Uniform prior, over-pessimistic objective, known failures |
| DoGE (Fan et al., 2023) | Gradient-alignment reweighting | Fails to beat uniform on 5/6 datasets (per Aioli) |
| Aioli (Chen et al., ICLR 2025) | Online mixing-law estimation | Very small models (160M primary); consistent but modest gains |
| BETR (Mizrahi et al., 2025) | Neural embedding hard-filter | GPU-dependent, model-dependent, binary keep/discard |
| Compel (Obbad, Miranda et al., 2025) | CR hard-filter [0.65, 0.80] | Hard threshold, no task alignment, modest gains (+0.5-1.1pp) |

See `experiments/00_related_work/literature_review.md` for the full review (30+ papers).

## Team

- **Brando Miranda** (Stanford) — brando9@stanford.edu
- **Elyas Obbad** (Stanford) — eobbad@stanford.edu
- **Sanmi Koyejo** (Stanford) — sanmi@stanford.edu

## Builds on

- [Compel](https://openreview.net/forum?id=KFafeqE5fe) — Compression-ratio filtering for pre-training data
- [ZIP-FIT](https://arxiv.org/abs/2410.18194) — Embedding-free data selection via compression-based alignment
- [Diversity Coefficient](https://arxiv.org/abs/2306.13840) — Data quality metric for variability in NL data
