# Compel-ZipMix: Compression-Aligned Mixtures for Efficient Task-Aware Language-Model Pre-training

Mixture composition is a pivotal yet under-explored axis of large-language-model pre-training. **Compel-ZipMix** partitions the corpus into compression-ratio buckets and assigns each bucket a weight derived from a ZIP-based similarity to a small validation suite, enabling *practically free, domain-conditioned* foundation checkpoints.

## Core Idea

> **You already know what you want the model to be good at — so let that goal choose the pre-training data.**

A practitioner can specify their goal *concretely*: through **examples / data**, or through a **rubric / constitution** (which can itself be generated as **synthetic data**). That specification is enough to assemble a small **validation "benchmark" set** that operationalizes the goal — a compact, machine-checkable picture of what "good" looks like.

Given that benchmark and the **entire** (pre-)training corpus, Compel-ZipMix assigns **every** piece of training data a **weight according to how well it aligns with the goal**, then trains on the resulting goal-aligned mixture. The guiding principle:

> **benchmark ≈ pre-train → best performance.**

This turns mixture design from a fixed, hand-tuned recipe into a **goal-conditioned** one. Change the validation benchmark — e.g. swap general MMLU for Lean / MiniF2F / a synthetic rubric — re-score the corpus, and you get a *new* mixture targeted at the *new* goal, for the cost of a single CPU-only scoring pass. No human-defined domains, no uniform prior, no retraining the data pipeline.

**How "alignment to the goal" is measured cheaply — the ZipMix mechanism.** Scoring a whole corpus against the goal has to be fast and model-free, so ZipMix uses **compression** as the alignment signal:

1. **Compression buckets** partition the corpus by compression ratio `CR(d) = bytes_zip(d) / bytes_raw(d)`, a model-free proxy for information density.
2. **Alignment prior** weights each bucket by its **ZIP-FIT** similarity (normalized compression distance) to the validation benchmark `V` — i.e. how "goal-like" the data in that bucket is.

Two project goals motivate this (see the [idea deck & research journal](https://docs.google.com/presentation/d/1cI569bcyQpxB66MUnflvBBXJKxe26GDodLig1xQ1iEI/edit?slide=id.g372a0b56031_0_223#slide=id.g372a0b56031_0_223)):

- **Goal 1 — fix DoReMi's failure modes.** Choose mixture weights for *general intelligence* + *self-improvement* + *domain-specific* skills (e.g. coding / Lean), instead of DoReMi's uniform prior over coarse, hand-defined domains. Human-labeled mixtures vs. unsupervised, information-density/compression-based ones.
- **Goal 2 — pre-train a general + domain-specific model.** Concretely, a foundation model for Lean: translate a large Python corpus to Lean (via the VeriBench agent), filter by compilation, mix into FineWeb-Edu, and let the goal-aligned weighting do the rest.

📊 **Full idea, motivation, and running research journal:** <https://docs.google.com/presentation/d/1cI569bcyQpxB66MUnflvBBXJKxe26GDodLig1xQ1iEI/edit?slide=id.g372a0b56031_0_223#slide=id.g372a0b56031_0_223>

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
| Idea deck & Research Journal | https://docs.google.com/presentation/d/1cI569bcyQpxB66MUnflvBBXJKxe26GDodLig1xQ1iEI/edit?slide=id.g372a0b56031_0_223#slide=id.g372a0b56031_0_223 |
| Compel (predecessor, ICLR 2026 submission) | https://openreview.net/forum?id=KFafeqE5fe |
| ZIP-FIT (compression-based alignment) | https://arxiv.org/abs/2410.18194 |
| Alignment Coefficient | https://arxiv.org/abs/2501.08496 |

## Repository Structure

```
zip-mix/
  latex_paper/          # ICLR 2025 submission (Compel-ZipMix paper)
    main.tex            # Main paper source
    math_commands.tex   # Math macros
    zipmix_refs.bib     # Bibliography
  experiments/
    00_related_work/    # Comprehensive literature review
    01_compression_threshold_buckets/  # Core experiment: CR buckets + alignment prior
    02_alignment_prior_analysis/       # CPU-only: CR distributions, ZIP-FIT priors, bucket sensitivity
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

- **Brando Miranda** (Stanford) — bmiranda@stanford.edu
- **Elyas Obbad** (Stanford) — eobbad@stanford.edu
- **Sanmi Koyejo** (Stanford) — sanmi@stanford.edu

## Builds on

- [Compel](https://openreview.net/forum?id=KFafeqE5fe) — Compression-ratio filtering for pre-training data
- [ZIP-FIT](https://arxiv.org/abs/2410.18194) — Embedding-free data selection via compression-based alignment
- [Diversity Coefficient](https://arxiv.org/abs/2306.13840) — Data quality metric for variability in NL data
