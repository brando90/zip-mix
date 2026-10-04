# Zip-Mix experiment index

**Doc link:** <https://github.com/brando90/zip-mix/blob/main/experiments/README.md>

**TLDR:** Experiments 04 and 05 initialize runnable validation-guided pretraining and supervised fine-tuning tests. Earlier proposals and the compression pilot remain preserved; no superiority result has been established.

| Home | Setup and goal | Status |
|---|---|---|
| [00_related_work](00_related_work/README.md) | Historical literature notes | Retained; current source-verified design is in Experiment 04 |
| [01_compression_threshold_buckets](01_compression_threshold_buckets/README.md) | Compression-bucket proposal | Proposal |
| [02_alignment_prior_analysis](02_alignment_prior_analysis/README.md) | Public corpus compression distributions and priors | Existing 20,000-document corpus pilots; full prior analysis incomplete |
| [03_zipmix_fixes_doremi_proxy_mixtures](https://github.com/brando90/zip-mix/blob/main/experiments/03_zipmix_doremi_fix/README.md) | Earlier robust-optimization proxy-mixture proposal | Designed, not run |
| [04_validation_guided_zipmix](04_validation_guided_zipmix/README.md) | Nine methods × three seeds; from-scratch target/broad held-out language-model loss | Initializing; [live results](04_validation_guided_zipmix/results.md) |
| [05_validation_guided_sft](05_validation_guided_sft/README.md) | Six methods × three seeds; Qwen2.5-0.5B supervised multiple-choice fine-tuning | Initializing; [live results](05_validation_guided_sft/results.md) |

Every active experiment keeps its code, versioned protocol, data manifest, resumable checkpoint and results in its canonical folder. Raw data/model files are ignored. Updates describe actual completion separately from a successful launch. Numerical evidence and complete denominators determine claims.
