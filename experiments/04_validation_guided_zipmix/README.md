# Experiment 04: validation-guided Zip-Mix generalization screen

**Doc link:** <https://github.com/brando90/zip-mix/blob/main/experiments/04_validation_guided_zipmix/README.md>

**TLDR:** Test whether compression-based validation alignment improves held-out language modeling over strong simple controls and compact implementations of DoReMi and DoGE. This is an initial mechanism screen, not evidence yet of optimal training or state-of-the-art benchmark performance.

Started: 10-04-2026. Publication pending until checks and requested review finish.

## Goal and decision

The consequential uncertainty is whether compression buckets preserve target-relevant information after controlling for bucket population and text length. The first test uses the existing 20,000-document Pile pilot, a biomedical development set, and independent biomedical and non-biomedical held-out documents. A positive result requires a practically meaningful target-loss improvement over token-proportional sampling and the shuffled-score control without a material broad-domain regression. Every method and seed remains in the report even if it loses. See [protocol](expt_v1/PROTOCOL.md) for precise bounds and [research design](research_design.md) for larger confirmation and post-training.

## Method

1. Deduplicate and split by document identity before fitting a tokenizer or scoring alignment. Remove training documents sharing a normalized 13-word sequence with retained development/evaluation prefixes.
2. Train the byte-pair tokenizer only on training text. Use equal-length, within-document token blocks so all final models see an identical training-token budget.
3. Score fixed-byte document prefixes against development text. Preserve the draft's maximum-similarity Zip-Mix score and separately compute the mean aggregation used in the published ZIP-FIT method.
4. Freeze nine methods and three paired initialization seeds. Compare target and broad held-out negative log-likelihood; report perplexity as its exponential. Include reference/proxy training and scoring overhead in cost accounting.
5. Review code and protocol with explicitly requested Claude Opus 5.5 at maximum effort, execute the full manifest durably on one Stanford Network Analysis Project graphics processing unit, and retain all results.

## Status

| Step | Status | Evidence |
|---|---|---|
| Existing project/configuration inspection | Done | Preserved pre-existing uncommitted work; current global rules read |
| Literature and baseline corrections | In progress | [Research design](research_design.md) |
| Data preparation and provenance | In progress | `expt_v1/prepare.py`; ignored data directory |
| Training implementation and deterministic tests | In progress | `expt_v1/` |
| Requested Opus 5.5 maximum-effort review | Done; fixes applied | [Review report](qa/opus55_max_review.md) |
| Full pretraining screen | Pending | [Live results](results.md) |
| Benchmark-level supervised fine-tuning | Planned separately | Experiment 05 |

## Structure

```text
04_validation_guided_zipmix/
  README.md, research_design.md, preflight.md
  results.md, CKPT_zipmix.md
  expt_v1/
    PROTOCOL.md, cc.md, config.json
    prepare.py, train.py, analyze.py, test_train.py
    data/       # ignored raw/tokenized data and tokenizer
    runs/       # ignored checkpoints; small report receipts published separately
```

## Dependencies and scope

Python with PyTorch, NumPy, SciPy, pandas, PyArrow, tokenizers and LZ4. No paid model-provider calls, external dashboards or generated labels are used for training. The source pilot is `../02_alignment_prior_analysis/data_cache/pile_texts.parquet`; its source revision was not recorded by the earlier pilot, so a local SHA-256 file hash (a cryptographic content identifier) is frozen instead. These inherited sampling limitations prevent population-wide claims about the full Pile corpus.

Large data, models and checkpoints live in an isolated node-local runtime, with verified hashes and a private deployment receipt. The public repository contains scripts, configuration, provenance hashes and aggregate/per-example numerical evidence, not copied corpus text or credentials.

## Relation to earlier proposals

Experiment 03 remains unchanged. Its proposed mid-compression concentration gate is replaced **for this new experiment only** by a leakage audit and actual held-out training test: a peaked prior alone does not validate utility. DoReMi's reference/excess-loss correction is implemented explicitly rather than interpreting raw difficult-domain loss as its method. Stronger pretrained benchmark tests and contemporary baselines are required before any superiority claim.

