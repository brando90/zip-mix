# Zip-Mix pretraining: live results

**Doc link:** <https://github.com/brando90/zip-mix/blob/main/experiments/04_validation_guided_zipmix/results.md>

Updated 10-04-2026 12:00 PDT. The full frozen 27-cell run started at 11:58 PDT on an A100. Its first cell passed step 896/2,048 with zero failed cells at the initial observation. Final target and broad held-out comparisons remain pending; superiority is unresolved.

Data preparation retained 14,341 training documents and 103,123 token blocks (13,199,744 available next-token positions). Development uses 64 target documents. Evaluation contains 128 target and 512 broad-domain documents. Exact deduplication removed 2 documents; the declared 13-word overlap filter removed 623 training candidates.

The original Zip-Mix maximum-similarity prior has total-variation distance 0.020707 from token-proportional bucket mass (a deterministic description of this frozen pool; population uncertainty unavailable, p-val=n/a). Thus it moves only 2.07% of probability mass. A small prior shift may produce no detectable training benefit. A peaked histogram is not a successful generalization result.

[Data provenance and numerical diagnostics](data_manifest.json) · [Protocol](expt_v1/PROTOCOL.md) · [Checkpoint](CKPT_zipmix.md) · [Research roadmap](research_design.md)

**TLDR-end:** [zip-mix: pretraining] The full 27-cell pretraining comparison is running under a durable supervisor. Generalization results remain pending.
