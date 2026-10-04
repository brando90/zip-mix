## Scientific question (locked)

Can validation-guided compression mixtures improve multiple-choice science accuracy after small supervised fine-tuning (SFT), compared with specified sampling controls, using exactly the same number of supervised answer tokens? This experiment is a screening study. It does not establish optimality, broad generalization, or superiority to methods absent from the manifest.

Frozen on 10-04-2026, before any measured training or benchmark scoring. Exact public revisions and numerical constants live in `common.py`; changing any scientific choice creates another version or separately identified prospective condition. The emitted data manifest contains its own hash and every prepared file's hash. The training run also fingerprints its implementation and analysis code.

### Inputs and split roles

Use `Qwen/Qwen2.5-0.5B` base revision `060db6499f32faf8b98477b0a26969ef7d8b9987`. The tokenizer is from the same revision. Models start from identical pretrained weights; seeds 0, 1, 2 vary sampling and runtime random-number generators.

Training source repositories are `allenai/sciq`, `allenai/openbookqa` configuration `main`, `allenai/ai2_arc` configuration `ARC-Challenge`, and `tau/commonsense_qa`; all use their official training splits. Pin each revision before downloading. Sort eligible items by a cryptographic hash of the normalized question and sorted choice texts; retain at most 3,000 per domain. Domains with fewer eligible items keep all eligible items. Preserve exclusion counts.

Alignment uses the first 64 eligible hash-ranked SciQ validation records, without answer labels. Target evaluation uses the first 500 eligible hash-ranked SciQ test records. Retention uses the first 500 eligible hash-ranked CommonsenseQA validation records. The cap is fixed before evaluation, not selected from outcomes. Evaluation examples are never used to select mixture weights, tune hyperparameters, select checkpoints, or stop early. There is no hyperparameter search in this version.

Remove training items whose normalized question or normalized question-plus-choice identity occurs anywhere in the full official alignment/target/retention splits. Also remove training items sharing any contiguous 13-word content sequence with those full held-out splits. Scan question and individual choice text separately, excluding prompt wrappers and answer labels. This intentionally conservative content rule does not certify absence of semantic paraphrases. Record removals. Deduplicate normalized training questions across sources deterministically. Remove exact or substantive near-duplicates from later held-out roles with priority alignment, target, retention before applying their caps. Missing the required held-out count is a preparation failure, not permission to reduce the cap silently.

Prompt: `Choose the correct answer.\nQuestion: {question}\nA. {choice_a}\n...\nAnswer:`. Continuations are ` A` through ` E`. Deterministically shuffle choices per content identity and remap the correct answer; never infer ordering from model predictions. Assert isolated continuation length and exact prompt-continuation tokenization for every record. Keep examples only when prompt plus answer token fits 256 tokens; do not truncate the question or choices. There is no end-of-sequence token in the supervised target.

### Compression and mixtures

Compute LZ4 framed compression ratio on the entire question plus labeled choice text, omitting the common instruction wrapper and answer label. Fit five quantile bins using training candidates only. Exact quantile edges and empty/tied bin counts are in the manifest.

Score each candidate's first at most 1,024 UTF-8 bytes against the same view of each alignment record. Drop only an incomplete trailing UTF-8 code point. Gzip compression level is 6 and modification time is zero. Use candidate-then-development concatenation and cache development compressed sizes. Keep raw per-pair similarity `1 - normalized_compression_distance`; store candidate maximum and mean. Shorter candidates remain shorter: this is a real length confound, not equal-length scoring. Report view lengths and use source-by-byte-length-quartile permutation as a negative control.

1. `sample_proportional`: uniform over eligible training examples. Each example supplies one supervised token, so this is also proportional to available supervised tokens, not proportional to all prompt tokens.
2. `uniform_domain`: equal mass per source, uniform within source.
3. `compel_filter`: uniform among documents with full-content LZ4 ratio in `[0.65, 0.80]`. Empty support fails preparation. Sparse support is reported and repetition remains visible.
4. `zipmix_static`: historical ZipMix maximum similarity. Clip negative scores to zero; sum scores per compression bin and normalize bin masses; sample uniformly within the chosen bin. Zero total score falls back to sample-proportional and is flagged. Bin sums retain population weighting.
5. `direct_zipfit`: published ZIP-FIT mean similarity. Select the top 25% of candidate documents by mean score, with content hash breaking ties; sample uniformly from that subset.
6. `shuffled_zipmix`: same bucket formula as `zipmix_static`, but permute document maximum scores within each source and byte-length quartile with fixed permutation seed 1405 before aggregation.

This six-arm design intentionally preserves the historical maximum-score ZipMix and published mean-score direct selector. Their contrast combines aggregation and granularity changes and cannot independently identify either effect. A mean-score bucket ablation belongs in a separate prospective expansion. A permutation of development-example order would leave both aggregations unchanged and is not used.

### Model training and evaluation

Full fine-tuning with float32 master parameters and optimizer states, using bfloat16 autocasting for forward computations on exactly one available NVIDIA A100 device. All parameters are trainable and there are no adapters. Float32 master updates prevent small optimizer steps from disappearing through bfloat16 rounding. Use AdamW, learning rate `2e-5`, weight decay `0.01`, betas `(0.9, 0.95)`, epsilon `1e-8`, gradient-norm clipping at `1.0`, eight linear warmup steps and a cosine decay to zero across 128 total optimizer steps. Batch size is 16 without gradient accumulation. All arms train for exactly 2,048 supervised answer tokens, with replacement.

Left-pad batches and supply attention masks and explicit positions from cumulative unmasked tokens. Score the actual final prompt position. Compute full-vocabulary cross entropy only for the next-token correct answer label; this is equivalent to a causal language-model loss with all prompt labels masked. Use last-position logits where supported to avoid allocating unnecessary sequence-by-vocabulary logits. Report the actual processed prompt-token count, padded token work, unique examples, repetition, and elapsed cost rather than implying equal total compute.

Evaluate the fixed final model, plus the base checkpoint once, on both frozen 500-item sets. Choose the answer with greatest final-position log probability among the valid single-token letter options. No decoding or generated-text parser is involved. Record per-item predictions/correctness and full-vocabulary correct-letter negative log likelihood. Correctness labels are read only by training-example loss or held-out evaluation; mixture scoring uses content without them.

### Manifest, resources, and recovery

The full declared training set is six methods × three seeds = 18 cells, plus one base evaluation. Each cell has a durable ledger row before execution. Run sequentially on one explicitly bound available device. The coordinator measures forward/training throughput and projects the full 18-cell runtime before admission. A global scheduler timeout must cover the complete bounded manifest plus setup, evaluation, saving, and cleanup. This script does not itself reserve a GPU or install an external watchdog.

Write atomic checkpoints every 16 optimizer steps and atomic progress after each step. A process-level interruption may resume a logical cell once from its last verified checkpoint with model, optimizer, sampler, and runtime random states. Record checkpoint step, replayed optimizer steps, and replayed input-token exposure. Replay is part of physical cost, not another independent seed. A second interrupted continuation exceeds the fixed bound and leaves that row failed. A caught numerical/runtime exception becomes a preserved failed row; do not automatically search new hyperparameters or retry it indefinitely. Complete other admitted rows.

Do not stop healthy cells because interim metrics are unfavorable, the coordinator's chat turn ends, or an early cell succeeds. A lock prevents two writers to one output directory. At completion, retain final weights, per-item predictions, receipts, and input/code identities; optimizer recovery files may be deleted only after those final artifacts are durable. All failed, interrupted, pending, or unattempted rows remain visible. Any missing required row or base evaluation makes the study incomplete.

### Analysis and interpretation

Report each arm's accuracy numerator/denominator, mean and standard deviation across seeds, and paired-seed accuracy difference against sample-proportional sampling. Also report historical ZipMix minus each of the five comparison methods. Use descriptive 95% Student t intervals over three paired seed differences, conditional on the finite evaluation sets; omit intervals with fewer than three complete pairs. State the normality assumption and instability at three seeds; do not treat 500 examples or many input tokens as independent model replications. Report unadjusted exploratory exact two-sided sign-test p-values against the named null that the method wins with probability 0.5 among non-tied seed pairs. The minimum p-value for three non-tied pairs is `2 / 2^3 = 0.25`. Ties and comparison-specific denominators are explicit. No confirmatory significance claim is made.

Completion itself is not scientific success. A favorable target result motivates a new study with held-out confirmation data, more seeds, matched total selection/training costs, and faithful relevant baselines. An unfavorable result still completes this screen. Untargeted CommonsenseQA retention does not establish transfer to a family absent from the training pool. All headline claims remain limited to the setup actually measured.
