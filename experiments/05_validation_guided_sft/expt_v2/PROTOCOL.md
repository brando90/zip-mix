# Experiment 05 version 2: fixed-byte validation targeting

**Doc link:** <https://github.com/brando90/zip-mix/blob/main/experiments/05_validation_guided_sft/expt_v2/PROTOCOL.md>

**TLDR:** Freeze seven methods and three seeds to test target-dependent compression mixtures with equal-byte scoring inputs and an explicit wrong-target control. Results are not yet measured.

## Scientific question (locked)

Does choosing a fixed-budget training mixture from validation-target compression similarity improve multiple-choice science accuracy, once the compressed candidate and target views all have exactly the same byte length?

Version 2 is a prospective repair created on 10-04-2026 before any Experiment 05 benchmark evaluation or model training. The requested single Opus 5.5 maximum-effort review identified a major length-confounding problem in the untrained version 1 calibration. Its inputs and source remain preserved. This version inherits its reviewed training/evaluation implementation and uses deterministic tests for the targeted repair; no second model-review round is requested.

### Preserved training and evaluation contract

Use the same immutable Qwen2.5-0.5B base revision, tokenizer, one-token answer labels, question prompts, frozen 500-item SciQ test and 500-item CommonsenseQA validation arrays, and source revision pins as version 1. No held-out scores have been used to create this version. Exact model/configuration values are in `common.py` and emitted manifests.

Each available method has three seeds (0, 1, 2), 128 optimizer steps × 16 examples = 2,048 supervised answer-label tokens. Full fine-tuning uses float32 master parameters and optimizer states, bfloat16 forward autocasting, AdamW at learning rate 2e-5, the same eight-step warmup and cosine schedule, and identical clipping/regularization. Evaluate the final checkpoint using actual final-position logits with correct left-padding and positional indices. Base evaluation runs once. Input-token exposure, padded work, repetition, and elapsed cost remain measured separately.

### Data split and real-content packing

Read the hashed, immutable version 1 prepared inputs and verify every file against its manifest. Reserve the first 256 hash-ranked CommonsenseQA training records exclusively for the wrong-target development condition, removing them from every training arm before packing. Use the pinned official SciQ validation split for true-target development. Version 1 removed training overlap against the full official held-out splits; recheck exact normalized questions/content and substantive 13-word content overlap against true development, reserved wrong development, and evaluation records. No answer labels enter compression scoring.

Within each training source independently, order records by their content hash and concatenate whole question-plus-choice strings separated by a single newline. Accumulate records until at least 4,096 real UTF-8 bytes exist. The scoring unit is the first exactly 4,096 bytes of that group. Every member retains its original full, at-most-256-token training prompt and inherits the group's score. The final member can extend beyond the scored prefix; this mismatch is recorded as a limitation. Start the next group with the next unused record. Discard the final underfilled group within each source before any method selects data. Never pad, repeat, or synthesize content to fill a scoring window.

Apply the same grouping rule separately to true and wrong development content. Use up to eight views per target, with equal view count `min(8, available_true_views, available_wrong_views)` and at least two views required. This availability rule is frozen before reading selector diagnostics. Every scoring view is a 4,096-byte prefix of actual encoded content; truncation may cut a trailing UTF-8 code point because compression consumes bytes, not decoded text. No artificial text padding is added.

Compression ratio uses LZ4 framed bytes of the fixed 4,096-byte view. Fit five quantile bins over training packs, not development data. Alignment uses gzip level 6, deterministic headers, and candidate-then-development concatenation. Store maximum and mean normalized-compression similarities against the true target and maximum similarity against the wrong target. Pack scores and bins are inherited by individual examples, each of which contributes one supervised training token.

Packing fixes the direct scoring-length confound. It does not remove source identity, repetition, genre, number of records per pack, or content-density effects. These limitations remain testable with the wrong-target and shuffled-score controls. The wrong target is related multiple-choice commonsense content, not semantically unrelated random bytes.

### Seven-method manifest

| Method | Frozen behavior |
|---|---|
| `sample_proportional` | Uniform over eligible examples after reservation/packing; proportional to available supervised answer tokens |
| `uniform_domain` | Equal mass per remaining source; uniform within source |
| `compel_filter` | Uniform over examples whose pack compression ratio is in [0.65, 0.80]; empty support becomes an explicit unavailable-method row, never a silent replacement |
| `zipmix_static` | Historical maximum true-target similarity, negative scores clipped to zero, summed per bucket and normalized; uniform within bucket |
| `direct_zipfit` | Mean true-target pack similarity; top 25% of examples by inherited score with hash tie-breaking, uniform within selected subset |
| `shuffled_zipmix` | Globally permute equal-byte pack maximum scores once with seed 1405, inherit the permuted score per member, then apply the same bucket aggregation |
| `wrongtarget_zipmix` | Same candidate bins and aggregation as ZipMix, scored against equally many 4,096-byte reserved CommonsenseQA views |

The direct ZIP-FIT arm is a pack-score adaptation using the paper's mean aggregation; it is not a faithful reproduction of document-level ZIP-FIT. Likewise Compel is transferred to packs of short benchmark questions. Names remain recognizable but reports must state these differences. Direct versus bucket selection also changes aggregation, so the contrast cannot isolate those effects independently.

The full denominator is seven methods × three seeds = 21 declared training cells. An unavailable method remains in that denominator with scheduler status `failed`, availability `unavailable_empty_support`, and failure kind `scientific_method_unavailable`; its absence is never interpreted as measured poor accuracy. All seven methods have support in the first emitted version 2 manifest, so all 21 cells are expected to train.

### Scientific admission and analysis

Before training, verify uniform 4,096-byte scoring lengths, split separation, source/pack coverage, finite weights, true-versus-wrong target prior differences, and a complete resource projection. Prior diagnostics describe whether the intervention changes data exposure. A favorable prior pattern is not a performance result, and parameters must not be changed to manufacture a benchmark gain. Freeze and complete the admitted training manifest even if early results are unfavorable.

Use the reviewed version 1 audit/analysis behavior extended to seven methods. Keep all cells, verify predictions and result receipts, report numerator/denominator and across-seed standard deviation, and use descriptive paired-seed Student t 95% intervals only with three complete pairs. Intervals condition on the fixed test examples and weak small-sample normality assumption. Report unadjusted exploratory exact two-sided sign-test p-values against the named null of 0.5 win probability among non-tied seeds; minimum p-value at three non-tied pairs is 0.25. Do not imply confirmatory significance or treat tokens/items as independent training replications.

Compare true-target ZipMix against all controls, especially wrong-target and shuffled-score ZipMix. Positive science accuracy differences with neutral retention would support a targeted-adaptation signal in this setup; broad generalization and comparison against faithful DoReMi, DoGE, LESS, or modern mixing methods remain separate work. CommonsenseQA training content remains in the candidate pool, so its validation score is untargeted retention/transfer rather than an unseen family.

### Execution and stopping

Use a single verified available A100 and a durable coordinator-owned supervisor with a timeout covering all 21 bounded cells plus setup, evaluation, checkpoint writes, analysis, and cleanup. The script saves every 16 steps and permits one declared infrastructure resume per interrupted cell, restoring model, optimizer, sampler, and runtime state. Preserve replay accounting and failed rows. Do not stop healthy admitted work at a chat boundary or from observed scores. Completion requires all 21 valid training results plus valid base evaluation; merely exiting the controller or finalizing a subset is not completion.
