# Experiment 06: prospective contrastive supervised fine-tuning screen results

Verified training cells: 9/9; statuses: {'complete': 9}; base: complete.

Intervals are descriptive 95% Student-t intervals across training seeds, conditional on the fixed evaluation examples; intervals are omitted below three seeds. Three seeds provide weak power and do not quantify uncertainty over new tasks or corpora.
Raw p-values are exploratory exact two-sided sign tests: null P(method accuracy > baseline accuracy)=0.5 among non-tied seeds. At three non-tied pairs, the minimum p-value is 0.25. The two planned target contrasts also receive Holm adjustment as a fixed family of two. Retention tests remain unadjusted diagnostics. No confirmatory significance or superiority verdict is made.

Positive accuracy differences favor the method named first. Differences are percentage points (pp). Intervals are not clipped to feasible accuracy bounds.

All methods reuse the fixed Experiment 05 version 2 candidate pool and equal-byte pack scores. Contrastive ZipMix uses positive target-minus-wrong-target alignment; source-matched and shuffled controls test whether its selection adds value beyond source composition and score ordering. This is a prospective mechanism test, not evidence of semantic alignment by construction.

Any method with empty selection support is retained as unavailable, with null scores across its seeds. Availability is a preprocessing result, not measured evidence against the method's learning hypothesis; the full declared matrix remains incomplete.

| Method | Verified seeds | SciQ accuracy % [95% interval] | Across-seed standard deviation pp | Retention accuracy % [95% interval] |
|---|---:|---|---:|---|
| contrastive_zipmix | 3/3 | 75.333 [71.348, 79.318] | 1.604 | 49.400 [40.457, 58.343] |
| source_matched | 3/3 | 77.067 [72.090, 82.043] | 2.003 | 53.133 [45.086, 61.180] |
| contrastive_shuffled | 3/3 | 75.267 [74.980, 75.554] | 0.115 | 51.000 [47.025, 54.975] |

Arm-level means: p-val=n/a (no test against an arbitrary accuracy level).

| Contrastive ZipMix minus control | Paired seeds | SciQ difference pp [95% interval] | Target raw p-value | Target Holm p-value | Retention difference pp [95% interval] | Retention raw p-value |
|---|---:|---|---:|---:|---|---:|
| source_matched | 3/3 | -1.733 [-10.625, 7.159] | 1 | 1 | -3.733 [-15.389, 7.922] | 0.5 |
| contrastive_shuffled | 3/3 | 0.067 [-4.159, 4.292] | 1 | 1 | -1.600 [-9.361, 6.161] | 1 |

Base checkpoint: one deterministic evaluation, reported once; no training-seed interval or p-value applies.
- retention: 285/500 = 57.000%; interval=n/a; p-val=n/a.
- target: 404/500 = 80.800%; interval=n/a; p-val=n/a.

| Frozen cell | Ledger status | Analysis status | SciQ correct / 500 | Retention correct / 500 |
|---|---|---|---:|---:|
| contrastive_zipmix__seed0 | complete | complete | 385 | 247 |
| contrastive_zipmix__seed1 | complete | complete | 369 | 229 |
| contrastive_zipmix__seed2 | complete | complete | 376 | 265 |
| source_matched__seed0 | complete | complete | 374 | 247 |
| source_matched__seed1 | complete | complete | 393 | 274 |
| source_matched__seed2 | complete | complete | 389 | 276 |
| contrastive_shuffled__seed0 | complete | complete | 376 | 263 |
| contrastive_shuffled__seed1 | complete | complete | 377 | 247 |
| contrastive_shuffled__seed2 | complete | complete | 376 | 255 |

CommonsenseQA training examples are in the candidate pool. Its validation result measures untargeted-task retention/transfer, not an unseen training family. The pretrained backbone may already have benchmark exposure.

Equal supervision is 2,048 answer-label tokens per completed cell. Input-token counts, training time and recovery overhead are recorded separately and need not match. The source-matched control matches contrastive ZipMix's total probability for each source and samples uniformly within that source. This screen does not compare against DoGE, DoReMi, LESS or newer mixing methods.

Original infrastructure interruptions remain in per-cell recovery receipts. Successful resumption is not evidence of an uninterrupted run.

**TLDR-end:** [zip-mix: contrastive fine-tuning screen] All frozen training cells and the base evaluation have verified artifacts; competitive superiority remains unresolved.

**Snapshot:**
```text
verified_training_cells=9 expected_training_cells=9
status_counts={'complete': 9}
base_status=complete
```
