# Experiment 05 version 2: length-matched supervised fine-tuning screen results

Verified training cells: 21/21; statuses: {'complete': 21}; base: complete.

Intervals are descriptive 95% Student-t intervals across training seeds, conditional on the fixed evaluation examples; intervals are omitted below three seeds. Three seeds provide weak power and do not quantify uncertainty over new tasks or corpora.
P-values are unadjusted exploratory exact two-sided sign tests: null P(method accuracy > baseline accuracy)=0.5 among non-tied seeds. At three non-tied pairs, the minimum p-value is 0.25. No confirmatory significance or superiority verdict is made.

Positive accuracy differences favor the method named first. Differences are percentage points (pp). Intervals are not clipped to feasible accuracy bounds.

Version 2 replaces short-question compression scores with fixed-byte, within-source packed scoring views. The wrong-target control uses reserved CommonsenseQA training calibration data excluded from the candidate pool; it tests whether the target choice changes the mixture's effect under matched view lengths. Matching lengths reduces the short-string confound but does not itself establish semantic alignment.

Any method with empty selection support is retained as unavailable, with null scores across its seeds. Availability is a preprocessing result, not measured evidence against the method's learning hypothesis; the full declared matrix remains incomplete.

| Method | Verified seeds | SciQ accuracy % [95% interval] | Across-seed standard deviation pp | Retention accuracy % [95% interval] |
|---|---:|---|---:|---|
| sample_proportional | 3/3 | 75.000 [69.038, 80.962] | 2.400 | 54.333 [47.479, 61.188] |
| uniform_domain | 3/3 | 74.800 [70.469, 79.131] | 1.744 | 56.733 [51.732, 61.735] |
| compel_filter | 3/3 | 76.533 [75.283, 77.784] | 0.503 | 53.600 [51.879, 55.321] |
| zipmix_static | 3/3 | 77.200 [73.320, 81.080] | 1.562 | 54.933 [52.925, 56.941] |
| direct_zipfit | 3/3 | 77.933 [77.646, 78.220] | 0.115 | 46.000 [41.261, 50.739] |
| shuffled_zipmix | 3/3 | 75.333 [70.638, 80.029] | 1.890 | 53.667 [50.756, 56.578] |
| wrongtarget_zipmix | 3/3 | 74.067 [70.759, 77.375] | 1.332 | 56.267 [51.001, 61.532] |

Arm-level means: p-val=n/a (no test against an arbitrary accuracy level).

| ZipMix minus baseline | Paired seeds | SciQ difference pp [95% interval] | Target p-value | Retention difference pp [95% interval] | Retention p-value |
|---|---:|---|---:|---|---:|
| sample_proportional | 3/3 | 2.200 [-4.702, 9.102] | 0.25 | 0.600 [-8.232, 9.432] | 1 |
| uniform_domain | 3/3 | 2.400 [-4.321, 9.121] | 1 | -1.800 [-6.216, 2.616] | 0.25 |
| compel_filter | 3/3 | 0.667 [-3.617, 4.950] | 1 | 1.333 [0.574, 2.092] | 0.25 |
| direct_zipfit | 3/3 | -0.733 [-4.749, 3.282] | 1 | 8.933 [4.159, 13.707] | 0.25 |
| shuffled_zipmix | 3/3 | 1.867 [0.432, 3.301] | 0.25 | 1.267 [-3.241, 5.775] | 1 |
| wrongtarget_zipmix | 3/3 | 3.133 [1.699, 4.568] | 0.25 | -1.333 [-4.715, 2.049] | 1 |

Base checkpoint: one deterministic evaluation, reported once; no training-seed interval or p-value applies.
- retention: 285/500 = 57.000%; interval=n/a; p-val=n/a.
- target: 404/500 = 80.800%; interval=n/a; p-val=n/a.

| Frozen cell | Ledger status | Analysis status | SciQ correct / 500 | Retention correct / 500 |
|---|---|---|---:|---:|
| sample_proportional__seed0 | complete | complete | 375 | 277 |
| sample_proportional__seed1 | complete | complete | 363 | 256 |
| sample_proportional__seed2 | complete | complete | 387 | 282 |
| uniform_domain__seed0 | complete | complete | 378 | 293 |
| uniform_domain__seed1 | complete | complete | 364 | 285 |
| uniform_domain__seed2 | complete | complete | 380 | 273 |
| compel_filter__seed0 | complete | complete | 383 | 266 |
| compel_filter__seed1 | complete | complete | 385 | 272 |
| compel_filter__seed2 | complete | complete | 380 | 266 |
| zipmix_static__seed0 | complete | complete | 377 | 274 |
| zipmix_static__seed1 | complete | complete | 390 | 279 |
| zipmix_static__seed2 | complete | complete | 391 | 271 |
| direct_zipfit__seed0 | complete | complete | 390 | 219 |
| direct_zipfit__seed1 | complete | complete | 389 | 236 |
| direct_zipfit__seed2 | complete | complete | 390 | 235 |
| shuffled_zipmix__seed0 | complete | complete | 366 | 264 |
| shuffled_zipmix__seed1 | complete | complete | 384 | 266 |
| shuffled_zipmix__seed2 | complete | complete | 380 | 275 |
| wrongtarget_zipmix__seed0 | complete | complete | 363 | 283 |
| wrongtarget_zipmix__seed1 | complete | complete | 376 | 291 |
| wrongtarget_zipmix__seed2 | complete | complete | 372 | 270 |

CommonsenseQA training examples are in the candidate pool. Its validation result measures untargeted-task retention/transfer, not an unseen training family. The pretrained backbone may already have benchmark exposure.

Equal supervision is 2,048 answer-label tokens per completed cell. Input-token counts, training time and recovery overhead are recorded separately and need not match. Compel's long-document compression band is applied to fixed-byte packs of short questions; each member question inherits its pack's selection status. This screen does not compare against DoGE, DoReMi, LESS or newer mixing methods.

Original infrastructure interruptions remain in per-cell recovery receipts. Successful resumption is not evidence of an uninterrupted run.

**TLDR-end:** [zip-mix: supervised fine-tuning screen] All frozen training cells and the base evaluation have verified artifacts; competitive superiority remains unresolved.

**Snapshot:**
```text
verified_training_cells=21 expected_training_cells=21
status_counts={'complete': 21}
base_status=complete
```
