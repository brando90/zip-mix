# Compact pretraining mixture comparison

Completed 27/27 frozen cells; recovered cells: 0; full clean matrix: True.

Final training budgets match. Learned-mixture reference/proxy costs are additional, so total compute is not matched.
Target and broad evaluation were held out from all scoring, training, mixture updates and stopping decisions.

Negative log-likelihood (NLL) is measured in nats per next-token prediction; lower is better.
Intervals are descriptive 95% paired-seed Student-t intervals (at least 3 seeds), with unadjusted two-sided exact sign tests.
At 3 non-tied seed pairs the smallest possible two-sided sign-test p-value is 0.25; this run cannot establish p<0.05 superiority.
Seed uncertainty is conditional on the fixed corpus and held-out examples; it does not quantify corpus sampling uncertainty.

| Arm | Complete seeds | Target NLL [95% interval] | Across-seed standard deviation | Broad NLL [95% interval] |
|---|---:|---|---:|---|
| token_proportional | 3/3 | 5.17650 [5.14115, 5.21185] | 0.01423 | 4.81850 [4.80442, 4.83258] |
| uniform_domain | 3/3 | 5.06125 [5.03897, 5.08354] | 0.00897 | 4.68217 [4.67579, 4.68856] |
| uniform_bucket | 3/3 | 5.34616 [5.32792, 5.36440] | 0.00734 | 4.92528 [4.89881, 4.95176] |
| compel_filter | 3/3 | 5.23661 [5.20056, 5.27267] | 0.01451 | 5.18312 [5.15455, 5.21168] |
| zipmix_static | 3/3 | 5.16156 [5.14695, 5.17617] | 0.00588 | 4.82565 [4.80012, 4.85119] |
| direct_zipfit | 3/3 | 5.08465 [5.06547, 5.10382] | 0.00772 | 4.90241 [4.89998, 4.90483] |
| shuffled_zipmix | 3/3 | 5.17787 [5.15365, 5.20208] | 0.00975 | 4.82738 [4.81404, 4.84072] |
| doremi | 3/3 | 5.25849 [5.18668, 5.33030] | 0.02891 | 4.81780 [4.81132, 4.82429] |
| doge | 3/3 | 4.74891 [4.72910, 4.76871] | 0.00797 | 4.78837 [4.77338, 4.80337] |

Arm-level intervals are descriptive: p-val=n/a (no arm mean tested against an arbitrary null).

| Comparison (ZipMix minus baseline) | Paired seeds | Target difference [95% interval] | p-value | Broad difference [95% interval] |
|---|---:|---|---:|---|
| token_proportional | 3/3 | -0.01494 [-0.06109, 0.03121] | 0.25 | 0.00715 [-0.03202, 0.04633] |
| uniform_domain | 3/3 | 0.10031 [0.06897, 0.13165] | 0.25 | 0.14348 [0.11751, 0.16945] |
| uniform_bucket | 3/3 | -0.18460 [-0.21047, -0.15872] | 0.25 | -0.09963 [-0.13025, -0.06901] |
| compel_filter | 3/3 | -0.07505 [-0.11376, -0.03635] | 0.25 | -0.35746 [-0.38695, -0.32797] |
| direct_zipfit | 3/3 | 0.07691 [0.04438, 0.10945] | 0.25 | -0.07675 [-0.09990, -0.05361] |
| shuffled_zipmix | 3/3 | -0.01630 [-0.02746, -0.00515] | 0.25 | -0.00173 [-0.03687, 0.03341] |
| doremi | 3/3 | -0.09693 [-0.15909, -0.03477] | 0.25 | 0.00785 [-0.02409, 0.03979] |
| doge | 3/3 | 0.41265 [0.39285, 0.43246] | 0.25 | 0.03728 [0.02672, 0.04784] |

P-values above test the named sign null P(ZipMix NLL < baseline NLL)=0.5 among non-ties; they do not test the magnitude of a mean difference.
No confirmatory superiority claim is made from these exploratory comparisons. Missing or failed cells are not dropped from the declared denominator.

| Cell | Status / recoveries | Reference seconds | Proxy seconds | Final seconds | Training floating-point-operation proxy |
|---|---|---:|---:|---:|---:|
| token_proportional, seed 0 | complete / 0 | n/a | n/a | 36.2 | 266227842809856 |
| uniform_domain, seed 0 | complete / 0 | n/a | n/a | 36.5 | 266227842809856 |
| uniform_bucket, seed 0 | complete / 0 | n/a | n/a | 35.6 | 266227842809856 |
| compel_filter, seed 0 | complete / 0 | n/a | n/a | 41.7 | 266227842809856 |
| zipmix_static, seed 0 | complete / 0 | n/a | n/a | 36.2 | 266227842809856 |
| direct_zipfit, seed 0 | complete / 0 | n/a | n/a | 41.8 | 266227842809856 |
| shuffled_zipmix, seed 0 | complete / 0 | n/a | n/a | 38.8 | 266227842809856 |
| doremi, seed 0 | complete / 0 | 9.4 | 9.5 | 37.0 | 388248937431040 |
| doge, seed 0 | complete / 0 | n/a | 57.8 | 36.8 | 341104423600128 |
| token_proportional, seed 1 | complete / 0 | n/a | n/a | 35.3 | 266227842809856 |
| uniform_domain, seed 1 | complete / 0 | n/a | n/a | 35.4 | 266227842809856 |
| uniform_bucket, seed 1 | complete / 0 | n/a | n/a | 38.7 | 266227842809856 |
| compel_filter, seed 1 | complete / 0 | n/a | n/a | 42.6 | 266227842809856 |
| zipmix_static, seed 1 | complete / 0 | n/a | n/a | 36.3 | 266227842809856 |
| direct_zipfit, seed 1 | complete / 0 | n/a | n/a | 36.0 | 266227842809856 |
| shuffled_zipmix, seed 1 | complete / 0 | n/a | n/a | 39.6 | 266227842809856 |
| doremi, seed 1 | complete / 0 | 13.1 | 9.8 | 43.3 | 388248937431040 |
| doge, seed 1 | complete / 0 | n/a | 54.8 | 40.7 | 341104423600128 |
| token_proportional, seed 2 | complete / 0 | n/a | n/a | 45.3 | 266227842809856 |
| uniform_domain, seed 2 | complete / 0 | n/a | n/a | 43.0 | 266227842809856 |
| uniform_bucket, seed 2 | complete / 0 | n/a | n/a | 44.0 | 266227842809856 |
| compel_filter, seed 2 | complete / 0 | n/a | n/a | 44.3 | 266227842809856 |
| zipmix_static, seed 2 | complete / 0 | n/a | n/a | 40.6 | 266227842809856 |
| direct_zipfit, seed 2 | complete / 0 | n/a | n/a | 40.1 | 266227842809856 |
| shuffled_zipmix, seed 2 | complete / 0 | n/a | n/a | 43.8 | 266227842809856 |
| doremi, seed 2 | complete / 0 | 12.0 | 14.1 | 43.5 | 388248937431040 |
| doge, seed 2 | complete / 0 | n/a | 54.6 | 41.3 | 341104423600128 |

The operation proxy is 6 × parameter count × gradient tokens + 2 × parameter count × forward-only tokens.
It excludes quadratic attention, optimizer, evaluation and preprocessing operations. Stage wall times include in-stage transfers and logging.
If a cell resumed after interruption, uncheckpointed repeated work is additional and not reconstructed by this cost proxy; failure receipts are retained.
All learned-mixture models use the same size as final models, with fresh initialization for the final stage; this is not a cross-scale transfer result.
direct_zipfit is continuous mean-compression-score sampling, a soft adaptation of ZIP-FIT, not its hard top-ranked selection.
DoGE uses all parameters, a pooled development target, and the official mean-gradient-norm normalization; proxy minibatches are stratified by domain.

**TLDR-end:** [zip-mix: compact pretraining] The frozen matrix completed; estimates remain exploratory and conditional on this compact setup.

**Snapshot:**
`complete=27 expected=27; analysis.json contains per-arm and paired-seed estimates.`
