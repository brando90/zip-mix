# Finite-pool bin capacity and prospective contrastive selector

**Doc link:** <https://github.com/brando90/zip-mix/blob/main/experiments/04_validation_guided_zipmix/expt_v1/results/bin_capacity.md>

10-04-2026. These are exact counts and probabilities for the frozen training pools, not estimates of benchmark performance. Confidence intervals and p-values are therefore not applicable. The analysis uses Experiment04 training metadata and completed mixture receipts, plus Experiment05 version 2 training/development selector arrays. No Experiment05 benchmark metrics, target arrays, or retention arrays were read. Full values, input hashes, and the prospective timestamp are in [bin_capacity.json](bin_capacity.json).

For a nonempty bin (b), let (q_b=n_{target,b}/n_b). Any nonnegative bin weights (w_b) summing to one, with uniform sampling inside each bin, give target exposure (∑_b w_b q_b ≤  max_b q_b). Putting all mass on a maximizing bin attains this bound. Required positive weight floors would lower the attainable maximum. This is a source-exposure bound, not a bound on test loss or generalization. Domain selectors and selectors that vary sampling within a bin can exceed it.

**The pretraining bins cannot reproduce DoGE's target exposure.** PubMed comprises at most 23.5496% of any fixed compression bin. Static ZipMix puts 15.2032% of its mass on PubMed, versus 14.8289% for the population prior. The learned DoGE final-stage priors put 39.9618%, 38.9863%, and 40.6287% on PubMed for seeds 0, 1, and 2. Thus all three lie outside the feasible set of uniform-within-bin priors. This identifies a structural restriction; it does not establish how much of the observed performance difference the restriction causes.

| Experiment04 bin | Training blocks | PubMed blocks | PubMed fraction |
|---|---:|---:|---:|
| 0 | 11,795 | 182 | 1.5430% |
| 1 | 18,069 | 2,820 | 15.6068% |
| 2 | 27,440 | 6,462 | 23.5496% |
| 3 | 25,366 | 3,214 | 12.6705% |
| 4 | 16,340 | 2,292 | 14.0269% |
| 5 | 4,029 | 322 | 7.9921% |
| 6 | 84 | 0 | 0% |

Each block has 128 loss tokens, so these probabilities also describe expected training-token exposure. Fixed compression-ratio boundaries are 0.50, 0.60, 0.67, 0.73, 0.80, and 0.90, with the final bin open-ended.

**The supervised pool permits much stronger SciQ exposure through its existing bins.** Its maximum is 87.5121%, compared with 35.8677% for the true-target ZipMix prior, 30.5298% for the population prior, and 26.9331% for the wrong-target prior. The existing direct selector reaches 100% SciQ because it selects individual examples and is not subject to this bound.

| Experiment05 bin | Training questions | SciQ questions | SciQ fraction |
|---|---:|---:|---:|
| 0 | 1,230 | 239 | 19.4309% |
| 1 | 2,066 | 1,808 | 87.5121% |
| 2 | 2,170 | 882 | 40.6452% |
| 3 | 2,085 | 62 | 2.9736% |
| 4 | 2,246 | 0 | 0% |

Here the sampling unit is one question with one supervised answer token. Compression scores are inherited from equal-width 4,096-byte packs; bins were fixed using pack compression-ratio quantiles. These bounds concern answer-token exposure. Input lengths, and therefore compute, vary.

The prospective candidate assigns each question its pack's positive contrast (a_i=max(s_i^{true}-s_i^{wrong},0)), sums those values within each existing bin, normalizes the bin masses, and samples uniformly within each bin. Clipping happens before aggregation. An all-zero contrast would make the method unavailable rather than trigger a substituted prior. The frozen pool has 5,512 positive-contrast questions, so this case does not occur.

**The contrastive candidate is distinct from all existing priors.** It assigns source masses 19.9082% ARC-Challenge, 6.3940% CommonsenseQA, 15.3769% OpenBookQA, and 58.3210% SciQ. Its bin masses are 21.3873%, 50.7324%, 23.8239%, 2.8592%, and 1.1973%. Total variation distance on the full question pool is 0.325421 from true-target ZipMix, 0.464953 from wrong-target ZipMix, 0.390508 from shuffled ZipMix, and 0.498702 from the direct selector. It is not a duplicate control. Nevertheless, uniform sampling within bins assigns 13.9321% of its mass to questions whose own inherited contrast is nonpositive.

A separately frozen follow-up would be informative if it compares the candidate with a prior that matches its four source masses exactly and samples uniformly within each source. Their total variation distance is 0.213562, while their source exposure is identical. This comparison separates source reweighting from the additional selection within each source. A shuffled contrast-score control would additionally test whether attaching the contrast scores to their original packs matters. Existing true-target/wrong-target comparisons alone do not isolate these effects.

Keep the existing model, pool, seeds 0/1/2, and 128 steps × 16 questions = 2,048 supervised tokens, with a distinct prospective identity and no outcome-based tuning. The candidate and source-matched control imply 49.8275 and 49.7203 input tokens per example respectively; actual processed tokens and time must still be recorded. Any follow-up remains descriptive at three seeds and must retain source/redundancy, pack-density, overflow-member, wrong-target choice, and pretrained contamination caveats. These development-only diagnostics justify testing the contrastive mechanism; they do not establish that it improves benchmark accuracy.
