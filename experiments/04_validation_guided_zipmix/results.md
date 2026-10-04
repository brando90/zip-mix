# Experiment 04: complete validation-guided pretraining results

**Doc link:** <https://github.com/brando90/zip-mix/blob/main/experiments/04_validation_guided_zipmix/results.md>

Experiment 04: nine methods × three seeds, 5.29-million-parameter models trained from scratch; Zip-Mix shows small target-loss gains over sampling controls, but trails the compact DoGE baseline and misses the prespecified screening threshold.

Updated 10-04-2026 12:28 PDT. **27/27 training cells completed, zero failures, zero recoveries.** The full frozen matrix and saved per-example losses were verified independently. Final predicted-token budget: 8,388,608 per model, 226,492,416 over the 27 final models. Reference and proxy work is additional. The target comprises held-out biomedical documents; broad evaluation covers eight other corpus domains. This is a compact language-model screen, not a benchmark-accuracy or state-of-the-art reproduction.

## Measured result

Negative log likelihood (NLL) is measured in nats per predicted token; lower is better. Zip-Mix target NLL is **5.16156 [5.14695, 5.17617]**, across-seed standard deviation 0.00588, p-val=n/a (no arbitrary absolute-level test). Its perplexity is **174.44 [171.91, 177.00]**, computed by exponentiating the mean NLL and interval endpoints, p-val=n/a.

| Comparator | Zip-Mix minus comparator target NLL [95% interval] | Exact sign-test p-value | Paired seed directions |
|---|---:|---:|---|
| Population sampling | −0.01494 [−0.06109, +0.03121] | 0.25 | Lower Zip-Mix loss in 3/3 |
| Shuffled Zip-Mix | −0.01630 [−0.02746, −0.00515] | 0.25 | Lower Zip-Mix loss in 3/3 |
| Domain Reweighting with Minimax Optimization (DoReMi), compact adaptation | −0.09693 [−0.15909, −0.03477] | 0.25 | Lower Zip-Mix loss in 3/3 |
| Domain Reweighting with Generalization Estimation (DoGE), compact adaptation | +0.41265 [+0.39285, +0.43246] | 0.25 | Higher Zip-Mix loss in 3/3 |
| Uniform source domains | +0.10031 [+0.06897, +0.13165] | 0.25 | Higher Zip-Mix loss in 3/3 |
| Direct mean-score sampling | +0.07691 [+0.04438, +0.10945] | 0.25 | Higher Zip-Mix loss in 3/3 |

Intervals are descriptive 95% paired-seed Student-t intervals, conditional on this fixed corpus and held-out sample. P-values use the unadjusted exact two-sided sign test with null P(Zip-Mix loss < comparator loss)=0.5 among non-tied seeds. Three pairs cannot produce p<0.25; a Student-t interval excluding zero is not the same test as the sign test. No confirmatory superiority claim follows. [Every method, seed, cost and broad-domain comparison](expt_v1/results/measured/analysis.md) is retained.

The **prespecified descriptive screening heuristic was not met**: mean target improvement had to reach 0.02 against both population and shuffled controls. Observed improvements were 0.01494 and 0.01630 respectively. Directions agreed in all three seeds, and broad-loss tolerances were met. Broad Zip-Mix minus population NLL was +0.00715 [−0.03202, +0.04633], p-val=1.0 under the same sign null; broad minus shuffled was −0.00173 [−0.03687, +0.03341], p-val=1.0. The threshold rule itself is descriptive, p-val=n/a (no inferential test against 0.02). [Machine-readable criterion components](expt_v1/results/measured/screening_decision.json).

![Paired effects](expt_v1/results/measured/paired_effects.png)

**Zip-Mix's modest gains over population and shuffled sampling do not establish superiority over the competitive controls.** Points are paired seed differences and bars are 95% Student-t intervals; negative favors Zip-Mix. DoGE's compact implementation has the lowest measured target loss; uniformly sampling source domains has the lowest measured broad-domain loss. These observations remain specific to the frozen small-model setup.

## What this changes

This result supports a limited hypothesis: development-conditioned compression scores may contain useful information compared with their permutation control. It does not show that compression buckets are the best way to use that information, or that the original Zip-Mix rule generalizes much better than DoGE and other strong methods.

The original prior moves only 2.07% of finite-pool probability mass from population sampling. It raises biomedical sampling from 14.8289% to 15.2032%; the three learned DoGE mixtures allocate 39.9618%, 38.9863%, and 40.6287%. These are exact properties of the saved mixtures, p-val=n/a, not estimated performance effects. An exact finite-pool calculation shows that these bins can assign at most 23.5496% to biomedical blocks, so none can reproduce the learned DoGE source exposures. [The bound, proof and per-bin counts](expt_v1/results/bin_capacity.md) are recorded. They do not prove that exposure alone causes the loss differences. Direct selection also changes score aggregation, so its comparison cannot isolate bucketing by itself.

Keep the failed screening gate; do not lower it after seeing these results or scale the unchanged method simply because all jobs finished. The independently frozen post-training comparison continues. Any subsequent algorithm change must have a new prospective identity, explicit controls, and honest accounting for these observed tests.

## Cost and verification

Recorded sums over the completed cells: 1,309.45 seconds in training stages and 1,321.16 seconds across cell invocations, excluding shared preparation and controller overhead. Equal final training tokens do not equal total compute: DoReMi and DoGE pay their recorded reference/proxy cost. Dollar cost is unavailable. [Loss/perplexity and per-method recorded times](expt_v1/results/measured/perplexity_and_screening.md).

The requested single Claude Opus 5.5 maximum-effort review applied implementation fixes before launch; 23 remote pretraining tests passed. All exported metric and loss-array hashes were verified. Reanalysis recomputed every saved loss aggregate; all point estimates, standard deviations, paired differences, p-values and counts matched exactly. Cross-version confidence-interval endpoints differed by at most 8.90×10⁻¹³, below the existing 1e-10 verification tolerance; original remote analyses are preserved. [Export receipt](expt_v1/results/measured/EXPORT.md) · [review](qa/opus55_max_review.md) · [frozen protocol](expt_v1/PROTOCOL.md) · [research plan](research_design.md).

**TLDR-end:** [zip-mix: pretraining] All 27 runs completed cleanly. Zip-Mix improves slightly over the sampling controls but does not meet the prespecified gain threshold and trails the compact DoGE, uniform-domain, and direct-score comparators; broad superiority remains unproven.

**Snapshot:**
```text
complete=27/27; failed=0; recovered=0
Zip-Mix target NLL=5.16156
population target NLL=5.17650
DoReMi target NLL=5.25849
DoGE target NLL=4.74891
screening_heuristic_met=false
```
