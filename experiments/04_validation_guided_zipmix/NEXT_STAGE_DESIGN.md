# Next-stage decision: test alignment within fixed source and compression strata

**Doc link:** <https://github.com/brando90/zip-mix/blob/main/experiments/04_validation_guided_zipmix/NEXT_STAGE_DESIGN.md>

**TLDR:** Bin-only weights cannot select better examples within a bin. Test alignment within fixed source×compression groups using fresh held-out questions; compare against direct scoring later to establish whether bins add value.

10-04-2026. Design only; no execution is authorized by this note. The original proposal used completed Experiment 04 evidence and frozen training/development diagnostics before Experiment 05 or Experiment 06 outcomes were read. The dated execution-order update below uses completed Experiment 06 results; Experiment 05 outcomes remain unread. Existing runs and sources remain unchanged.

**Recommendation:** test whether target alignment ranks useful examples *within* fixed source×compression strata. Match the full joint source/bin distribution across methods. Increasing the number of independently weighted strata alone would relax the exposure constraint, but would not identify useful within-stratum selection.

## Execution-order update: 10-04-2026

The [completed Experiment 06](../06_contrastive_validation_zipmix/results.md) produced nine fine-tuned models, all below the unchanged base on both target and retention accuracy: the base scored 404/500 = 80.8% and 285/500 = 57.0%, respectively. Contrastive minus source-matched target accuracy was −1.733 percentage points [95% interval −10.625, 7.159], raw/Holm p-val=1.0 (paired-seed Student-t interval; two-sided exact sign test of equal probabilities of higher accuracy among non-ties). These observations warrant training calibration before another selector is admitted; they do not identify the cause of the regression or establish contamination.

First allocate a new training/validation tuning partition, separate from all reused benchmark test sets, and verify fine-tuning calibration against an unchanged-base control using only that validation partition. Predeclare its tuning allowance and acceptance criteria; choose no learning rate or other training setting from Experiment 05/06 test outcomes. The transform, nine-cell recipe and advance thresholds below remain a proposal, not authorization to repeat the setup despite its observed benchmark regression. Any training change supported by the new calibration must be frozen as a separate prospective version before selector admission, with fresh held-out evaluation and no silent changes to existing experiments.

## What the current evidence establishes

The [complete 27-cell pretraining screen](expt_v1/results/measured/analysis.md) gives ZipMix minus Domain Reweighting with Generalization Estimation (DoGE) target negative log-likelihood (NLL) difference **+0.41265 [0.39285, 0.43246] nats/token, p-val=0.25**. Against population sampling, the difference is **−0.01494 [−0.06109, 0.03121], p-val=0.25**. Intervals are descriptive paired-seed Student-t intervals; p-values are two-sided exact sign tests of equal probabilities of either direction among non-ties, not tests of effect magnitude. There are three paired seeds and additional proxy costs for DoGE. These results do not establish superiority or explain its causes.

The [finite-pool diagnosis](expt_v1/results/bin_capacity.json) establishes a separate fact: a prior uniform inside these compression bins can assign at most **23.5496%** of training tokens to PubMed, whereas DoGE's final priors assign **38.9863–40.6287%**. Static ZipMix assigns **15.2032%**, close to the population's **14.8289%**. These are exact finite-pool probabilities, with no confidence interval or hypothesis test applicable. The capacity restriction is real; its causal contribution to the loss gap is unmeasured.

## Three different model classes

Let training unit $i$ have source $s(i)$, compression bin $b(i)$, and fixed base probability $q_i>0$. Use a token block for pretraining or one supervised answer for this fine-tuning pool. Let $q_b=\sum_{i:b(i)=b}q_i$, and define conditional probabilities analogously. Empty bins or source/bin intersections are excluded.

| Family | Sampling probability | What can change |
|---|---|---|
| Bin weights only | $p_i=\pi_{b(i)}q_i/q_{b(i)}$ | Bin mass; conditional examples and sources inside a bin stay fixed |
| Within-bin alignment | $p_i=\pi_{b(i)}q_iw_i/\sum_{j:b(j)=b(i)}q_jw_j$ | Example ranking inside bins; source exposure can also change |
| Source×bin weights only | $p_i=M_{s(i),b(i)}q_i/q_{s(i),b(i)}$ | Joint source/bin mass; examples inside each intersection stay fixed |

Here π and M are nonnegative masses summing to one; w is a nonnegative score transform with positive denominators. For bin-only sampling, the attainable source vector is exactly the convex hull of the nonempty-bin vectors $q(s\mid b)$. For any event A defined on training units,

$$
p(A)=\sum_b\pi_b q(A\mid b)\leq\max_b q(A\mid b).
$$

The bound is attained by selecting a maximizing bin. Positive mandatory weight floors may lower it. More generally, the expectation of any *fixed per-unit utility* is a convex combination of its bin means. Trained-model generalization is a nonlinear function of the whole training distribution, so this does **not** upper-bound achievable accuracy or lower-bound trained loss.

A bin-only sum-score prior retains only each bin's total score. Permuting scores among equal-weight sampling units inside a bin leaves the prior exactly unchanged. It therefore cannot exploit which examples have high alignment inside that bin. For the current equal-unit base, constant score means reduce the method to the bin-population prior, apart from smoothing. Even without coarsening, a static compression score does not observe model state or update interactions; these formulas do not establish an optimal training policy.

Free source×bin masses can realize any source marginal supported by observed sources, but cannot create missing intersections or distinguish two examples in the same intersection. Within-bin weights can escape the original source-exposure ceiling: at fixed π, unrestricted per-unit weights attain up to $\sum_b\pi_b\mathbf{1}[q(A\mid b)>0]$. With a within-bin maximum-to-minimum weight ratio bounded by K, each conditional event probability is at most $Kq(A\mid b)/(1-q(A\mid b)+Kq(A\mid b))$. These are representational capacities, not predictions that a compression score attains them. Pack-inherited weights impose a further restriction: examples in the same pack retain their base relative probabilities.

For identification, fix **every joint mass** M and change only

$$
p_i=M_{s(i),b(i)}\frac{q_i w_i}{\sum_{j:s(j)=s(i),\ b(j)=b(i)}q_jw_j}.
$$

This preserves both source and compression exposure **and their interaction**. Matching only the two separate marginals would not preserve that interaction. Experiment 06's source-matched control is useful, but can still differ in its joint source/bin table. The proposed test addresses that remaining distinction. A gain would support useful alignment ranking **within these groups**; it would not show that compression bins improve over a direct scorer. Binning's added value needs a later fair direct-score comparison with the same data, target information, concentration and resource allowance.

## One prospective experiment

Use the existing [Experiment 05 version 2 pool](../05_validation_guided_sft/expt_v2/data_manifest.json), its 351 equal-byte packs, 9,797 questions and fixed compression bins. These are Compel-inspired compression strata; the quantile bins are not a reproduction of Compel's published hard filter. No new data acquisition or scoring calls are needed for this diagnostic. Fix M to the population source×bin frequencies and q to uniform question sampling, so all methods retain **30.5298% SciQ exposure**.

For each pack, use the **signed** contrast $r=s^{true,max}-s^{wrong,max}$. Positive clipping made all three CommonsenseQA strata constant in the existing pool; signed ranks preserve relative information while fixed M prevents a changed source mix. This is a new named score transform, not a faithful ZIP-FIT reproduction. Within each source×bin stratum of m packs, let R be the average rank, with ties averaged, and set $z=(R-0.5)/m-0.5$, $w=\exp(\log(4)z)$. Inherit w to questions and normalize with the fixed-M formula above. This fixes the tilt in advance, caps the within-stratum weight ratio below four, and keeps every example supported. Constant or singleton strata get z=0 and remain uniform.

Freeze **three methods × three paired seeds**:

1. Conditional signed-contrast ranks, as above.
2. Uniform sampling within each source×bin stratum, with the same M.
3. One pack-rank permutation within each source×bin stratum, then identical normalization. Use a fixed seed 1707, visiting strata in sorted source/bin order and packs in their frozen order. Reuse the same permutation for all training seeds.

Pack permutation preserves the pack-rank multiset, not the inherited question-weight multiset: packs have unequal question counts. Report effective sample size, pack density, repetition and input-length diagnostics for both selectors. An improvement identifies useful within-stratum ranking, not semantic understanding; length, redundancy and latent difficulty may mediate it.

Retain the pinned Qwen2.5-0.5B model, training pool, seeds 0/1/2, 128 steps × 16 one-token answers, optimizer, precision and common checkpoint cadence 64 from the prepared follow-up. Before any admission, freeze fresh, previously uninspected target and retention questions, with leakage checks against all training/scoring inputs; unused official-split records are a candidate source, not yet an audited artifact. Propose 500 eligible questions per set; if unavailable, stop this proposal until a separately specified holdout exists. Do not use Experiment 05/06's held-out questions to select or tune this formulation or to pass its advance gate. Complete current results may inform whether to spend resources, but not select the score transform, tilt, comparator family or thresholds. This remains **nine new cells plus base**, with a new identity and resource admission; proposed ceiling two device-hours, subject to measured full-matrix feasibility. It is a mechanism screen, not a competitive benchmark suite. No launch follows from this note.

**Training/development diagnostics show a nondegenerate, mild intervention.** There are 13 nonempty source/bin strata, all with nonconstant signed scores and at least two packs. Exact effective sample sizes $1/\sum_i p_i^2$ are 8,488.46 and 8,480.91 for aligned and shuffled selectors, versus 9,797 uniform. Total variation distances are 0.168522 from uniform and 0.224942 from the shuffled selector. Expected input lengths are 48.5235, 48.5847 and 48.5909 tokens for aligned, uniform and shuffled respectively. The maximum observed joint-mass discrepancy is $5.56\times10^{-17}$. These calculations read no held-out arrays or outcome receipts and are not performance estimates.

Reproduce these numbers with [next_stage_diagnostics.py](next_stage_diagnostics.py); run `python3 experiments/04_validation_guided_zipmix/next_stage_diagnostics.py` from the repository root, using the existing private data snapshot. The [numeric receipt](expt_v1/results/next_stage_diagnostics.json) records all three input hashes, the exact score/permutation recipe, all 13 strata and their counts, 111 frozen-pool probability checks, and 1,400 scalar probability identities across 100 synthetic draws. It exports aggregate diagnostics only; no raw arrays, model calls or held-out results are involved.

## Stop and advance rules

- **Before admission:** verify every joint mass within 1e-12, all positive probabilities, effective sample size at least 80% of the pool, and total variation at least 0.05 from both controls. Current training-only diagnostics satisfy these proposed engineering gates. If reconstruction fails, do not train; investigate the discrepancy without selecting a new transform from benchmark outcomes.
- **Run the full manifest:** no early stopping, outcome-driven retuning or missing-cell deletion. Infrastructure incompletion is an incomplete experiment, not a negative scientific result.
- **Advance this formulation to a larger, independently held-out confirmation only if** its mean target-accuracy gain is at least 2 absolute percentage points against each control, every paired seed has a positive target difference against each, and mean retention regression is no worse than 1 percentage point against either. These are prospective resource-allocation heuristics, not significance thresholds or evidence that the true effect exceeds those margins. Report all paired intervals and raw/Holm-adjusted p-values across the two target comparisons; three pairs cannot give a two-sided sign-test p-value below 0.25.
- **Otherwise stop scaling this exact formulation.** Preserve any weak or mixed signal; do not declare compression ineffective in general or retune on these test outcomes. If actual processed or padded input-token costs differ by more than 5%, withhold an equal-compute claim and require a separately frozen budget-matched confirmation. Shared public benchmark families, possible pretrained exposure and three seeds still make this exploratory even with fresh questions and a passed advance heuristic.

## Literature boundary

[Compel](https://openreview.net/pdf?id=KFafeqE5fe) motivates compression stratification; [ZIP-FIT](https://arxiv.org/html/2410.18194v2#S2) is the direct target-alignment predecessor and uses mean aggregation, unlike the inherited maximum scores retained here for continuity. [DoGE](https://arxiv.org/html/2310.15393v2#S2) motivates the target-aware gradient comparison; [DoReMi](https://arxiv.org/abs/2305.10429) retains its distinct reference/excess-loss robust objective. This conditional-selection test does not replace faithful comparisons. If it earns advancement, target-aware [Data Selection via Importance Resampling](https://arxiv.org/abs/2302.03169), [Low-rank gradiEnt Similarity Search](https://github.com/princeton-nlp/LESS), direct ZIP-FIT and the applicable current mixture baselines in the [verified research design](research_design.md), including [Olmix](https://github.com/allenai/olmix) for pretraining, remain necessary before making competitive claims.

## Completion evidence, 10-04-2026 after Experiment 05

The full parent study subsequently completed. Its original Zip-Mix selector gained +1.867 [0.432, 3.301] percentage points over shuffled alignment and +3.133 [1.699, 4.568] over the wrong development target on science accuracy. Both are exploratory exact two-sided sign-test p-val=0.25 against equal winning probability among non-tied seeds; intervals are descriptive 95% Student-t intervals over three paired seeds on the fixed 500 questions. Every trained science score still trailed the unchanged base. These results preserve the rationale for testing validation targeting while retaining training calibration as the next prerequisite. They do not alter the prospective transform, sampling recipe, thresholds or fresh-evaluation requirement above, and no new study was launched. See the [complete results](../05_validation_guided_sft/results.md).
