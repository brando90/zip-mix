**A (TLDR):** As written, the design can't answer "does ZipMix fix DoReMi?" There are three main problems. First, the prior is probably close to token-proportional (TP) sampling. Second, the place where the prior enters DoReMi is never defined. Third, the statistics treat correlated cells as independent. I edited no files. "DRO" means Group Distributionally Robust Optimization, "CR" means compression ratio, and α⁽⁰⁾ is the alignment prior.

## 1. Critical flaws
1. **[critical] α⁽⁰⁾ mostly reflects bucket size.** It sums s(d) over each bucket, so big buckets win, and Exp 02 flags this itself (line 166). TP is the same sampling distribution whichever way you partition the data, so A7 ≈ TP and the A6 start ≈ TP. A "win" would then come from proportional weighting, not alignment. **Fix:** add a DRO arm that starts from TP weights. Also add a size-normalized prior (mean s(d) per bucket). Gate on α⁽⁰⁾ differing from bucket token share, with a bootstrap CI.
2. **[critical] "α⁽⁰⁾ init" is undefined.** DoReMi has three slots: reference-model weights, the DRO starting weights, and the smoothing target (uniform, c≈1e-3). The DRO updates and the smoothing pull the weights away from wherever they start. The runbook (prompt line 69) mixes up the reference weights and the starting weights. **Fix:** say which slots get α⁽⁰⁾, preferably reference weights plus smoothing target. Keep the reference model identical within each 2×2 comparison.
3. **[critical] H_max is never tested.** README line 23 calls H_max (min-max is too pessimistic) one of the two failure causes. But the 2×2 crosses prior × partition, and every DRO arm still uses min-max. **Fix:** add a non-min-max arm, e.g., mean excess loss or tempered DRO, on the same partition.
4. **[critical] The primary metric favors some arms.** A domain-uniform mean of Pile perplexity is roughly what A3/A4 optimize. It also penalizes TP on small domains. And it isn't what ZipMix targets: alignment to the validation suite. **Fix:** pre-register one primary metric, plus token-weighted perplexity and perplexity on held-out text similar to the validation suite.

## 2. Confounds and missing controls
5. **[major] "Matched compute" is false.** DRO arms need a reference run plus a proxy run. ZIP-FIT scoring of the full Pile is CPU-heavy: Exp 02 estimated 8–16 h for just 150K documents. **Fix:** report end-to-end compute per arm (AGENTS.md). Say whether α⁽⁰⁾ comes from a 50K-document sample.
6. **[major] DRO weights come from one run.** With one DRO run and 3 training seeds, the variance from weight estimation is missing. **Fix:** rerun the whole pipeline per seed.
7. **[major] Repetition.** Uniform weights over 7 buckets or ~17 domains repeat tiny groups (B1/B7, Enron) for many epochs at 3B tokens. **Fix:** log epochs per group and pre-set a cap.
8. **[major] H_part confounds partition with group count** (~17 vs 7). **Fix:** add a 7-group domain-merge or a bucket-count ablation.
9. **[minor] No uniform-over-buckets arm.** It's A5's starting point. Add it.

## 3. Factual claims
10. **[major] "DoReMi loses to TP" rests on one quote.** DoReMi's Pile baseline was the Pile's *default* (upsampled) weights, not TP, and it reported gains. **Fix:** cite published evidence (Aioli, DoGE, RegMix — check each) or label H0 anecdotal.
11. **[minor] Pile-uncopyrighted drops** Books3, BookCorpus2, OpenSubtitles, YTSubtitles and OWT2 (from memory; check the dataset card). That leaves about 17 domains. Use its official val/test splits.
12. **[minor] DoReMi defaults to check, not invent:** step size η=1, c=1e-3, a 280M proxy in the main runs. Its headline result was downstream accuracy at 8B, not perplexity.
13. **[minor] Marin "1.4B" and framework are unverified.** Marin is JAX/TPU-first (Levanter) and has its own DoReMi implementation. Check that it runs on A100/H200.

## 4. Statistics
14. **[critical] Pseudoreplication.** The domain × seed cells share a model per seed. Permuting them gives p-values that are too small. With seed as the unit and 3 seeds, the smallest two-sided sign-flip p is 0.25. **Fix:** use ≥6 seeds (smallest two-sided p 0.03125), or a mixed model with seed random and domain fixed.
15. **[major] No power analysis.** **Fix:** measure the across-seed SD in the pilot. Pre-register the minimum detectable effect in nats.
16. **[major] No multiplicity correction** across about 9 comparisons × 2 metrics. **Fix:** apply Holm correction, or name one confirmatory test (H_fix).
17. **[major] H_static claims equivalence from "within noise."** **Fix:** use TOST (two one-sided tests) with a pre-set margin.
18. **[minor] "≥0 downstream, no regression beyond CI" is ambiguous.** **Fix:** use a non-inferiority margin, e.g., CI lower bound > −0.5 points.

## 5. Runbook ambiguities
19. **[major] The gate wording conflicts.** README says "all corpora and suites"; the prompt says "mid-CR." Exp 02 has no numeric threshold and may have used `the_pile_deduplicated`. **Fix:** give a numeric criterion on the pile-uncopyrighted P7 × V_agi slice.
20. **[major] Units and filters aren't defined.** Are weights over tokens or documents? Where do documents under 512 bytes go? **Fix:** use token-level weights and a dedicated short-document bucket.
21. **[major] Two candidate frameworks.** Stage A may use DoReMi code while Stage B uses Marin, which confounds transfer. **Fix:** use one framework, or report the switch as a deviation.
22. **[minor] Missing runs.** The 24-run count leaves out reference/proxy runs. The pilot (5% of A1) won't time the DRO steps.
23. **[minor] Stage B selection is loose.** "Best of {A6, A7}" has no rule, which risks selection bias. H_static doesn't justify Stage B. **Fix:** pick on the dev split only; trigger Stage B on H_fix alone.
24. **[minor] Split overlap.** The DRO-training, development and evaluation splits aren't declared separate.
