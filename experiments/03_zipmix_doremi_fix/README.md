# Experiment 03: ZipMix Doremi Fix

## Objective
To demonstrate that Compel-ZipMix (compression-aligned mixtures) empirically fixes the accuracy loss and domain-collapse issues observed in standard DoReMi, by replacing uninformative coarse domains and uniform priors with self-partitioning compression-ratio buckets and a validation-aligned ZIP-FIT prior.

## Hypothesis
- **H1 (Prior Quality):** ZipMix-Static (sampling directly from the alignment prior) will achieve a higher downstream accuracy on the AGI validation suite (MMLU, MiniF2F) than the No-Free-Lunch uniform prior baseline used by DoReMi.
- **H2 (Refinement Robustness):** ZipMix-DRO will prevent the probability-mass funneling into noise buckets (defined as >50% weight assigned to the top CR quartile). This will lead to faster convergence and at least a +2.0 pp accuracy gain over the baseline, without sacrificing scaling properties.
- **H3 (Transferability):** The mixture weights learned by a proxy model using ZipMix-DRO will successfully transfer to target models up to 100x larger, achieving an Iso-FLOP loss reduction at least 90% as large as the proxy model's relative to the uniform baseline.

## Experimental Setting
- **Corpus:** The Pile and FineWeb.
- **Validation Suites (Prior):** MMLU-dev, GSM8K-dev, MiniF2F-dev.
- **OOD Evaluation Suites:** MATH-test, HumanEval-test.
- **Model Architecture:** LLaMA-style transformer.
- **Proxy Model:** 150M parameters.
- **Target Models:** 1.5B (10x) and 15B (100x) parameters.
- **Baselines:** Uniform Mixture (baseline), Standard DoReMi (hand-defined domains).
- **Compression Bucketing:** Data partitioned into 10 equal-mass percentiles via Normalized Compression Distance ($NCD_{zip}$).

## Metric Definition
- **Mixture Weight Quality:** Weight dynamic stability (variance of weights across training epochs) and mass assigned to noise buckets (highest CR quartile).
- **Downstream Accuracy:** Zero-shot and few-shot accuracy on in-domain (MMLU, GSM8K, MiniF2F) and OOD (MATH, HumanEval) test splits.
- **Pre-training Efficiency:** Iso-FLOP downstream accuracy at a fixed compute budget.

## Results
*(Pending execution)*
- **TL;DR finding:** (To be added once results are generated).
- **Config Table:** (To be added).
- **Results Table:** (To be added).

## Why These Numbers Are Correct
*(Pending execution)*
- All headline estimates will include 95% confidence intervals derived via bootstrap resampling over independent task instances.
- Pass/fail verdicts and accuracy gains will be supported by a paired t-test on instance-level accuracy against the null hypothesis of 0 difference.

## Known Limitations
- ZipMix relies heavily on the quality and representativeness of the small validation suite $V$.
- Compression ratio ($CR$) as an information-density proxy might be language-dependent and could skew weights in highly multilingual corpora.

## Verification Checklist
- [ ] Ensure full-set uninterrupted evaluations are run (no early stopping of evaluations per Trigger Rule 61).
- [ ] Measure standard deviations and repetition counts for aggregated scores.
- [ ] Verify that Target models do not exceed the $10^3$ proxy size scaling limit.
- [ ] Compare explicitly to a run of Standard DoReMi on the same datasets.

## Reproduction
*(Commands to be added upon completion of execution scripts)*

## Experimental Details
- **Dates:** MM-DD-YYYY to MM-DD-YYYY
- **Hardware:** SNAP nodes
- **Model IDs used for generation/evaluation:** claude-fable-5-1, gpt-6-astra
