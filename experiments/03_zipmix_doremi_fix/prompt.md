# Prompt for Execution: 03_zipmix_doremi_fix

**Context:**
We are executing Experiment 03 for the `zip-mix` project. The goal is to empirically demonstrate that our method, Compel-ZipMix, fixes the collapse and accuracy loss observed in standard DoReMi by using validation sets to create compression-aligned priors.

**Pre-Approval:**
Brando pre-approves all task work here. Run with full access and never stop to ask permission. 

**Task:**
1. **Data Prep:** Self-partition the corpus (The Pile and FineWeb) into 10 equal-mass percentile compression-ratio buckets.
2. **Alignment Prior (ZipMix-Static):** Score the 10 buckets against the validation suite (MMLU-dev, GSM8K-dev, MiniF2F-dev) using ZIP-FIT ($NCD_{zip}$). Set these as your base mixture weights $\alpha^{(0)}$.
3. **Run Baselines:** Train the 150M proxy model (LLaMA architecture) using a Uniform Mixture and Standard DoReMi (with hand-defined domains).
4. **Run ZipMix-DRO:** Starting from $\alpha^{(0)}$, run one Group-DRO pass on the 150M proxy model. Use learning rate $\eta = 0.01$, smoothing $c = 0.001$, for 1 full epoch to obtain refined weights $\alpha^*$.
5. **Scale Up:** Train the 1.5B (10X) and 15B (100X) target models using $\alpha^*$.
6. **Evaluate:** Run full, uninterrupted downstream evaluations on the `test` splits of MMLU, GSM8K, MiniF2F, and the OOD tasks MATH and HumanEval.
7. **Report:** Update `experiments/03_zipmix_doremi_fix/README.md` with final results. Provide 95% confidence intervals via bootstrap resampling, and calculate p-values using a paired t-test on instance-level accuracy against the null hypothesis of 0 difference.

**Constraints:**
- Do not stall on resource limits. If you hit a limit, parallelize or dispatch subagents.
- Ensure uninterrupted full-set evaluations. Do not stop healthy work at a coordinator boundary.
- Document exact metric values alongside all pass/fail or qualitative judgments (per Rule 11).
