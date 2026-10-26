# CKPT: Experiment 07 training calibration

**Doc link:** <https://github.com/brando90/zip-mix/blob/main/experiments/07_training_calibration/CKPT_training_calibration.md>

Created: 10-18-2026. Last updated: 10-26-2026.

## State
- **Goal:** fix the fine-tuning recipe so trained models at least match the unchanged base on fresh questions (Stage A), then test within-stratum alignment on fresh held-out questions (Stage B).
- **What ran:** data freeze on CPU (Experiment 05 v1/v2 identities and the design receipt reproduced exactly; all overlap audits 0; 21 tests). Stage A on FarmShare (job 1861877, one L40S): 20 cells + base, 0 failures. Stage B (job 1862429): 15 cells + base, 0 failures. Exploratory Exp07b sweep (job 1863035): 40 cells + base.
- **Results:** Stage A gate passed, selected lr 2e-6 (`expt_v1/results/stage_a/decision.json`). Stage B `advance=false` (`expt_v1/results/stage_b/analysis.json`): conditional_signed_rank 80.76% vs uniform_stratum 80.00% and permuted_stratum_rank 79.96% on holdout_target (base 78.80%); paired contrasts +0.76 pp [−0.70, +2.22] and +0.80 pp [−0.32, +1.92], 3/5 seeds positive each. Exp07b: `expt_v1/results/results_exp07b.md`. Appendix A added to both paper drafts.
- **Stuck / blocked:** nothing running. The null diagnosis figures quoted in the appendix (SciQ-indicator correlation 0.89, 0.07 pool-sd alignment gap, 5.6% same-arm seed disagreement) come from `diagnose_null.py` on the private prepared data and are not reproduced from committed files alone.
- **Next step:** none scheduled; a follow-up would need a target with more headroom (see Exp07b takeaway 3) before more selector work.

## Log
- 10-18-2026: protocol frozen, data prepared and verified on CPU, launchers written; nothing launched.
- 10-20-2026: Slurm launchers for FarmShare and Sherlock added.
- 10-21-2026: Stage A (job 1861877) gate passed, lr 2e-6; Stage B (job 1862429) advance=false; Exp07b exploratory sweep (job 1863035); Appendix A written for the DMLR and ICLR drafts.
- 10-26-2026: second-model review before merge; launcher mail addresses moved out of the repo (`ZM_MAIL_USER`), index and checkpoint updated to the completed state.
