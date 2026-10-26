# Experiment 07 version 1: training calibration, then within-stratum alignment

**Doc link:** <https://github.com/brando90/zip-mix/blob/main/experiments/07_training_calibration/expt_v1/PROTOCOL.md>

**TLDR:** Frozen 10-18-2026 before any Experiment 07 training. Stage A fixes the fine-tuning recipe on fresh calibration questions so fine-tuned models at least match the unchanged base; only if it passes does Stage B run the nine-cell alignment design (extended to five seeds) on fresh held-out questions. Nothing here has been measured.

## Why this exists

Every fine-tuned model in Experiments 05 and 06 scored below the unchanged Qwen2.5-0.5B base (404/500 = 80.8% science, 285/500 = 57.0% retention). [NEXT_STAGE_DESIGN.md](../../04_validation_guided_zipmix/NEXT_STAGE_DESIGN.md) therefore requires a training calibration on a new partition before another selector is admitted, then a test of target alignment within fixed source × compression strata. This protocol implements both, in that order, with every choice fixed before any outcome.

## Fresh data (frozen; `prepare_calibration.py`)

"Previously used" is reconstructed exactly, not approximated. The script rebuilds Experiment 05 version 1 and version 2 from the pinned public revisions and refuses to continue unless both `identities.json` digests equal the committed parent manifests. It then rebuilds the version 2 pool's 351 equal-byte packs and gzip/LZ4 scores and refuses to continue unless the Stage B probabilities reproduce the committed [design receipt](../../04_validation_guided_zipmix/expt_v1/results/next_stage_diagnostics.json): effective sample sizes 8,488.46 and 8,480.91, total variation 0.168522 and 0.224942.

Protected on entry (never reused): every Experiment 05/06 training record (10,116, including the 256-record wrong-target reserve), the full official SciQ validation and test splits and the full CommonsenseQA validation split. Exact normalized-question/content matches and any shared 13-word question/choice segment count as overlap, as in Experiment 05.

Fresh sources are the eligible training-split records ranked beyond Experiment 05's 3,000-per-source cap (SciQ 8,540, CommonsenseQA 6,618, OpenBookQA 1,828), plus ARC-Challenge's unused official validation split (298 after deduplication), because Experiment 05 used all 1,116 eligible ARC-Challenge training records. The official SciQ test split has only about 491 never-evaluated eligible questions left, fewer than the 500 the design requires, so it stays reserved for a later confirmation.

Roles are allocated greedily in this priority, each from a role-specific salted hash order (`zip-mix/exp07/v1`); an overlapping record always goes to the earlier role:

| Role | Source | Count | Read by |
|---|---|---|---|
| `holdout_target` | SciQ train remainder | 500 | Stage B only |
| `holdout_retention` | CommonsenseQA train remainder | 500 | Stage B only |
| `calibration_target` | SciQ train remainder | 500 | Stage A selection; Stage B transfer check |
| `calibration_retention` | CommonsenseQA train remainder | 500 | Stage A selection; Stage B transfer check |
| `calibration_dev_views` | SciQ train remainder | 400 records → 8 × 4,096-byte views | pool diagnostics only |
| `calibration_train` | all four sources | 8,121 (SciQ 3,000, CommonsenseQA 3,000, OpenBookQA 1,824, ARC-Challenge 297) | Stage A training |

Post-allocation audit: all 21 pairwise role/role and role/prior overlap counts are 0, and no fresh evaluation record overlaps the Stage B training pool. Calibration packs score like the Experiment 05 pool (mean maximum gzip alignment 0.1006 versus 0.0988; LZ4 ratio range 0.642–0.805 versus 0.649–0.804). Stage A's data directory does not contain the Stage B holdout arrays. Identities and counts: [calibration_manifest.json](calibration_manifest.json), [stage_b_manifest.json](stage_b_manifest.json).

## Training contract (both stages)

Unchanged from Experiment 05 version 2 except where listed: pinned Qwen2.5-0.5B revision `060db649`, full fine-tuning with float32 master weights and bfloat16 autocast, AdamW (β 0.9/0.95, weight decay 0.01), 8-step warmup then cosine, gradient clip 1.0, 128 steps × 16 one-token answer labels, left padding with explicit positions, final-checkpoint evaluation on final-position label logits. Differences: the learning rate is a declared grid (Stage A) or the Stage A selection (Stage B); resume checkpoints every 64 steps (the design's cadence) instead of 16; final weights are kept only with `--save-models`; every cell is evaluated on all of its stage's evaluation sets. One infrastructure resume per cell is allowed and recorded; a resumed CPU smoke cell reproduces the uninterrupted result exactly (tested).

## Stage A: recipe calibration

- **Cells:** learning rate ∈ {2e-5 (the Experiment 05 recipe, as reference), 1e-5, 5e-6, 2e-6} × seeds {0, 1, 2, 3, 4} = **20 cells plus one base evaluation**, uniform sampling over `calibration_train`.
- **Selection (only on calibration validation):** a rate is eligible if its mean retention accuracy is at least base − 1 percentage point. Choose the eligible rate with the highest mean target accuracy; rates within 0.2 points (one question in 500) of the best tie, and ties go to the larger rate.
- **Gate (all must hold, else stop):** (1) the selected mean target accuracy ≥ base target accuracy on `calibration_target`; (2) at least 3 of 5 seeds ≥ base; (3) the recipe moves the model: mean fraction of target predictions that differ from base ≥ 2%, so a near-zero learning rate cannot pass by doing nothing.
- **If the gate fails:** Stage B is not admitted (the trainer refuses it). Report the grid as measured; any new recipe is a separate prospective version. Never tune on Experiment 05/06 test outcomes or the Stage B holdout.
- `analyze_calibration.py --stage calibration` writes `decision.json` with a content hash; Stage B verifies the hash and that it is not a smoke decision.

## Stage B: alignment within fixed source × compression strata

- **Pool and selectors:** the Experiment 05 version 2 pool (9,797 questions, 351 packs, 13 strata) with NEXT_STAGE_DESIGN.md's three selectors, unchanged: conditional signed-contrast ranks (w = exp(log 4 · z)), uniform sampling within each stratum at the population joint mass (equal to uniform sampling), and one within-stratum pack-rank permutation (seed 1707).
- **Admission (checked during preparation, before any training):** joint masses within 1e-12, all probabilities positive, effective sample size ≥ 80% of the pool, total variation ≥ 0.05 from both controls. Current values: maximum joint-mass error 5.6e-17, effective fractions 0.866 and 0.866, total variation 0.169 and 0.225. Passed.
- **Cells:** 3 selectors × seeds {0, 1, 2, 3, 4} = **15 cells plus base**, at the Stage A learning rate. Seeds 0–2 are the design's nine cells and are also reported separately; seeds 3–4 raise the paired-seed count to five, so the smallest two-sided sign-test p-value becomes 0.0625 instead of 0.25.
- **Evaluation:** fresh `holdout_target` (science) and `holdout_retention` (CommonsenseQA) decide the outcome; the calibration validation sets are also scored for the transfer check.
- **Advance to a larger independent confirmation only if,** against each control: mean holdout target gain ≥ 2 percentage points, at least 4 of 5 paired seeds positive, and mean retention change ≥ −1 point. Otherwise stop scaling this exact formulation; keep weak or mixed signals visible and declare nothing about compression in general.
- **Recipe transfer check:** if the uniform control's mean calibration-target accuracy is more than 1 point below base, the recipe did not transfer to this pool and the screen is labelled exploratory only.
- **Reporting:** all cells, paired-seed Student-t 95% intervals, exact sign-test p-values, Holm adjustment across the two target comparisons. Run the full manifest: no early stopping, no outcome-driven retuning, no deleting missing cells; infrastructure incompletion is an incomplete experiment, not a negative result.

## Resources

One A100 (40 or 80 GB; about 10 GB is needed). Experiment 05 measured 288 seconds of training per 128-step cell with eight checkpoint writes; with one write per cell, estimate 3.5–5 minutes per cell including evaluation. **Stage A ≈ 1.5 A100-hours, Stage B ≈ 1.2 A100-hours, about 3 A100-hours in total** with setup; declared ceiling 3 hours per stage. Launchers for a SNAP node, a rented 1× A100 and Modal are in [launch/](launch/README.md). No paid model-provider service is called.

## What this cannot show

A passed Stage A shows only that one training recipe does not degrade this base on these benchmark families. A Stage B gain would show useful alignment ranking inside these strata, not that compression bins beat a direct scorer, and remains exploratory: shared public benchmark families, possible pretraining exposure and five seeds. Faithful DoReMi/DoGE, direct ZIP-FIT and modern mixture baselines remain separate work.

**TLDR-end:** [zip-mix: exp07 protocol] Stage A (20 cells) must show a fine-tuning recipe that matches the unchanged base on fresh calibration questions before Stage B (15 cells) tests within-stratum alignment on fresh held-out questions; about 3 A100-hours, nothing launched.
