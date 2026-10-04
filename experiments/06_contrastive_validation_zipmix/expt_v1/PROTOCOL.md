# Frozen contrastive selector protocol

**Doc link:** <https://github.com/brando90/zip-mix/blob/main/experiments/06_contrastive_validation_zipmix/expt_v1/PROTOCOL.md>

Date: 10-04-2026. Condition: `06_contrastive_validation_zipmix/expt_v1`.

## Hypothesis and information boundary

A positive contrast between compression alignment to true SciQ development content and reserved CommonsenseQA training content may identify useful target-specific training structure. The primary question is whether selecting through compression bins helps beyond the candidate's four source proportions. No Experiment 05 accuracy, loss, predictions, or result receipts were inspected to choose these selectors. The new data manifest records the prospective preparation timestamp and input hashes. Completed Experiment 04 loss evidence and training/development-only pool diagnostics motivated this exploratory follow-up.

## Frozen data and selectors

Reuse parent manifest `0d6dec93e7d7ccffd87f95e7ebd19e78c84ec72a546007b122888789aba5cb44` from `05_validation_guided_sft/expt_v2`. Keep all 9,797 candidate questions in exactly the same order, the 351 original packs and their bins, the eight 4,096-byte true and eight 4,096-byte wrong development views, tokenizer, prompt formatting, label tokens, and held-out arrays. The wrong-development records were already removed from the candidate pool. Parent exact and meaningful 13-gram decontamination checks remain binding. Preparation copies necessary frozen arrays without loading held-out contents; no new examples, score views, truncation or contamination rules are introduced.

For pack (p), let (a_p=max(s_p^{true,max}-s_p^{wrong,max},0)). Assign each question its pack's (a_p). For bin (b), (W_b=∑_{i∈b}a_{p(i)}). The primary question probability is (W_{b(i)}/(n_{b(i)}∑_bW_b)). Clipping precedes aggregation; inherited questions retain their population weights rather than giving every pack equal final weight. If the total positive score is zero, record unavailable support and retain all affected cells in the denominator; do not substitute a method.

1. `contrastive_zipmix`: the probability above.
2. `source_matched`: calculate the primary probability mass of each of the four sources, then spread that mass uniformly over that source's questions. Source masses match within floating-point tolerance; the expected question-level distribution differs.
3. `contrastive_shuffled`: one NumPy random-generator permutation, seed 1606, of the 351 positive pack contrasts. Inherit the permuted scores to every member of the receiving pack, then perform the same bin aggregation. This permutation is frozen for all three training seeds. Do not independently shuffle individual questions or reorder development views.

All three probability vectors have nonzero support in the frozen preparation. The new manifest binds the vectors, their diagnostic masses, the inherited data files and preparation sources. The trainer additionally hashes its own source, common configuration and analyzer into a new run identity.

## Training and budget

Nine training cells: three methods × seeds 0, 1, 2, plus one base evaluation. Use Qwen/Qwen2.5-0.5B revision `060db6499f32faf8b98477b0a26969ef7d8b9987`. The trainer is byte-identical to the reviewed parent version. Each cell begins at the same model revision, fully fine-tunes float32 master parameters with bfloat16 automatic mixed precision, and uses 128 updates of 16 questions, giving 2,048 supervised answer tokens per cell and 18,432 across the full matrix. Exactly one asserted answer-label token is supervised; prompt positions are masked by the equivalent last-position loss. Predictions use valid-choice logits at the actual final unpadded prompt token.

Optimizer and schedule are unchanged: AdamW learning rate 2e-5, beta values (0.9, 0.95), epsilon 1e-8, weight decay 0.01, gradient clipping 1.0, eight warmup steps and cosine decay. Input limit 256, sampling with replacement, initialization seeds and all other training settings match Experiment 05. Actual non-padding/padded input tokens, unique examples, time and memory are recorded. Fixed answer-token budgets do not equalize input compute.

The sole execution change is checkpoint cadence 64 instead of 16 steps. This reduces checkpoint writes from eight to two per cell. Keep the existing single-resume limit and full recovery accounting; do not reset completed cells. Final model weights are retained, and completed cells release their optimizer checkpoint. Use ordinary persistent local ext4 storage on the admitted host. Cross-experiment time differences include this operational change and cannot be attributed solely to selection cost.

Parent admission requires one idle A100 device, persistent storage headroom and a durable tested supervisor with a maximum two-device-hour budget. Do not launch separate cells concurrently on the same device. Once admitted, execute the entire nine-cell matrix and base evaluation; an individual successful cell is not completion. The coordinator's turn ending does not stop healthy work. Keep failures, interruptions and unavailable cells in the frozen denominator.

## Evaluation and analysis

Evaluate only the fixed SciQ target and CommonsenseQA retention arrays, 500 questions each, after each cell's final update. Evaluate the base model once. No early stopping, checkpoint choice, learning-rate tuning, target reweighting or method replacement may use these outcomes.

Primary comparison: paired seed-level target-accuracy difference `contrastive_zipmix minus source_matched`. Secondary mechanism comparison: `contrastive_zipmix minus contrastive_shuffled`; retention accuracy and full-vocabulary label loss are diagnostics for both. Report means, paired 95% Student-t intervals, all individual seed values and the exact two-sided sign test against probability 1/2 of a positive non-tied difference. With three non-tied pairs the minimum two-sided p-value is 0.25. These descriptive tests cannot establish conventional statistical significance. The two planned target comparisons form a family; report raw and Holm-adjusted p-values for these two target contrasts in the scientific narrative; no confirmatory significance claim is planned. Secondary outcomes and any later cross-experiment comparisons require explicit multiplicity caveats and cannot supply confirmatory claims through selective reporting.

Only after both full matrices are available may the report compare these nine cells with Experiment 05's 21 frozen cells. Retain all methods and seeds, list the comparison family and label those contrasts exploratory. Reusing the same public benchmark pools and shared pretraining checkpoint does not provide independent replication. A distinct future held-out evaluation is needed for a stronger generalization claim.

## Interpretation limits

The source-matched comparison separates source proportions from within-source selection; it does not prove compression has semantic understanding. Equal scoring width removes raw-byte-length differences but retains content redundancy, pack-density and overflow-member effects. The selected wrong-target source is only one contrast. The shuffled control changes source mass and should not replace the source-matched control. Pretrained-model benchmark contamination remains unexcluded. Three seeds, 2,048 labels per cell and one model provide a screening experiment, not an optimality or state-of-the-art result.
