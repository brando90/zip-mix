# Experiment 04 — prospective protocol

**Doc link:** <https://github.com/brando90/zip-mix/blob/main/experiments/04_validation_guided_zipmix/expt_v1/PROTOCOL.md>

**TLDR:** Freeze an initial nine-method, three-seed from-scratch screen. Diagnose whether validation alignment changes held-out loss; no broad superiority conclusion can be accepted from this small screen.

Written 10-04-2026 before any training outcomes. Configuration and input hashes will be frozen in each run manifest. Only pretraining throughput may determine the bounded allocation; never use held-out performance to change settings. Once measured training starts, deviations belong in the checkpoint and a new prospective condition, not this file.

## Scientific question (locked)

Does the original maximum-similarity Zip-Mix prior improve biomedical held-out negative log-likelihood over token-proportional sampling and an alignment-permutation control, while preserving non-biomedical loss? Primary contrasts are Zip-Mix minus token-proportional, shuffled Zip-Mix, DoReMi and DoGE. Lower loss is better.

An **agent-suggested screening criterion** for proceeding to larger confirmation is mean target-loss reduction >=0.02 natural-log units relative to both token-proportional and shuffled Zip-Mix, paired direction consistent across all three seeds, and mean broad loss increase <=0.02. This criterion is a decision heuristic, not statistical proof. Report every comparison regardless of outcome. With three paired seeds, an exact two-sided sign test has minimum p-value 0.25: this screen cannot prove significance at 0.05. A failure or inconclusive result does not authorize selecting a favorable benchmark or dropping a method.

## Locked parameters

- Source: inherited 20,000-document Pile-uncopyrighted pilot, exact source hash recorded; no new source sampling chosen by scores.
- Domains: original domains with >=250 source documents and >=100 eligible deduplicated documents after the >=1024-byte filter. Deterministic identity ordering per domain: first 32 development, next 64 evaluation, remaining training. Corpus prefix cap 8192 bytes; no text crosses document boundaries.
- Target: PubMed Abstracts and PubMed Central. All other eligible domains form the broad non-target evaluation. These are domain-level language-model tests, not benchmark accuracy tests or a measure of artificial general intelligence.
- Leakage controls: normalized exact full-document deduplication; training prefixes sharing any normalized13-word sequence with a development/evaluation prefix removed. This does not certify absence of all paraphrases or overlap beyond the retained prefixes.
- Tokenizer: byte-pair encoding,8192 vocabulary requested, fit on training text only,128-token contexts plus one next-token label. At most2048 tokens per training document. Evaluation uses the first129 tokens of each held-out document.
- Compression: LZ4 default framing on retained document prefix. Bucket boundaries 0.50,0.60,0.67,0.73,0.80,0.90; last bucket unbounded. Gzip level 6, fixed 1024-byte source and development prefixes; compression-distance concatenation source then target. All target development documents (up to 64) used; negative similarities preserved in data then clipped to zero for sampling.
- Methods: token_proportional, uniform_domain, uniform_bucket, compel_filter, zipmix_static, direct_zipfit, shuffled_zipmix, doremi, doge. Three seeds0,1,2. Expected final-model cells=27. Original Zip-Mix uses maximum similarity; direct ZIP-FIT uses mean similarity with soft sampling, explicitly an aggregation comparator rather than paper-identical top-k selection. Compel retains compression ratio[0.65,0.80]. Empty buckets receive zero mass. Zip-Mix sums block scores plus 1e-3 on nonempty groups: this is the token-block extension of the draft's document formula.
- Learned methods use separately charged reference/proxy training, average learned weights, then fresh final training from the same initialization used by simple methods. DoReMi reference samples token-proportionally; proxy uses clipped per-token excess loss. DoGE uses all-parameter gradient alignment with target development loss. Full numeric optimizer/model/budget settings are frozen in `config.json` before launch; no method-specific tuning in this screen.
- Equal final training tokens and initialization seeds; report total cost including selection, reference and proxy overhead separately. This is **not equal total compute**. No claim of compute optimality follows.
- No early stopping from evaluation, no test tuning, no result-dependent reruns. One interrupted cell may resume its last checkpoint with preserved random-number/optimizer state; a failure without resumable state remains failed until a separately documented recovery condition is frozen.

## Locked metrics and analysis

Per-example mean negative log-likelihood in natural-log units and perplexity; primary target mean and broad mean with domain-level values shown. Across-seed mean and standard deviation; paired seed-difference 95% Student-t interval and exact two-sided sign-test p-value against zero. Three seeds yield fragile intervals and weak tests. Item counts do not increase the number of independent training replicates. All27 cells remain visible as pending/running/complete/failed. Missing scores are null, never silently dropped or replaced by zero. Report full-population results incomplete if any required cell is missing.

## Scale verification

Compression similarities may be negative before explicit nonnegative clipping; they are not probabilities. Only the sampler normalizes weights. Loss is natural-log loss per predicted token; perplexity=exp(loss), without additional division. Count final/reference/proxy tokens separately.

## Verified data schema

Inherited parquet columns: doc_id,text,domain. `train.npz`: tokens[N,129],domain[N],bucket[N],score[N],mean_score[N],cr[N],doc_id[N]. `dev.npz`: tokens/domain for target development only. `eval.npz`: target_tokens/target_domain and broad_tokens/broad_domain. `manifest.json` carries data, tokenizer and split hashes; no test loss enters preparation.

## Preconditions and wall-clock

Deterministic tests pass; one-device timing smoke verifies finite loss, checkpoint/evaluation output and expected memory; data hashes verified after transfer. Use a single graphics processing unit, explicit CUDA_VISIBLE_DEVICES, durable process ownership and a recorded progress monitor. Initial allocation bound 12 device-hours for the full screen, pending throughput verification. No new cloud spending. Model-provider subscription use is limited to the one user-requested review.

## Completion and abort rules

Smoke outputs are engineering-only and excluded from scientific results. Freeze the final manifest before admitting its first cell and allocate enough time for all 27 cells. A coordinator ending its turn is never a reason to stop healthy admitted work. Stop admission only for invalid data, nonfinite numerical behavior, resource safety, explicit cancellation, or the prespecified resource ceiling; preserve all failures and remaining cells. Never claim completion because one model finished.

## Identity

Repository owner: brando90, verified from the configured Git remote. No contact message, model-provider key, external dashboard or paper publication is needed for this experiment.
