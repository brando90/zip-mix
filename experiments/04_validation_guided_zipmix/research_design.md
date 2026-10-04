# Experiment 04: validation-guided ZipMix research design

**Doc link:** <https://github.com/brando90/zip-mix/blob/main/experiments/04_validation_guided_zipmix/research_design.md>

**TLDR:** Test a falsifiable compression-alignment mechanism first, then confirm at meaningful pretraining and post-training scales against faithful, competitively tuned baselines. The executing version freezes its actual parameters; this roadmap does not report results.

Created 10-04-2026. Status: prospective design; this document reports no training result. The executing version's frozen `PROTOCOL.md`, configuration, and manifest must record the settings actually admitted. Numbers below are bounded research choices, not measured performance or hardware estimates.

## Scientific question

Does choosing training-data mixtures from compression-based similarity to a small development set improve performance on previously unused examples and task families, at a competitive total cost, compared with token-proportional sampling, Domain Reweighting with Minimax Optimization (DoReMi), Domain Reweighting with Generalization Estimation (DoGE), and relevant modern selection methods?

“Generalizes much better,” “optimal training,” and “state of the art” are hypotheses, not initialization milestones. A finite benchmark suite also does not establish artificial general intelligence. We can demonstrate a cheaper, useful target-conditioned selector without proving universal optimality. Negative outcomes are scientifically useful: compression can select surface form rather than transferable content, and aggregation into a handful of buckets can erase document-level alignment.

The cheapest sufficient first test combines a small model trained from scratch, where held-out negative log likelihood is informative, with supervised fine-tuning (SFT) of an existing capable model, where exact-answer accuracy can change. Reinforcement learning (RL) is a subsequent, separately frozen study; adding it immediately would introduce reward, rollout, and curriculum confounds before the selection mechanism is established.

## Corrections to the existing proposal

1. **DoReMi is not defined by an uninformative reference prior.** Its official implementation recommends domain-size reference weights and explicitly uses reference-model excess loss to reduce undue focus on intrinsically difficult domains. Keep the robust objective and reference model in any faithful baseline. A raw-loss adversary is a different method. Failure in a reported setting does not establish that DoReMi generally fails. [Official implementation](https://github.com/sangmichaelxie/doremi), [paper](https://arxiv.org/abs/2305.10429).
2. **The published ZIP-FIT algorithm averages over target examples.** Algorithm 1 defines alignment as `1 - mean_j NCD(x, v_j)`. Existing Experiment 02 describes a maximum as though it were the paper method. Preserve old outputs as a named legacy variant; use the mean prospectively, with maximum similarity only as a clearly labeled ablation. [ZIP-FIT Algorithm 1](https://arxiv.org/html/2410.18194v2#S2).
3. **A middle-compression peak is not a training-success gate.** The Compel band is a useful external baseline, not the desired answer to the validation experiment. Favoring another band may be appropriate for code or mathematics. Admitting training only when ZipMix reproduces the band makes the mechanism test circular. [Compel paper](https://openreview.net/pdf?id=KFafeqE5fe), [author repository](https://github.com/stair-lab/compel).
4. **A sum of scores is also a population prior.** For roughly constant scores, `sum_{d in bucket} score(d)` is proportional to bucket population. A stratified sample with equal counts per bucket must be inverse-probability weighted before estimating a corpus total. Otherwise it silently replaces the method with mean-score weighting. Test the correct null against the corresponding population prior, not automatically against uniform weights.
5. **Initialization alone need not change a converged robust optimum.** A ZipMix-initialized DoReMi update may eventually forget its starting weights. Plot weight trajectories and compare final and averaged weights. A persistent prior penalty is a separate method requiring its own coefficient, ablation, and name; it must not be introduced under the claim that initialization fixes the robust objective.
6. **The current baseline landscape extends beyond DoReMi and DoGE.** Aioli provides a shared implementation framework, Olmix studies current offline mixing choices, and On-Policy Mix (OP-Mix) addresses multiple training phases. Their papers are relevant competitors, not evidence that any one is universally best. [Aioli source](https://github.com/HazyResearch/aioli), [Olmix paper](https://arxiv.org/abs/2602.12237), [OP-Mix paper](https://arxiv.org/abs/2605.15220).

## Separate the claims before testing them

| Claim | Necessary evidence | Insufficient evidence |
|---|---|---|
| Compression scoring contains target information | Target-specific changes survive length, source, and shuffled-target controls | Correlation with compression ratio alone |
| Buckets help | ZipMix improves over direct ZIP-FIT selection and a source-domain prior at equal resources | Beating uniform source weights only |
| Development targeting helps | Correct-target ZipMix improves over shuffled or mismatched targeting | A peaked prior |
| It improves within-task generalization | Gains on unused examples from target task families | Gains on examples used for scoring or tuning |
| It transfers beyond target tasks | Gains on untouched task families with no targeting examples | A test split from the same benchmark alone |
| It is efficient | End-to-end cost including scoring, tuning, reference and proxy models | Equal final-training tokens only |
| It beats named baselines | Faithful implementations, competitive tuning, paired replicated comparisons | A locally simplified method carrying a paper's name |

The primary mechanistic risk is information loss: there may be large within-bucket variation in target relevance and only weak differences across buckets. Compare document scores with bucket averages. If a direct selector wins consistently and bucket weights remain nearly equal to the source prior, the right conclusion may be that compression alignment works while compression bucketing does not help.

## Data separation and contamination

Freeze immutable record identifiers, dataset revisions, text normalization, duplicate groups, and split hashes before selecting mixtures. Split at the original document/problem and duplicate-cluster level before chunking. Never split neighboring chunks from one document across roles.

| Data role | Permitted use | Prohibited use |
|---|---|---|
| Candidate training pool | Training, compression bins, source statistics | Contains neither development nor test records |
| Alignment development set | Compression scores; target-aware baseline gradients, features, and labels under the same contract | Final generalization estimate |
| Selection/tuning development set | Hyperparameters, model/mixture selection, scout decisions | Reported confirmatory test |
| Same-family test | One final evaluation at frozen checkpoints | Mixture scoring, parameter tuning, early stopping |
| Unseen-family test | Transfer evaluation; no examples from those families in alignment or tuning | Calling a same-family split “unseen task” |

For a benchmark with no official development split, make deterministic development and tuning subsets from its training split and remove all corresponding examples from the candidate pool. If it has only a test set, reserve it entirely for evaluation or explicitly partition it before any measurement and stop calling the remainder the official full benchmark. The latter weakens comparability and is not preferred.

Use normalized exact hashes plus near-duplicate detection before scoring, and report overlap rates on the actual sampled training exposure by arm afterward. A single matching common 13-word phrase is not definitive contamination; preserve hit counts and the rule that distinguishes boilerplate from a substantive duplicate. Account for paraphrases, repeated problem templates, code solutions, repository/file siblings, and official benchmark training examples embedded in instruction mixtures. No string test certifies absence of all contamination.

Compression scoring sees prompt text only in the primary SFT condition; responses are used to compute training loss. If target answer text is used for selection in an ablation, name that condition, give equally capable target-aware baselines the same information, and account for the different information budget. Test answers remain sealed to the training and selection path. Removing the recognizable shared chat wrapper from scoring prevents template bytes from creating spurious alignment; retain a fixed wrapper for model training.

For pretrained SFT backbones, prior exposure in the backbone is generally unknown. Comparisons share the exact backbone revision so this exposure is held constant, but a gain remains an incremental adaptation result rather than proof of never-seen pretraining generalization.

## A prospective, explicit ZipMix definition

Use a fixed implementation and version of LZ4 for compression ratio, `compressed_bytes / original_utf8_bytes`; record framing, compression level, and source unit. Empty input is invalid. Ratios above one are possible and belong to the open-ended last bucket. Begin with five training-only quantile bins to avoid empty bins at small scale; retain the historical seven fixed boundaries and Compel's `[0.65, 0.80]` band as named sensitivity conditions. Quantile choice is practical, not evidence of optimality.

For gzip alignment, cache individual compressed lengths and score fixed-size, length-controlled text views drawn reproducibly from each document and each development example. Keep the primary view rule constant across arms, including direct selection. Record UTF-8 truncation, separators, header settings, and gzip version. Long-document views must respect the compressor's finite window; scoring arbitrarily long concatenations is not a semantic-distance oracle. Inspect both concatenation orders and short-string header effects in calibration.

The paper-faithful score uses a fixed concatenation order:

```text
NCD(x, v) = [C(x || v) - min(C(x), C(v))] / max(C(x), C(v))
a(x) = 1 - mean_v NCD(x, v)
```

Here NCD means normalized compression distance and `C` is the byte length produced by the recorded compressor. Real compressors can yield scores outside the nominal range; store raw scores. A prospective nonnegative mapping such as `w(x) = max(a(x), 0)` is an explicit ZipMix design choice, not a theorem or a silent repair. Record clipping frequency; all-zero scores trigger a declared fallback to the reference mixture. A symmetrized score is a separately named sensitivity condition.

Use packed training-token proportions for the primary mixture. Let `t(x)` be a document's usable token count and `q_m = sum_{x in B_m} t(x) / sum_x t(x)` be the training pool's baseline token share. Let `mu_m` be the token-weighted mean of nonnegative alignment weights in bucket `m`. Define:

```text
r_m = q_m * mu_m / sum_j(q_j * mu_j)
alpha_m = (1 - lambda) * q_m + lambda * r_m
```

Fix `lambda = 1` for the first implementation to minimize method degrees of freedom; a declared small tuning grid may later choose mixing with the baseline using tuning data only. Sampling bins then equal-length packed blocks makes `alpha` a token mixture. Do not sample variable-length documents uniformly and call the resulting frequency a token mixture. Document-count weighting is a separate historical ablation. Unobserved or empty buckets receive no fictitious training records.

Report `q`, `alpha`, effective unique tokens, repeat counts, source proportions, bucket populations, score spread, and the maximum weight. A score subsample estimates `mu_m` with the correct token and sampling-probability weights; reusing its observed counts as full corpus counts is invalid.

The direct ZIP-FIT baseline uses the same development examples, compression implementation, byte views, and score budget, then samples/selects at document level. This isolates whether buckets add a useful inductive bias or simply discard information.

## Baseline suite and implementation contracts

| Baseline | Role and required behavior | Source |
|---|---|---|
| Token-proportional sampling | Mandatory natural-mixture baseline from actual usable tokens | Definition above |
| Uniform source-domain sampling | Separates domain balancing from alignment; not a substitute for the natural mixture | Fixed source metadata |
| Uniform compression-bin sampling | Separates changing the partition from targeting | Training-only bins |
| Compel | Compression hard filter; equal final-training tokens with repeats accounted for | [Author source](https://github.com/stair-lab/compel) |
| Direct ZIP-FIT | Document-level alignment selection; essential nearest predecessor | [Paper](https://arxiv.org/abs/2410.18194) |
| DoReMi | Reference model, excess-loss update, appropriate reference weights, averaged proxy weights, fresh final model | [Official implementation](https://github.com/sangmichaelxie/doremi) |
| Target-aware DoGE | Development-target gradient alignment; expose the same target information as ZipMix; fresh final model after proxy learning | [Algorithm 2](https://arxiv.org/html/2310.15393v2#S2) |
| Aioli | Online mixing-law estimation, charging probing updates to its training budget | [Official implementation](https://github.com/HazyResearch/aioli) |
| Olmix | Current offline regression/mixing comparator; pin the implementation, swarm budget, and constraints | [Official implementation](https://github.com/allenai/olmix) |
| Data Selection via Importance Resampling (DSIR) | Cheap non-neural target-matching comparator, especially useful in SFT | [Paper](https://arxiv.org/abs/2302.03169) |
| Low-rank gradiEnt Similarity Search (LESS) | Strong targeted instruction-selection comparator; charge warmup and gradient storage | [Official implementation](https://github.com/princeton-nlp/LESS) |
| Benchmark-Targeted Ranking (BETR) | Neural target-selection comparison for larger pretraining; include embedding/classifier costs | [Paper](https://arxiv.org/abs/2507.12466) |

An ordinary raw-gradient dot-product reweighter in a tiny trainer can be called a **DoGE-style ablation**, not a faithful reproduction. Likewise, online raw-loss weighting is **group robust weighting**, not DoReMi. If small-scale engineering makes only these approximations runnable today, the scout must label them explicitly and retain the named-baseline comparison as outstanding.

DoReMi is intentionally target-agnostic, whereas ZipMix is targeted. Therefore beating DoReMi on a targeted metric alone does not isolate compression's contribution. The target-aware DoGE, direct ZIP-FIT, DSIR, and LESS comparisons are needed for that. For the partition comparison, also compare the same target-aware score over source domains, compression bins, and direct documents.

Current methods to consider after the initial suite are Adaptive Data Optimization (ADO), which adapts domain allocation during training; OP-Mix for continued training; and Group Robust Multi-target Adaptive Pretraining (GRAPE), which addresses multiple targets. Include one when its setting matches the prospective experiment and a faithful implementation fits the recorded budget. [ADO paper](https://arxiv.org/abs/2410.11820), [OP-Mix paper](https://arxiv.org/abs/2605.15220), [GRAPE author repository](https://github.com/Olivia-fsm/GRAPE_data_mixture_with_multi_target). This is a literature-informed comparator selection, not an exhaustive ranking of all work available on 10-04-2026.

## Staged experimental suite

### Stage 0: cheap calibration and implementation validity

Use a frozen stratified sample from at least three naturally labeled source domains, with fixed limits on records and development views. Suggested starting bound: 10,000 source records, 128 development views, and five compression bins. Measure scoring wall time before enlarging it. Add no language-model calls to this preprocessing.

Check duplicates/split integrity; finite scores and their range; bucket coverage; true token-mass aggregation; sampling reproducibility; mean versus maximum aggregation; sensitivity to byte length and concatenation order; and throughput. Compare a math-like target, a code-like target, and an equal-size length-matched mismatched target. Shuffling examples only within the target set leaves a mean score unchanged and is not a valid negative control: permute document scores within source-by-length strata or replace the target with a distinct length-matched set.

Admission criteria are correct identities, disjoint roles, finite training/evaluation losses, adequate eligible data, and a measured resource projection that covers the complete manifest. A particular favorable weight pattern is never required. Zero useful score spread is a valid negative result and motivates the smallest training comparison, not a large launch or retrospective rescaling to manufacture a signal.

### Stage 1: from-scratch pretraining scout

Recommended first complete training experiment: a decoder-only model with roughly 60–160 million parameters, random initialization, 512-token contexts, and a frozen 50–100 million training-token budget per final model. The executor chooses and freezes one exact architecture and budget after throughput-only calibration. No pretrained weights are loaded. All methods use one tokenizer, optimizer schedule, packing rule, precision, context size, and fixed final checkpoint.

Use a compact multi-domain source pool with traceable source metadata and enough unique tokens for the natural mixture. A bounded cached subset of the project's Pile-uncopyrighted source or a SlimPajama subset is suitable; do not download an entire web corpus merely to start. State the precise retained domains and deviations from any original paper. A WikiText-only or synthetic corpus is an infrastructure/control study and cannot stand in for heterogeneous web-data validation.

Primary final-model arms:

1. Token-proportional sampling.
2. Uniform compression bins.
3. Compel hard filter.
4. Direct ZIP-FIT.
5. ZipMix with the mean-score prior.
6. ZipMix with document scores permuted within source-by-length strata.
7. Faithful DoReMi.
8. Target-aware DoGE.
9. Aioli, if its official implementation can be integrated faithfully within the same stage; otherwise admit it in the next explicitly versioned benchmark suite and state its absence.

Freeze three paired seed values for the scout, giving 24 final training runs for the eight mandatory arms, or 27 including Aioli. Auxiliary reference/proxy runs are additional manifest rows; every one must be counted. A three-seed scout estimates direction and variability and tests pipeline feasibility. It is not powered to establish “much better.” Report confidence intervals but keep the scout descriptive; do not turn millions of token losses into millions of independent training replications.

Primary metric: mean held-out token negative log likelihood, macro-averaged over the frozen target components. Also report natural-mixture held-out loss, per-domain loss, worst-domain loss, and perplexity `exp(mean loss)`. Do not average perplexities when the intended estimand is exponential mean loss. Hold out source documents and a complete task-family block separately. Tiny-model benchmark accuracies are secondary diagnostics; report chance level and measured counts where appropriate.

Use the selection/tuning split to choose future settings. After publishing the entire scout, freeze a new confirmation manifest; do not silently promote the winning scout result into confirmation. A negative scout still completes all admitted cells.

### Stage 2: targeted SFT scout with usable benchmark signal

Recommended backbone: the exact pinned revision of `Qwen/Qwen2.5-1.5B` base, an openly available pretrained model of approximately 1.54 billion parameters. This is a resource-driven choice, not a claim that it is the newest model. [Official model card](https://huggingface.co/Qwen/Qwen2.5-1.5B).

Use a frozen, decontaminated instruction candidate pool with natural source labels. A bounded sample of `allenai/tulu-v2-sft-mixture` provides such metadata, but its task overlap must be audited before use. An explicitly constructed mixture from source training splits is a valid smaller alternative. Do not pool unexplained arbitrary rows and then claim source-domain comparability. [Dataset card](https://huggingface.co/datasets/allenai/tulu-v2-sft-mixture).

Use two separately specified target conditions, mathematical question answering and Python programming. For the first inexpensive launch, admit one condition and declare the other prospectively; do not choose the condition after inspecting test scores. Math development examples can come from a disjoint subset of the Grade School Math 8K (GSM8K) training split; programming development examples can come from the official Mostly Basic Python Problems (MBPP) development or training pool. Record exact available split names before freezing. [GSM8K source](https://huggingface.co/datasets/openai/gsm8k), [MBPP source](https://huggingface.co/datasets/google-research-datasets/mbpp).

Use eight mandatory arms: token-proportional, source-uniform, Compel, direct ZIP-FIT, ZipMix, permuted-score ZipMix, DSIR, and LESS. A target-gradient source-domain mixture arm is valuable but must be called an SFT adaptation unless it faithfully implements the cited method. DoReMi and DoGE remain mandatory in the pretraining comparison; their inclusion there does not by itself establish superiority over targeted SFT selection methods.

Freeze three seeds, one low-rank adaptation configuration, one optimization schedule, and a bounded total budget such as two million supervised response tokens per model. Count both supervised tokens and all processed input tokens. Match response-token budget for the adaptation estimand and report forward-pass cost as well; padding and long prompts can otherwise make nominally equal budgets unequal. Record actual unique examples and repetition. Empty responses, truncation, and masked-label handling are explicit validation checks.

Primary same-family metric is exact final-answer accuracy for math or deterministic code-test success for programming, with a frozen generation template, decoding parameters, timeout, and answer parser. For code, run generated programs in an isolated evaluator with fixed resources. Retain response negative log likelihood as a smooth secondary metric. Cross-family transfer requires whole held-out task families; label a new dataset from the same skill as within-skill transfer, not broad generalization. Include a frozen general-capability retention suite to detect forgetting.

Use the base checkpoint's evaluation as a reference, not as an extra competing mixture with a zero-token advantage. Selecting only the best checkpoint by test accuracy is prohibited; use the fixed final checkpoint or a tuning-only rule shared by all arms.

### Stage 3: confirmation, transfer, and the meaning of “much better”

Before this stage begins, select the ZipMix variant using only scout development data, freeze the best competitive baseline configurations using the same allowance, and retain all named comparators relevant to the claim. Add Olmix for a current offline pretraining comparison and at least one appropriate targeted neural selector for SFT if not already present. If a comparator cannot be run faithfully, narrow the claim explicitly; never rename an approximation to fill a table.

Use at least eight new paired training seeds for confirmation at one feasible scale. Increase the from-scratch token budget sufficiently to test whether gains persist beyond early learning, and transfer one frozen mixture to a larger model without retuning it on the larger model's test performance. A second corpus/source pool checks that the result is not unique to one collection. These are separate prospective manifests, admitted only when measured throughput and available allocation cover every row.

Suggested practical margins, chosen prospectively by this design rather than borrowed from literature:

- Pretraining: at least a 1% relative reduction in perplexity against each claimed comparator, equivalent to `delta_NLL < log(0.99)` when delta is ZipMix minus baseline.
- SFT: at least two absolute percentage points of accuracy improvement, with no material regression on the frozen retention suite. Define the retention noninferiority margin before evaluation; one percentage point is a candidate design choice, not a universal standard.

These thresholds formalize a candidate interpretation of “much better”; the versioned protocol must name the accepted margins and primary comparison family before the run. A result that beats zero but not the practical margin supports a smaller improvement claim. If only the targeted suite improves, report targeted adaptation rather than broad generalization.

### Stage 4: RL extension, only after SFT establishes a signal

Hold the reward/verifier, initial checkpoint, optimizer, prompt pool, rollout count, maximum generation length, and update count fixed; alter only prompt-mixture selection. Compare token-proportional prompts, target-aware compression selection, and one appropriate difficulty/curriculum baseline. Freeze target labels and guard against selecting on held-out reward outcomes. Charge failed and discarded rollouts, reward evaluation, and all sampled tokens. This is a new study, not evidence produced by running SFT. No RL result is currently available.

## Fairness and compute accounting

Publish two distinct comparisons:

1. **Equal final training budget:** same final architecture and usable training tokens. Report selector, reference, proxy, and tuning costs separately. This isolates the value of a learned mixture when selection can be amortized.
2. **Equal total budget:** charge compression preprocessing, embedding passes, gradient scoring, reference/proxy training, hyperparameter trials, online probing, final training, and evaluation. Compare each method's attainable performance on its prespecified budget trajectory. Reusing a mixture across several target models gets an explicit amortization count.

Match hardware class within a comparison when possible, and record device model, active device count, wall time, accelerator-hours, central-processing-unit hours, and peak memory. Do not infer total method cost from final-model token counts or from a theoretical floating-point-operation estimate alone. Selecting more duplicates can be a real behavior of a method; disclose unique-token exposure rather than changing duplication constraints only for ZipMix.

Use the same tuning split and a small prespecified tuning allowance for every method. The first scout may use documented defaults plus one feasible alternate, with the same total allowance charged to each method. Freeze grids before inspecting outcomes; report every attempted setting. Match opportunities rather than forcing unrelated algorithms to share the same numerical learning-rate parameter. A method with a larger required selector cost can be strong at equal final tokens and unattractive at equal total cost; report both.

Resource admission is based on measured throughput, not an assumed “five hours per run.” Suggested first-stage envelope: at most 24 available accelerator-hours for the complete pretraining scout including auxiliary work, and a separately recorded envelope for SFT after its own calibration. If the chosen full manifest does not fit, prospectively shrink the common token budget or model size before measurement; never stop healthy admitted cells because one arm looks unpromising. More substantial confirmation needs its own recorded allocation and deadline. Available Stanford Network Analysis Project (SNAP) hardware and current occupancy are runtime facts, not established by this design.

## Statistics and reporting

The experiment, not the token, is the training replication. Record every seed and model run. For each arm report test numerator/denominator, mean, across-seed standard deviation, and the paired difference from each prespecified comparator. Repeated evaluation of one checkpoint is not another training seed. Seed pairing shares initial weights and evaluation items; data order differs only as required by the mixture.

For the three-seed scouts, report descriptive paired differences and uncertainty with an explicit small-sample warning. With three independent pairs an exact two-sided sign-flip test has minimum attainable p-value `2/2^3 = 0.25`; a significance claim at 0.05 cannot be rescued by treating documents or tokens as independent model replicas. Do not add seeds after reading the test merely to cross a threshold.

For confirmation, use a paired seed-level randomization test under an exchangeability null, applying the practical margin to each paired difference. Report raw and Holm-adjusted p-values across the frozen primary method comparisons. Eight pairs permit finer p-values, but do not guarantee adequate power for every effect or multiplicity family. Choose any additional replication from scout development variability before accessing confirmation test outcomes.

Use paired hierarchical resampling over independent training seeds and original test items for confidence intervals; for a claim about new task families, an outer task-family level is required and the small number of families must be disclosed. Never resample token positions independently or split near-duplicate items across bootstrap clusters. If the finite benchmark suite is the estimand, hold its family weights fixed rather than pretending the tasks were a random sample of all possible tasks. Give the exact interval method and resample count.

Write each headline result as `estimate [95% lower, upper] p-val=... (named test; named null or margin)`. For purely descriptive scouts use `p-val=n/a` and say no confirmatory test was run. For perplexity ratios transform loss differences monotonically before presenting intervals. For source/score correlations report each measured correlation with its sample unit and interval; predictive correlation alone cannot establish a benefit from training.

The final report includes all manifest rows: expected, attempted, completed, failed, interrupted, and unattempted. Missing measurements remain missing, not convenient zeros or removed rows. Separate an execution failure from a completed model that performs poorly. If the complete method comparison is not finished, superiority remains unresolved. Report completed-subset diagnostics with their denominator and without replacing the declared full-set result.

## Operational handoff and completion

Keep all experiment code, configurations, frozen source manifests, split hashes, selection weights, logs, checkpoints, model paths, and reports under this experiment's canonical version, or record a linked external-storage exception. Initialize `results.md` and a dated resumable checkpoint at launch. Use a durable queue and resource-aware single owner on SNAP; do not assume the coordinator's chat lifetime is the job lifetime.

Baseline source commits and any local changes must be pinned in the manifest. Verify selectors and samplers with deterministic invariants before training, including no validation gradients reaching optimizer updates, no test reads in selection, and no overlapping source identifiers. These checks are essential scientific validity tests, not superficial tests of formatting.

The user-requested quality-assurance review is one pass using the requested Opus 5.5 model at maximum effort through the approved authenticated command-line client. The executing coordinator owns that dispatch and records the actual model/effort and any unmet requirement. This design does not launch a reviewer or substitute another model.

The immediate deliverable is a correctly labeled complete scout and a reproducible result, favorable or unfavorable. A broader claim requires the prospective confirmation suite above. Neither initializing files, fitting a peaked prior, nor launching a process establishes the scientific goal.
