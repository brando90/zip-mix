# Literature Review: Data Mixture Optimization for LLM Pre-Training

Comprehensive literature review for the Compel-ZipMix project.

---

## 1. Direct Competitors and Positioning

### 1.1 BETR: Targeted Pre-Training (Most Similar Work)
- **Paper:** "Language Models Improve When Pretraining Data Matches Target Tasks"
- **Authors:** David Mizrahi, Anders Boesen Lindbo Larsen, Jesse Allardice, et al.
- **arXiv:** [2507.12466](https://arxiv.org/abs/2507.12466)
- **Scale:** 500+ models, 7B parameters, DCLM-RefinedWeb (24T tokens) and Nemotron-CC (6.3T tokens)
- **Method:** Embed documents + benchmark examples via Arctic-Embed, score by max-rank similarity, train FastText classifier, hard-filter top 10% of tokens.
- **Results:** 2.1x compute multiplier over DCLM-Baseline, 4.7x over unfiltered. Optimal filter rate: `F_opt = 4e-5 * C^0.25`.
- **Key Differentiation from ZipMix:**

| Dimension | BETR | Compel-ZipMix |
|---|---|---|
| Similarity measure | Neural embeddings (Arctic-Embed) -- GPU, model-dependent | NCD via gzip -- CPU-only, model-free |
| Selection type | Binary hard filter (keep/discard) | Soft continuous bucket weights (no data discarded) |
| Corpus partitioning | None -- flat document pool | Compression-ratio buckets (information density proxy) |
| Refinement | None | Optional Group-DRO pass (ZipMix-DRO) |
| Retargeting cost | Re-embed, retrain FastText, re-filter | Single CPU-only ZIP-FIT rescore |
| Theoretical grounding | Empirical embedding similarity | NCD grounded in Kolmogorov complexity |

### 1.2 DoReMi
- **Paper:** "DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining"
- **Authors:** Sang Michael Xie, Hieu Pham, Xuanyi Dong, et al. (Google DeepMind, Stanford)
- **Venue:** NeurIPS 2023
- **arXiv:** [2305.10429](https://arxiv.org/abs/2305.10429)
- **Method:** 3-step pipeline: (1) train 280M reference model, (2) train 280M proxy with Group-DRO to learn domain weights via minimax over excess loss, (3) train 8B model with learned weights.
- **Results:** +6.5pp avg one-shot accuracy on The Pile, 2.6x faster convergence.
- **Critical Limitations:**
  - Starts from uniform prior over coarse hand-defined domains (H_prior)
  - Minimax objective is over-pessimistic, conflating "hard because useful" with "hard because irreducible" (H_max)
  - Requires 2 proxy models (reference + DRO proxy)
  - DoReMi-10k actually HURTS accuracy vs. uniform baseline (per DoGE paper)

### 1.3 DoGE (Criticizes DoReMi)
- **Paper:** "DoGE: Domain Reweighting with Generalization Estimation"
- **Authors:** Simin Fan, Matteo Pagliardini, Martin Jaggi (EPFL)
- **arXiv:** [2310.15393](https://arxiv.org/abs/2310.15393)
- **Method:** Bi-level optimization using gradient alignment scores. Only 1 proxy model needed (vs. DoReMi's 2).
- **Results:** At 684M scale on SlimPajama, DoGE-10k avg perplexity 15.806 vs. DoReMi-10k's 17.172 (DoReMi worse than uniform 16.526).
- **Key critique of DoReMi:** "Dissonance between ideal goal of minimizing avg validation loss and the employed objective which seeks to simply mimic the well-trained model."
- **Note:** Aioli paper shows DoGE itself fails to beat uniform on 5/6 datasets.

### 1.4 ADO (Criticizes DoReMi)
- **Paper:** "Adaptive Data Optimization: Dynamic Sample Selection with Scaling Laws"
- **Authors:** Yiding Jiang, Allan Zhou, Zhili Feng, Sadhika Malladi, J. Zico Kolter (CMU, Stanford, Princeton)
- **arXiv:** [2410.11820](https://arxiv.org/abs/2410.11820)
- **Method:** Online per-domain scaling law estimation + learning potential allocation. Zero proxy models.
- **Results:** At 1.3B on The Pile, ADO 0.590 vs. DoReMi 0.575 avg accuracy. Outperforms on 4/7 tasks.
- **Key critique:** DoReMi requires 760 GPU hours for proxy training, is tokenizer-sensitive, and simpler baselines are competitive.

### 1.5 Aioli
- **Paper:** "Aioli: A Unified Optimization Framework for Language Model Data Mixing"
- **Authors:** Mayee F. Chen, Michael Y. Hu, Nicholas Lourie, Kyunghyun Cho, Christopher Re (Stanford, NYU)
- **Venue:** ICLR 2025
- **arXiv:** [2411.05735](https://arxiv.org/abs/2411.05735)
- **Method:** Unifies all mixing methods into a Linear Mixing Optimization (LMO) framework. Shows existing methods fail because they set mixing-law parameters (A^t matrix) inaccurately. Aioli estimates A^t online during training with zero extra runs.
- **Results:** Only method that beats stratified sampling on ALL 6/6 datasets (avg -0.274 perplexity). DoReMi catastrophically fails on 2 datasets (+5.4 and +6.9 perplexity). DoGE worse on 5/6 datasets.
- **Limitation:** Main experiments at 160M; 1.4B only in appendix on 2 settings. Very small models.
- **Key finding:** Perplexity-downstream correlation is only 0.529, meaning lower perplexity doesn't reliably predict better downstream performance.

---

## 2. Brando's Prior Work (Foundation for ZipMix)

### 2.0 Compel: Compression-Ratio Filtering (Direct Predecessor)
- **Paper:** "Curating High Quality Pretraining Data for Language Models via Compression Ratios"
- **Authors:** Elyas Obbad, Brando Miranda, David Leo Wright Hall, Rylan Schaeffer, Sanmi Koyejo, Percy Liang
- **Venue:** Submitted to ICLR 2026 (rejected; ratings: 6, 4, 4, 2)
- **OpenReview:** [KFafeqE5fe](https://openreview.net/forum?id=KFafeqE5fe)
- **Method:** Compute LZ4 compression ratio CR(d) for each document. Retain only documents with CR in [0.65, 0.80]. No model training, no embeddings — purely statistical. Processes 40K+ docs/sec on 500 CPUs.
- **Results at 1.4B:** FineWeb +1.1pp, FineWeb-EDU +0.7pp, DCLM +0.4pp (macro-avg across 13 benchmarks)
- **Results at 8B:** FineWeb +0.2pp macro / +0.6pp micro avg
- **Key insight:** Compression ratio is a "Goldilocks zone" proxy — low CR = repetitive/boilerplate, high CR = noisy/unnatural, mid-range = information-rich quality text.
- **Limitations (from reviews):** Gains are modest (+0.5-1.1pp); hard thresholds are manually tuned and global; doesn't capture semantic quality. Reviewer ZcWT called gains "marginal."
- **Relevance to ZipMix:** Compel is the foundation. ZipMix extends it from hard-filter to soft bucket-level weighting, adds alignment prior via ZIP-FIT scores, and optionally refines with Group-DRO. Compel's hard-filter is a special case of ZipMix (set extreme bucket weights to 0).

### 2.1 ZIP-FIT / Alignment Coefficient
- **Paper:** "Quantifying the Importance of Data Alignment in Downstream Model Performance"
- **Authors:** Krrish Chawla, Aryan Sahai, Mario DePavia, Sudharsan Sundar, Brando Miranda, Elyas Obbad, Sanmi Koyejo
- **Venue:** ICLR DMLR (2024), ICML DataWorld (2025)
- **arXiv:** [2501.08496](https://arxiv.org/abs/2501.08496)
- **Core idea:** Task2Vec-based alignment coefficient quantifies similarity between training and evaluation data. Strong negative correlation between alignment coefficient and loss/perplexity.
- **Relevance:** Foundation for ZipMix's alignment prior. Demonstrates data alignment matters as much as data quantity.

### 2.2 Diversity Coefficient
- **Paper:** (Diversity coefficient for data characterization)
- **Authors:** Brando Miranda et al.
- **arXiv:** [2306.13840](https://arxiv.org/abs/2306.13840)
- **Relevance:** Provides the complementary "diversity" dimension to alignment. ZipMix's compression buckets capture information diversity; the alignment prior captures task relevance.

---

## 3. Scaling Laws and Contamination

### 3.1 Training on the Test Task Confounds Evaluation
- **Paper:** "Training on the Test Task Confounds Evaluation and Emergence"
- **Authors:** Ricardo Dominguez-Olmedo, Florian E. Dorner, Moritz Hardt
- **Venue:** ICLR 2025 (Oral)
- **arXiv:** [2407.07890](https://arxiv.org/abs/2407.07890)
- **Key finding:** Training on task-relevant data confounds evaluation and makes "emergent capabilities" disappear.
- **Relevance to ZipMix:** Critical for addressing "isn't ZipMix just contamination?" objection. Must distinguish alignment-weighted mixing from direct contamination.

### 3.2 Predicting Emergent Capabilities by Finetuning
- **Paper:** "Predicting Emergent Capabilities by Finetuning"
- **Authors:** Charlie Snell, Eric Wallace, Dan Klein, Sergey Levine
- **arXiv:** [2411.16035](https://arxiv.org/abs/2411.16035)
- **Key finding:** Finetuning on task data shifts emergence thresholds to smaller models. "Emergence laws" can predict capability breakthroughs.
- **Relevance:** ZipMix's alignment prior does a softer version at pre-training time -- may shift when capabilities emerge to smaller/cheaper models.

### 3.3 Quantifying Test Set Contamination (KEY: upper bound for alignment)
- **Paper:** "Quantifying the Effect of Test Set Contamination on Generative Evaluations"
- **Authors:** Rylan Schaeffer, Joshua Kazdan, Baber Abbasi, Ken Ziyu Liu, Brando Miranda, et al.
- **arXiv:** [2601.04301](https://arxiv.org/abs/2601.04301)
- **Key finding:** Even a single test set replica enables models to achieve loss below the irreducible error of the uncontaminated corpus. Scaling laws over contamination dose and model size.
- **Relevance:** This is the "ceiling" of what ZipMix approximates. Confirms benchmark-aligned data is disproportionately valuable at moderate scale.

### 3.4 The Contamination Paradox
- **Paper:** "The Contamination Paradox: Why Test Set Leakage Can Be Both Potent and Negligible"
- **Authors:** Rylan Schaeffer, Joshua Kazdan, et al.
- **Venue:** NeurIPS 2025 LLM Evaluation Workshop
- **Key finding:** Contamination is potent at Chinchilla-optimal scale but negligible at 5x+ overtraining. ZipMix operates in the potent regime.

### 3.5 Investigating Data Contamination for Pre-training
- **Paper:** "Investigating Data Contamination for Pre-training Language Models"
- **Authors:** Minhao Jiang, Ken Ziyu Liu, Ming Zhong, Rylan Schaeffer, et al.
- **arXiv:** [2401.06059](https://arxiv.org/abs/2401.06059) (~106 citations)
- **Key finding:** Even partial contamination at pretraining measurably inflates benchmark performance. Standard n-gram detection is inadequate.

### 3.6 Are Emergent Abilities a Mirage? (NeurIPS 2023 Outstanding Paper)
- **Paper:** "Are Emergent Abilities of Large Language Models a Mirage?"
- **Authors:** Rylan Schaeffer, Brando Miranda, Sanmi Koyejo
- **arXiv:** [2304.15004](https://arxiv.org/abs/2304.15004) (~942 citations)
- **Key finding:** Emergent abilities are artifacts of metric choice, not genuine phase transitions. Under linear metrics, scaling is smooth and predictable.
- **Relevance:** If scaling is smooth, then compression-based alignment should yield predictable improvements across model scales.

### 3.7 Why Has Predicting Downstream Capabilities Remained Elusive? (likely "dark matter" ref)
- **Paper:** "Why Has Predicting Downstream Capabilities of Frontier AI Models with Scale Remained Elusive?"
- **Authors:** Rylan Schaeffer, Hailey Schoelkopf, Brando Miranda, et al.
- **Venue:** NeurIPS 2024 (Datasets & Benchmarks)
- **arXiv:** [2406.04391](https://arxiv.org/abs/2406.04391)
- **Key finding:** Downstream scaling laws break down because prediction requires modeling probability mass fluctuations across incorrect alternatives, not just concentration on correct answers.
- **Relevance:** This is very likely the real paper behind the `schaeffer2025darkmatter` placeholder in zipmix_refs.bib. Justifies ZipMix's 10^3 scaling cap.

### 3.8 Pretraining Scaling Laws for Generative Evaluations (ICLR 2026)
- **Paper:** "Pretraining Scaling Laws for Generative Evaluations of Language Models"
- **Authors:** Rylan Schaeffer, Noam Levi, Brando Miranda, Sanmi Koyejo
- **arXiv:** [2509.24012](https://arxiv.org/abs/2509.24012)
- **Key finding:** Gold-reference log-likelihood is the most stable predictor of pass@k across ~5 orders of magnitude.
- **Relevance:** Provides the scaling law framework ZipMix would be assessed under.

### 3.9 Benchmark Shadows (Caution for ZipMix)
- **Paper:** "Benchmark Shadows: Data Alignment, Parameter Footprints, and Generalization in LLMs"
- **Authors:** Hongjian Zou, Yidan Wang, et al.
- **arXiv:** [2604.07363](https://arxiv.org/abs/2604.07363)
- **Key finding:** Benchmark-aligned data improves narrow metrics but limits broader generalization. Coverage-expanding data yields better generalization.
- **Relevance:** Caution for ZipMix — broad "AGI validation suite" partially mitigates, but must address.

### 3.10 ZIP-FIT (Direct Predecessor)
- **Paper:** "ZIP-FIT: Embedding-Free Data Selection via Compression-Based Alignment"
- **Authors:** Elyas Obbad, Iddah Mlauzi, Brando Miranda, Rylan Schaeffer, et al.
- **arXiv:** [2410.18194](https://arxiv.org/abs/2410.18194)
- **Key finding:** Compression-based alignment for fine-tuning data selection. 85.1% faster cross-entropy convergence, 65.8% faster selection than DSIR.
- **Relevance:** The method ZipMix extends from fine-tuning to pre-training mixture design.

### 3.11 When Scaling Meets LLM Finetuning
- **Paper:** "When Scaling Meets LLM Finetuning: The Effect of Data, Model and Finetuning Method"
- **Authors:** Biao Zhang, Zhongtao Liu, Colin Cherry, Orhan Firat
- **Venue:** ICLR 2024
- **arXiv:** [2402.17193](https://arxiv.org/abs/2402.17193)
- **Key finding:** Finetuning follows multiplicative joint scaling laws. Model scaling > data scaling in impact.
- **Relevance:** Informs ZipMix's 10^3 proxy-to-target scaling constraint.

---

## 4. Data Mixing Laws and Scaling

### 4.1 Data Mixing Laws
- **Paper:** "Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance"
- **Authors:** Ye et al.
- **arXiv:** [2403.16952](https://arxiv.org/abs/2403.16952)
- Discovers functional forms predicting performance as a function of mixture proportions. Nested scaling laws.

### 4.2 BiMix
- **Paper:** "BiMix: A Bivariate Data Mixing Law for Language Model Pretraining"
- **Authors:** Ce Ge, Zhijian Ma, Daoyuan Chen, Yaliang Li, Bolin Ding
- **arXiv:** [2405.14908](https://arxiv.org/abs/2405.14908)
- Jointly models data quantity + mixing proportions. Entropy-driven training-free mixes can match expensive methods.

### 4.3 RegMix
- **Paper:** "RegMix: Data Mixture as Regression for Language Model Pre-training"
- **Authors:** Liu, Zheng et al.
- **Venue:** ICLR 2025 Spotlight
- **arXiv:** [2407.01492](https://arxiv.org/abs/2407.01492)
- Trains 512 models at 1M params to fit regression model. Matches DoReMi at 10% compute cost.

### 4.4 AutoScale
- **Paper:** "AutoScale: Scale-Aware Data Mixing for Pre-Training LLMs"
- **arXiv:** [2407.20177](https://arxiv.org/abs/2407.20177)
- Mixes performing well at small scale may not transfer to larger scales.

### 4.5 Chinchilla (Foundational)
- **Paper:** "Training Compute-Optimal Large Language Models"
- **Authors:** Hoffmann et al. (DeepMind)
- **arXiv:** [2203.15556](https://arxiv.org/abs/2203.15556)
- 20 tokens per parameter rule. Baseline for all data-efficiency work.

---

## 5. Online / Dynamic Data Mixing

### 5.1 ODM (Online Data Mixing)
- **arXiv:** [2312.02406](https://arxiv.org/abs/2312.02406)
- Multi-armed bandit for online mixing. Reaches final perplexity in 19% fewer iterations.

### 5.2 DGA (Dynamic Gradient Alignment)
- **Authors:** Simin Fan, David Grangier, Pierre Ablin
- **arXiv:** [2410.02498](https://arxiv.org/abs/2410.02498)
- Aligns pre-training gradients with target-task gradients dynamically.

### 5.3 Sheared LLaMA
- **Venue:** ICLR 2024
- **arXiv:** [2310.06694](https://arxiv.org/abs/2310.06694)
- Dynamic batch loading adjusting domain composition based on per-domain loss reduction.

---

## 6. Instance-Level Data Selection

### 6.1 DSIR
- **Paper:** "Data Selection for Language Models via Importance Resampling"
- **Authors:** Sang Michael Xie, Shibani Santurkar, Tengyu Ma, Percy Liang
- **arXiv:** [2302.03169](https://arxiv.org/abs/2302.03169)
- Hashed n-gram features for importance resampling. Foundational baseline.

### 6.2 D4
- **arXiv:** [2308.12284](https://arxiv.org/abs/2308.12284)
- Deduplication + diversification using pre-trained embeddings. 20% training efficiency gains.

### 6.3 DsDm (Datamodels)
- **Venue:** ICML 2024
- **arXiv:** [2401.12926](https://arxiv.org/abs/2401.12926)
- 2x compute multiplier via data selection with datamodels.

### 6.4 PDS (Optimal Control)
- **Venue:** ICLR 2025 Oral
- **arXiv:** [2410.07064](https://arxiv.org/abs/2410.07064)
- Data selection as optimal control via Pontryagin's Maximum Principle. 1.8x data reduction.

### 6.5 Rho-1
- **Venue:** NeurIPS 2024 Oral
- **arXiv:** [2404.07965](https://arxiv.org/abs/2404.07965)
- Token-level selective language modeling. 30% absolute improvement on math with 3% of tokens.

### 6.6 Ask-LLM
- **arXiv:** [2402.09668](https://arxiv.org/abs/2402.09668)
- LLM-based quality assessment. Outperforms full-data training while rejecting 90% of data.

---

## 7. Data Curation Benchmarks and Corpora

| Corpus | Size | Key Features | arXiv |
|---|---|---|---|
| DCLM | 240T tokens (raw) | Standardized benchmark for data curation experiments | [2406.11794](https://arxiv.org/abs/2406.11794) |
| FineWeb / FineWeb-Edu | 15T / 1.3T tokens | Classifier-filtered Common Crawl | [2406.17557](https://arxiv.org/abs/2406.17557) |
| The Pile | 800GB, 22 domains | Standard for mixing experiments | [2101.00027](https://arxiv.org/abs/2101.00027) |
| RedPajama | 100T+ tokens | Open dataset with quality signals | [2411.12372](https://arxiv.org/abs/2411.12372) |
| SlimPajama | 627B tokens | Deduplicated RedPajama | [2309.10818](https://arxiv.org/abs/2309.10818) |

---

## 8. Measuring and Characterizing Data

### 8.1 Measuring Data
- **Paper:** "Measuring Data"
- **Authors:** Margaret Mitchell, Alexandra Sasha Luccioni, Nathan Lambert, et al.
- **arXiv:** [2212.05129](https://arxiv.org/abs/2212.05129)
- Framework for systematically quantifying dataset composition along measurable dimensions. ZipMix's compression-ratio buckets are a concrete instance of this paradigm.

---

## 9. Industry Approaches

- **Llama 3** ([2407.21783](https://arxiv.org/abs/2407.21783)): 15T tokens, scaling-law-based mix selection, quality upsampling during annealing.
- **GPT-3** ([2005.14165](https://arxiv.org/abs/2005.14165)): Pioneered non-proportional sampling (higher-quality datasets sampled 2-3x).
- **Gopher** ([2112.11446](https://arxiv.org/abs/2112.11446)): Oversampled books (27%) which improved reading comprehension.

---

## 10. Key Baselines ZipMix Must Compare Against

1. **Uniform / Stratified sampling** -- the default; surprisingly hard to beat consistently (per Aioli)
2. **Token proportional sampling** -- "crucial baseline" per dlwh; weighting by token count
3. **DoReMi** -- the established method; known to fail in some settings
4. **BETR** -- the closest competitor at scale; hard-filter approach
5. **Aioli** -- the current SOTA for consistent improvement; online method
6. **No filtering** -- raw corpus baseline

---

## 11. Open Questions for ZipMix Paper

1. **Contamination objection:** Reviewers will ask "isn't ZipMix just contamination?" (per 2407.07890). Need clear distinction between alignment-weighted mixing and direct contamination.
2. **Where does Compel fit in the FineWeb pipeline?** Before/after which filtering step?
3. **Novelty vs. BETR:** Model-free + soft weighting + compression buckets + DRO refinement. Must be clearly articulated.
4. **DoReMi failure evidence:** Have DoGE results (worse than uniform at 10k steps) and Aioli results (catastrophic on 2 datasets). Need more citable industry complaints.
5. **Token proportional sampling baseline** must be included (per dlwh).
6. **Perplexity-downstream disconnect:** Aioli shows correlation is only 0.529. How does ZipMix handle this?
