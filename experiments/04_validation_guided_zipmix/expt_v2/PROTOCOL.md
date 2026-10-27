# Experiment 04 Version 2 — Joint Domain × Compression-Bin Validation Alignment Protocol

**Doc link:** <https://github.com/brando90/zip-mix/blob/main/experiments/04_validation_guided_zipmix/expt_v2/PROTOCOL.md>

**TLDR:** Prospective eight-method, six-seed (`48`-cell) pretraining study testing whether joint `(domain, compression_bin)` validation alignment (`zipmix_domain_bin`) overcomes the `23.5496%` marginal compression-bin capacity ceiling identified in `expt_v1` while isolating within-domain compression stratification (`domain_matched_zipmix`) and fixed-exposure within-stratum ranking (`within_domain_conditional`).

Written 10-16-2026 before any `expt_v2` training outcomes.

## Scientific motivation and hypotheses (locked)

In `expt_v1` (`27/27` cells across seeds `0, 1, 2`), static marginal compression-bin Zip-Mix (`zipmix_static`) reduced biomedical target held-out negative log-likelihood (NLL) relative to `token_proportional` (`-0.01494 [-0.06109, +0.03121]` nats/token, sign-test `p-val=0.25`), `shuffled_zipmix` (`-0.01630 [-0.02746, -0.00515]`, `p-val=0.25`), and compact `doremi` (`-0.09693 [-0.14626, -0.04760]`, `p-val=0.25`), but trailed compact `doge` (`+0.41265 [0.39285, 0.43246]`, `p-val=0.25`). Finite-pool bin-capacity analysis (`expt_v1/results/bin_capacity.md`) proved two structural bottlenecks of `expt_v1`:

1. **Marginal compression-bin capacity ceiling:** Any nonnegative prior over the seven fixed global compression-ratio bins that samples uniformly within each bin can allocate at most `23.5496%` of its probability mass to PubMed blocks (`zipmix_static` allocated `15.2032%`, versus `14.8289%` for `token_proportional`), whereas `doge` reweights the 10 domain labels using exponentiated gradient alignment and allocated `38.99%–40.63%` mass to PubMed (at `2.38×` total floating-point operations and `5.3×` wall-clock time).
2. **Statistical power ceiling at three seeds:** With $n=3$ paired training seeds, an exact two-sided binomial sign test has a minimum possible p-value of $2 \times (0.5)^3 = 0.25$, preventing any non-parametric test from reaching $p < 0.05$.

In the frozen `train.npz` pool (`103,123` blocks across 10 Pile domains), mean positive target compression similarity $\bar{s}_d$ already ranks the two target domains first across all 10 domains (`PubMed Central`: `0.1757`, `PubMed Abstracts`: `0.1624`), ahead of `USPTO Backgrounds` (`0.1489`), `ArXiv` (`0.1418`), `Wikipedia (en)` (`0.1312`), `Pile-CC` (`0.1284`), `StackExchange` (`0.1165`), `FreeLaw` (`0.1105`), `Github` (`0.1009`), and `DM Mathematics` (`0.0641`). However, summing linearly over global compression bins washes out domain identity.

### Locked mechanisms tested in `expt_v2`

1. **Joint `Domain × Compression-Bin` Unit-$z$ Boltzmann Validation Alignment (`zipmix_domain_bin`):**
   For each nonempty observed `(domain, bucket)` cell $g = (d, b)$ with block count $n_g$ and mean clipped target compression similarity $\bar{s}_g = \frac{1}{n_g} \sum_{i \in g} \max(s_i^{\max}, 0)$, standardize $\bar{s}_g$ across observed cells to zero mean and unit population standard deviation:
   $$z_g = \frac{\bar{s}_g - \mu_{\bar{s}}}{\sigma_{\bar{s}} + 10^{-12}}$$
   Weight each cell by the parameter-free unit-temperature Boltzmann factor $W_g = n_g \exp(z_g)$ and sample uniformly within each cell:
   $$p_i = \frac{W_{g(i)}}{n_{g(i)} \sum_{g'} W_{g'}} = \frac{\exp(z_{g(i)})}{\sum_{g'} n_{g'} \exp(z_{g'})}$$
   On the frozen training pool, `zipmix_domain_bin` places `38.7720%` probability mass on PubMed (`63.7397%` effective sample size, total variation `0.2718` from `token_proportional`) with zero reference model, zero proxy model, and `1.00×` final training compute.
2. **Exact Domain-Marginal Control (`domain_matched_zipmix`):**
   Matches `zipmix_domain_bin`'s exact 10-domain marginal distribution ($M_d = \sum_b P_{\text{zipmix\_domain\_bin}}(d, b)$, maximum absolute discrepancy $< 10^{-15}$, `38.7720%` PubMed mass), but samples uniformly across all blocks within each domain $d$ regardless of compression bin. Comparing `zipmix_domain_bin` minus `domain_matched_zipmix` isolates whether within-domain compression-bin weighting improves target or broad NLL at identical domain exposure.
3. **Cell-Alignment Permutation Control (`shuffled_domain_bin`):**
   Permutes the standardized cell scores $z_g$ across observed `(domain, bucket)` cells (`seed + sampling_seed_offset + 197`) before applying $W_g = n_g \exp(z_{\pi(g)})$. Comparing `zipmix_domain_bin` minus `shuffled_domain_bin` isolates true biomedical target alignment from generic cell concentration.
4. **Fixed-Exposure Within-Stratum Conditional Rank Tilt (`within_domain_conditional`):**
   Preserves the exact population `(domain, bucket)` joint distribution $M_{d,b} = n_{d,b} / N$ (`14.8289%` PubMed mass), and tilts block weights inside each cell $g = (d, b)$ of size $m_g$ by $w_i = \exp(\log(4) z_i)$, where $z_i = (R_i - 0.5)/m_g - 0.5$ is the centered average rank of $\max(s_i^{\max}, 0)$ within cell $g$ (`0` if constant). Comparing `within_domain_conditional` minus `token_proportional` tests whether compression similarity ranks better training blocks inside fixed `(domain, bin)` strata when both marginals and their interaction are locked.

## Locked experimental matrix and evaluation splits

- **Arms (8):** `token_proportional`, `zipmix_static`, `zipmix_domain_bin`, `domain_matched_zipmix`, `shuffled_domain_bin`, `within_domain_conditional`, `doremi`, `doge`.
- **Seeds (6):** `0, 1, 2, 3, 4, 5` (`48` cells total). At $n=6$ paired seeds, an exact two-sided binomial sign test has minimum possible p-value $2 \times (0.5)^6 = 0.03125 < 0.05$. Both 6-seed and continuity 3-seed (`0, 1, 2`) summaries are reported.
- **Training & model architecture:** Identical to `expt_v1` (`CausalLM`, 4 layers, 4 heads, `d_model=256`, `d_ff=1024`, `vocab_size=8192`, `seq_len=128`, `batch_size=32`, `final_steps=2048` = `8,388,608` final training tokens per cell; `reference_steps=512`, `proxy_steps=512` for learned methods; `bf16` autocast with `fp32` master weights).
- **Dual held-out evaluation (`eval.npz` + `eval_window2.npz`):**
  1. **Fresh non-overlapping token window (`fresh_target` = 122 blocks, `fresh_broad` = 508 blocks in `eval_window2.npz`):** Token positions `[129:258]` from the held-out test documents in `split_manifest["test"]`. These blocks were already 13-gram decontaminated against `train.npz` during `expt_v1` preparation (which decontaminated against the full 8,192-byte document prefixes) but were never evaluated in `expt_v1`.
  2. **Continuity token window (`target` = 128 blocks, `broad` = 512 blocks in `eval.npz`):** Token positions `[0:129]` from `expt_v1`, retained byte-for-byte for exact comparability with `expt_v1`.

## Primary comparison family and multiplicity

The primary comparison family for `zipmix_domain_bin` on `fresh_target` (and `target`) consists of the six static/learned baseline contrasts:
1. `zipmix_domain_bin minus token_proportional`
2. `zipmix_domain_bin minus zipmix_static`
3. `zipmix_domain_bin minus shuffled_domain_bin`
4. `zipmix_domain_bin minus domain_matched_zipmix`
5. `zipmix_domain_bin minus doremi`
6. `zipmix_domain_bin minus doge`

Report paired-seed means, 95% Student-$t$ confidence intervals (`df = 5`), across-seed standard deviations, raw exact two-sided binomial sign-test p-values, and Holm step-down adjusted p-values across the six-comparison primary family. Also report `within_domain_conditional minus token_proportional` as the locked fixed-exposure within-stratum test.
