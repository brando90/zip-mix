#!/usr/bin/env python3
"""Descriptive, seed-level paired analysis; missing cells remain in denominator.

Primary effect: ZipMix-static minus each comparator in target held-out mean
negative log-likelihood (NLL), in nats/token; negative favors ZipMix. Student-t
95% intervals use paired independent training seeds (n>=3), not token examples.
Exact two-sided sign-test p-values test P(delta<0)=1/2 after omitting exact ties.
These are exploratory comparisons, unadjusted for multiplicity, with weak power
at three seeds. Broad held-out NLL is the prespecified generality diagnostic.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, t

from train import atomic_json, sha256


def mean_interval(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    if not n:
        return {"n": 0, "mean": None, "sd": None, "ci95": None}
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if n > 1 else None
    half = float(t.ppf(0.975, n - 1) * sd / math.sqrt(n)) if n >= 3 else None
    return {"n": n, "mean": mean, "sd": sd, "ci95": [mean - half, mean + half] if half is not None else None}


def paired_summary(zipmix, baseline):
    seeds = sorted(set(zipmix) & set(baseline))
    differences = np.array([zipmix[s] - baseline[s] for s in seeds], dtype=float)
    out = mean_interval(differences)
    nonzero = differences[differences != 0]
    p = float(binomtest(int((nonzero < 0).sum()), len(nonzero), p=0.5, alternative="two-sided").pvalue) if len(nonzero) else 1.0
    out.update(paired_seeds=seeds, differences=differences.tolist(),
               zipmix_mean=float(np.mean([zipmix[s] for s in seeds])) if seeds else None,
               baseline_mean=float(np.mean([baseline[s] for s in seeds])) if seeds else None,
               p_value=p if seeds else None, test="exact two-sided sign test; null P(delta<0)=0.5 among non-ties",
               ties=int((differences == 0).sum()),
               inference="descriptive paired-seed Student-t interval; no multiplicity adjustment")
    return out


def estimate_text(row):
    if row["mean"] is None:
        return "not measured"
    if row["ci95"] is None:
        return f"{row['mean']:.5f} [not estimated: n<3]"
    lo, hi = row["ci95"]
    return f"{row['mean']:.5f} [{lo:.5f}, {hi:.5f}]"


def analyze(output_dir):
    frozen = json.loads((output_dir / "frozen_config.json").read_text())
    cfg = frozen["config"]
    manifest = json.loads((output_dir / "manifest.json").read_text())
    if manifest["fingerprint"] != frozen["fingerprint"]:
        raise ValueError("frozen configuration and manifest differ")
    expected = {(a, s) for a in cfg["arms"] for s in cfg["seeds"]}
    actual = [(c["arm"], c["seed"]) for c in manifest["cells"]]
    if len(actual) != len(set(actual)) or set(actual) != expected or manifest["expected_cells"] != len(expected):
        raise ValueError("manifest does not enumerate exact frozen arm x seed matrix")
    records, counts = {}, {}
    for cell in manifest["cells"]:
        counts[cell["status"]] = counts.get(cell["status"], 0) + 1
        if cell["status"] != "complete":
            continue
        directory = output_dir / f"{cell['arm']}_seed{cell['seed']}"
        result = json.loads((directory / "metrics.json").read_text())
        if (result["status"] != "complete" or result["fingerprint"] != frozen["fingerprint"]
                or result["arm"] != cell["arm"] or result["seed"] != cell["seed"]
                or result["heldout_nll_sha256"] != sha256(directory / "heldout_nll.npz")):
            raise ValueError(f"invalid result binding for {directory}")
        if result["stages"]["final"]["optimizer_tokens"] != cfg["final_steps"] * cfg["batch_size"] * cfg["seq_len"]:
            raise ValueError(f"final training budget mismatch for {directory}")
        # Recompute every reported aggregate from saved per-example losses.
        with np.load(directory / "heldout_nll.npz", allow_pickle=False) as losses:
            for split in ("target", "broad"):
                nll, domains = losses[f"{split}_nll"], losses[f"{split}_domain"]
                recorded = result["metrics"][split]
                if not len(nll) or len(nll) != recorded["examples"] or not np.isfinite(nll).all():
                    raise ValueError("invalid per-example held-out losses")
                domain_means = [float(nll[domains == d].mean()) for d in np.unique(domains)]
                for key, value in (("nll", nll.mean()), ("domain_macro_nll", np.mean(domain_means)),
                                   ("worst_domain_nll", max(domain_means))):
                    if not np.isclose(recorded[key], value, atol=1e-10, rtol=1e-10):
                        raise ValueError(f"aggregate {key} differs from saved losses")
        records[(cell["arm"], cell["seed"])] = result
    summaries, comparisons = {}, {}
    for arm in cfg["arms"]:
        summaries[arm] = {}
        for split in ("target", "broad"):
            for metric in ("nll", "domain_macro_nll", "worst_domain_nll"):
                values = [records[(arm, seed)]["metrics"][split][metric] for seed in cfg["seeds"] if (arm, seed) in records]
                summaries[arm][f"{split}_{metric}"] = mean_interval(values)
    for baseline in cfg["arms"]:
        if baseline == "zipmix_static":
            continue
        comparisons[baseline] = {}
        for split in ("target", "broad"):
            z = {s: records[("zipmix_static", s)]["metrics"][split]["nll"] for s in cfg["seeds"] if ("zipmix_static", s) in records}
            b = {s: records[(baseline, s)]["metrics"][split]["nll"] for s in cfg["seeds"] if (baseline, s) in records}
            comparisons[baseline][split] = paired_summary(z, b)
    complete = counts.get("complete", 0) == len(expected)
    recovered = sum(c.get("recoveries", 0) > 0 for c in manifest["cells"])
    clean = complete and recovered == 0
    summary = {"fingerprint": frozen["fingerprint"], "expected_cells": len(expected),
               "counts": counts, "full_matrix_complete": complete,
               "full_clean_matrix": clean, "cells_with_recovery": recovered, "arms": summaries,
               "zipmix_minus_baseline": comparisons, "unit": "nats per next-token prediction",
               "missing_policy": "No imputation; complete paired subsets explicitly labelled; full denominator retained",
               "claim_limit": "Compact mechanism test; no evidence by itself of broad benchmark or state-of-the-art superiority"}
    atomic_json(output_dir / "analysis.json", summary)
    lines = ["# Compact pretraining mixture comparison", "",
             f"Completed {counts.get('complete', 0)}/{len(expected)} frozen cells; recovered cells: {recovered}; full clean matrix: {clean}.", "",
             "Final training budgets match. Learned-mixture reference/proxy costs are additional, so total compute is not matched.",
             "Target and broad evaluation were held out from all scoring, training, mixture updates and stopping decisions.", "",
             "Negative log-likelihood (NLL) is measured in nats per next-token prediction; lower is better.",
             "Intervals are descriptive 95% paired-seed Student-t intervals (at least 3 seeds), with unadjusted two-sided exact sign tests.",
             "At 3 non-tied seed pairs the smallest possible two-sided sign-test p-value is 0.25; this run cannot establish p<0.05 superiority.",
             "Seed uncertainty is conditional on the fixed corpus and held-out examples; it does not quantify corpus sampling uncertainty.", "",
             "| Arm | Complete seeds | Target NLL [95% interval] | Across-seed standard deviation | Broad NLL [95% interval] |",
             "|---|---:|---|---:|---|"]
    for arm, values in summaries.items():
        target = values["target_nll"]
        sd = "n/a" if target["sd"] is None else f"{target['sd']:.5f}"
        lines.append(f"| {arm} | {target['n']}/{len(cfg['seeds'])} | {estimate_text(target)} | {sd} | {estimate_text(values['broad_nll'])} |")
    lines += ["", "Arm-level intervals are descriptive: p-val=n/a (no arm mean tested against an arbitrary null).", "",
              "| Comparison (ZipMix minus baseline) | Paired seeds | Target difference [95% interval] | p-value | Broad difference [95% interval] |",
              "|---|---:|---|---:|---|"]
    for baseline, values in comparisons.items():
        row = values["target"]
        p = "n/a" if row["p_value"] is None else f"{row['p_value']:.4g}"
        lines.append(f"| {baseline} | {row['n']}/{len(cfg['seeds'])} | {estimate_text(row)} | {p} | {estimate_text(values['broad'])} |")
    lines += ["", "P-values above test the named sign null P(ZipMix NLL < baseline NLL)=0.5 among non-ties; they do not test the magnitude of a mean difference.",
              "No confirmatory superiority claim is made from these exploratory comparisons. Missing or failed cells are not dropped from the declared denominator.", "",
              "| Cell | Status / recoveries | Reference seconds | Proxy seconds | Final seconds | Training floating-point-operation proxy |",
              "|---|---|---:|---:|---:|---:|"]
    for cell in manifest["cells"]:
        key = cell["arm"], cell["seed"]
        result = records.get(key)
        stages = result["stages"] if result else {}
        times = [f"{stages[s]['wall_seconds']:.1f}" if s in stages else "n/a" for s in ("reference", "proxy", "final")]
        cost = str(result["total_training_flops_proxy"]) if result else "n/a"
        lines.append(f"| {key[0]}, seed {key[1]} | {cell['status']} / {cell.get('recoveries', 0)} | {' | '.join(times)} | {cost} |")
    lines += ["", "The operation proxy is 6 × parameter count × gradient tokens + 2 × parameter count × forward-only tokens.",
              "It excludes quadratic attention, optimizer, evaluation and preprocessing operations. Stage wall times include in-stage transfers and logging.",
              "If a cell resumed after interruption, uncheckpointed repeated work is additional and not reconstructed by this cost proxy; failure receipts are retained.",
              "All learned-mixture models use the same size as final models, with fresh initialization for the final stage; this is not a cross-scale transfer result.",
              "direct_zipfit is continuous mean-compression-score sampling, a soft adaptation of ZIP-FIT, not its hard top-ranked selection.",
              "DoGE uses all parameters, a pooled development target, and the official mean-gradient-norm normalization; proxy minibatches are stratified by domain.", "",
              "**TLDR-end:** [zip-mix: compact pretraining] " + ("The frozen matrix completed; estimates remain exploratory and conditional on this compact setup." if complete else "The frozen matrix is incomplete; the table retains every pending, failed or interrupted cell."),
              "", "**Snapshot:**", f"`complete={counts.get('complete', 0)} expected={len(expected)}; analysis.json contains per-arm and paired-seed estimates.`", ""]
    from train import atomic_write
    atomic_write(output_dir / "analysis.md", lambda f: f.write("\n".join(lines).encode()))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.output_dir)
    print(json.dumps({"expected_cells": result["expected_cells"], "counts": result["counts"],
                      "full_clean_matrix": result["full_clean_matrix"]}))


if __name__ == "__main__":
    main()
