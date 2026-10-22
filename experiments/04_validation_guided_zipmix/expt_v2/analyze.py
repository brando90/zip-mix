#!/usr/bin/env python3
"""Paired-seed analysis for Experiment 04 v2 across all 6 seeds and both evaluation windows."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, t

from train import atomic_json, atomic_write, sha256


def mean_interval(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    if not n:
        return {"n": 0, "mean": None, "sd": None, "ci95": None}
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if n > 1 else None
    half = float(t.ppf(0.975, n - 1) * sd / math.sqrt(n)) if n >= 3 else None
    return {"n": n, "mean": mean, "sd": sd, "ci95": [mean - half, mean + half] if half is not None else None}


def paired_summary(candidate, baseline, seed_filter=None):
    seeds = sorted(set(candidate) & set(baseline))
    if seed_filter is not None:
        seeds = [s for s in seeds if s in seed_filter]
    differences = np.array([candidate[s] - baseline[s] for s in seeds], dtype=float)
    out = mean_interval(differences)
    nonzero = differences[differences != 0]
    p = float(binomtest(int((nonzero < 0).sum()), len(nonzero), p=0.5, alternative="two-sided").pvalue) if len(nonzero) else (1.0 if seeds else None)
    out.update(paired_seeds=seeds, differences=differences.tolist(),
               candidate_mean=float(np.mean([candidate[s] for s in seeds])) if seeds else None,
               baseline_mean=float(np.mean([baseline[s] for s in seeds])) if seeds else None,
               zipmix_mean=float(np.mean([candidate[s] for s in seeds])) if seeds else None,
               p_value=p if seeds else None,
               test="exact two-sided sign test; null P(delta<0)=0.5 among non-ties",
               ties=int((differences == 0).sum()),
               inference="paired-seed Student-t interval and exact two-sided binomial sign test")
    return out


def holm_adjust(p_values: dict[str, float | None]) -> dict[str, float | None]:
    present = [(k, v) for k, v in p_values.items() if v is not None]
    adjusted = dict.fromkeys(p_values)
    running = 0.0
    m = len(p_values)
    for rank, (name, value) in enumerate(sorted(present, key=lambda item: item[1])):
        running = max(running, (m - rank) * float(value))
        adjusted[name] = min(1.0, float(running))
    return adjusted


def estimate_text(row):
    if row["mean"] is None:
        return "not measured"
    if row["ci95"] is None:
        return f"{row['mean']:.5f} [not estimated: n<3]"
    lo, hi = row["ci95"]
    return f"{row['mean']:.5f} [{lo:.5f}, {hi:.5f}]"


def analyze(output_dir: Path):
    output_dir = Path(output_dir)
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
    available_splits = ["target", "broad"]
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
        cell_splits = ["target", "broad"]
        if "fresh_target" in result["metrics"] and "fresh_broad" in result["metrics"]:
            cell_splits = ["target", "broad", "fresh_target", "fresh_broad"]
            if "fresh_target" not in available_splits:
                available_splits = ["target", "broad", "fresh_target", "fresh_broad"]
        with np.load(directory / "heldout_nll.npz", allow_pickle=False) as losses:
            for split in cell_splits:
                nll, domains = losses[f"{split}_nll"], losses[f"{split}_domain"]
                recorded = result["metrics"][split]
                if not len(nll) or len(nll) != recorded["examples"] or not np.isfinite(nll).all():
                    raise ValueError(f"invalid per-example held-out losses for {split}")
                domain_means = [float(nll[domains == d].mean()) for d in np.unique(domains)]
                for key, value in (("nll", nll.mean()), ("domain_macro_nll", np.mean(domain_means)),
                                   ("worst_domain_nll", max(domain_means))):
                    if not np.isclose(recorded[key], value, atol=1e-10, rtol=1e-10):
                        raise ValueError(f"aggregate {key} differs from saved losses")
        records[(cell["arm"], cell["seed"])] = result

    summaries = {}
    for arm in cfg["arms"]:
        summaries[arm] = {}
        for split in available_splits:
            for metric in ("nll", "domain_macro_nll", "worst_domain_nll"):
                values = [records[(arm, seed)]["metrics"][split][metric]
                          for seed in cfg["seeds"] if (arm, seed) in records and split in records[(arm, seed)]["metrics"]]
                summaries[arm][f"{split}_{metric}"] = mean_interval(values)

    primary_arm = "zipmix_domain_bin" if "zipmix_domain_bin" in cfg["arms"] else "zipmix_static"
    comparisons = {}
    comparisons_seed012 = {}
    for baseline in cfg["arms"]:
        if baseline == primary_arm:
            continue
        comparisons[baseline] = {}
        comparisons_seed012[baseline] = {}
        for split in available_splits:
            z = {s: records[(primary_arm, s)]["metrics"][split]["nll"]
                 for s in cfg["seeds"] if (primary_arm, s) in records and split in records[(primary_arm, s)]["metrics"]}
            b = {s: records[(baseline, s)]["metrics"][split]["nll"]
                 for s in cfg["seeds"] if (baseline, s) in records and split in records[(baseline, s)]["metrics"]}
            comparisons[baseline][split] = paired_summary(z, b)
            comparisons_seed012[baseline][split] = paired_summary(z, b, seed_filter={0, 1, 2})

    # Apply Holm step-down across the primary comparison family on target and fresh_target
    for t_split in [s for s in ("target", "fresh_target") if s in available_splits]:
        raw_p = {b: comparisons[b][t_split]["p_value"] for b in comparisons}
        holm_p = holm_adjust(raw_p)
        for b in comparisons:
            comparisons[b][t_split]["p_value_holm"] = holm_p[b]

    # Secondary comparison: within_domain_conditional minus token_proportional
    within_cond_vs_tok = {}
    if "within_domain_conditional" in cfg["arms"] and "token_proportional" in cfg["arms"]:
        for split in available_splits:
            c_map = {s: records[("within_domain_conditional", s)]["metrics"][split]["nll"]
                     for s in cfg["seeds"] if ("within_domain_conditional", s) in records and split in records[("within_domain_conditional", s)]["metrics"]}
            t_map = {s: records[("token_proportional", s)]["metrics"][split]["nll"]
                     for s in cfg["seeds"] if ("token_proportional", s) in records and split in records[("token_proportional", s)]["metrics"]}
            within_cond_vs_tok[split] = paired_summary(c_map, t_map)

    # Also retain zipmix_static_minus_baseline for backward compatibility
    zipmix_static_comparisons = {}
    if "zipmix_static" in cfg["arms"]:
        for baseline in cfg["arms"]:
            if baseline == "zipmix_static":
                continue
            zipmix_static_comparisons[baseline] = {}
            for split in available_splits:
                z = {s: records[("zipmix_static", s)]["metrics"][split]["nll"]
                     for s in cfg["seeds"] if ("zipmix_static", s) in records and split in records[("zipmix_static", s)]["metrics"]}
                b = {s: records[(baseline, s)]["metrics"][split]["nll"]
                     for s in cfg["seeds"] if (baseline, s) in records and split in records[(baseline, s)]["metrics"]}
                zipmix_static_comparisons[baseline][split] = paired_summary(z, b)

    complete = counts.get("complete", 0) == len(expected)
    recovered = sum(c.get("recoveries", 0) > 0 for c in manifest["cells"])
    clean = complete and recovered == 0
    summary = {
        "fingerprint": frozen["fingerprint"],
        "expected_cells": len(expected),
        "counts": counts,
        "full_matrix_complete": complete,
        "full_clean_matrix": clean,
        "cells_with_recovery": recovered,
        "primary_arm": primary_arm,
        "available_splits": available_splits,
        "arms": summaries,
        "primary_minus_baseline": comparisons,
        "primary_minus_baseline_seeds_0_1_2": comparisons_seed012,
        "zipmix_minus_baseline": zipmix_static_comparisons if zipmix_static_comparisons else comparisons,
        "within_domain_conditional_minus_token_proportional": within_cond_vs_tok,
        "unit": "nats per next-token prediction",
        "missing_policy": "No imputation; complete paired subsets explicitly labelled; full denominator retained",
        "claim_limit": "Compact mechanism test; six-seed non-parametric sign test minimum p-value is 0.03125",
    }
    atomic_json(output_dir / "analysis.json", summary)

    lines = [
        "# Experiment 04 v2 — Joint Domain × Compression-Bin Pretraining Mixture Comparison",
        "",
        f"Completed {counts.get('complete', 0)}/{len(expected)} frozen cells; recovered cells: {recovered}; full clean matrix: {clean}.",
        "",
        "Negative log-likelihood (NLL) is measured in nats per next-token prediction; lower is better.",
        f"Across {len(cfg['seeds'])} independent training seeds, the minimum possible two-sided exact binomial sign-test p-value is {2 * (0.5 ** len(cfg['seeds'])):.5g}.",
        "",
        "## Per-Arm Held-Out Negative Log-Likelihood",
        "",
    ]
    if "fresh_target" in available_splits:
        lines += [
            "| Arm | Complete seeds | Fresh Target W2 NLL [95% CI] | Fresh Broad W2 NLL [95% CI] | Continuity Target W1 NLL [95% CI] | Continuity Broad W1 NLL [95% CI] |",
            "|---|---:|---|---|---|---|",
        ]
        for arm, values in summaries.items():
            t1, b1 = values["target_nll"], values["broad_nll"]
            t2, b2 = values["fresh_target_nll"], values["fresh_broad_nll"]
            lines.append(f"| `{arm}` | {t2['n']}/{len(cfg['seeds'])} | {estimate_text(t2)} | {estimate_text(b2)} | {estimate_text(t1)} | {estimate_text(b1)} |")
    else:
        lines += [
            "| Arm | Complete seeds | Target NLL [95% CI] | Across-seed SD | Broad NLL [95% CI] |",
            "|---|---:|---|---:|---|",
        ]
        for arm, values in summaries.items():
            target = values["target_nll"]
            sd = "n/a" if target["sd"] is None else f"{target['sd']:.5f}"
            lines.append(f"| `{arm}` | {target['n']}/{len(cfg['seeds'])} | {estimate_text(target)} | {sd} | {estimate_text(values['broad_nll'])} |")

    lines += [
        "",
        f"## Primary Paired Comparisons (`{primary_arm}` minus baseline)",
        "",
    ]
    if "fresh_target" in available_splits:
        lines += [
            "| Baseline | Paired seeds | Fresh Target W2 diff [95% CI] | Fresh W2 raw p | Fresh W2 Holm p | Continuity Target W1 diff [95% CI] | W1 raw p | W1 Holm p | Fresh Broad W2 diff [95% CI] |",
            "|---|---:|---|---:|---:|---|---:|---:|---|",
        ]
        for baseline, values in comparisons.items():
            r2, r1, b2 = values["fresh_target"], values["target"], values["fresh_broad"]
            p2 = "n/a" if r2["p_value"] is None else f"{r2['p_value']:.5g}"
            h2 = "n/a" if r2.get("p_value_holm") is None else f"{r2['p_value_holm']:.5g}"
            p1 = "n/a" if r1["p_value"] is None else f"{r1['p_value']:.5g}"
            h1 = "n/a" if r1.get("p_value_holm") is None else f"{r1['p_value_holm']:.5g}"
            lines.append(f"| `{baseline}` | {r2['n']}/{len(cfg['seeds'])} | {estimate_text(r2)} | {p2} | {h2} | {estimate_text(r1)} | {p1} | {h1} | {estimate_text(b2)} |")
    else:
        lines += [
            "| Baseline | Paired seeds | Target diff [95% CI] | raw p-value | Holm p-value | Broad diff [95% CI] |",
            "|---|---:|---|---:|---:|---|",
        ]
        for baseline, values in comparisons.items():
            r1, b1 = values["target"], values["broad"]
            p1 = "n/a" if r1["p_value"] is None else f"{r1['p_value']:.5g}"
            h1 = "n/a" if r1.get("p_value_holm") is None else f"{r1['p_value_holm']:.5g}"
            lines.append(f"| `{baseline}` | {r1['n']}/{len(cfg['seeds'])} | {estimate_text(r1)} | {p1} | {h1} | {estimate_text(b1)} |")

    if within_cond_vs_tok:
        lines += [
            "",
            "## Fixed-Exposure Within-Stratum Comparison (`within_domain_conditional` minus `token_proportional`)",
            "",
        ]
        for split, row in within_cond_vs_tok.items():
            p = "n/a" if row["p_value"] is None else f"{row['p_value']:.5g}"
            lines.append(f"- `{split}` ({row['n']}/{len(cfg['seeds'])} seeds): `{estimate_text(row)}` nats/token, sign-test `p-val={p}`")

    lines += [
        "",
        "**TLDR-end:** [zip-mix: joint domain-bin pretraining v2] " + (
            "All frozen pretraining cells completed and passed deterministic audit."
            if complete else "The frozen pretraining matrix is incomplete; all declared cells remain in the denominator."
        ),
        "",
        "**Snapshot:**",
        f"`complete={counts.get('complete', 0)} expected={len(expected)}; primary_arm={primary_arm}`",
        "",
    ]
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
