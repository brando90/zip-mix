#!/usr/bin/env python3
"""Render complete audited pretraining results without changing training code."""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "token_proportional": "Population sampling", "uniform_domain": "Uniform domains",
    "uniform_bucket": "Uniform compression bins", "compel_filter": "Compel adaptation",
    "zipmix_static": "Zip-Mix", "direct_zipfit": "Direct mean-score sampling",
    "shuffled_zipmix": "Shuffled Zip-Mix", "doremi": "DoReMi adaptation",
    "doge": "DoGE adaptation",
}


def interval_text(row, transform=lambda x: x):
    return (f"{transform(row['mean']):.4f} "
            f"[{transform(row['ci95'][0]):.4f}, {transform(row['ci95'][1]):.4f}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.run / "analysis.json").read_text())
    manifest = json.loads((args.run / "manifest.json").read_text())
    if not summary["full_matrix_complete"] or summary["counts"] != {"complete": 27}:
        raise ValueError("Render only the complete audited 27-cell screen")
    if summary["fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Analysis and manifest identities differ")
    arms = summary["arms"]
    costs = {}
    for cell in manifest["cells"]:
        record = json.loads((args.run / f"{cell['arm']}_seed{cell['seed']}" / "metrics.json").read_text())
        if record["fingerprint"] != summary["fingerprint"] or record["status"] != "complete":
            raise ValueError("Unbound or incomplete metric record")
        costs.setdefault(cell["arm"], []).append(record["total_stage_wall_seconds"])
    for name in LABELS:
        if arms[name]["target_nll"]["n"] != 3 or len(costs[name]) != 3:
            raise ValueError("Three verified seeds required for every method")
    controls = ["token_proportional", "shuffled_zipmix"]
    criteria = {}
    for name in controls:
        pair = summary["zipmix_minus_baseline"][name]
        criteria[name] = {
            "mean_target_reduction_at_least_0.02": pair["target"]["mean"] <= -.02,
            "target_direction_consistent_all_seeds": all(x < 0 for x in pair["target"]["differences"]),
            "broad_increase_at_most_0.02": pair["broad"]["mean"] <= .02,
            "target_difference": pair["target"]["mean"],
            "broad_difference": pair["broad"]["mean"],
        }
    passed = all(all(v[k] for k in ("mean_target_reduction_at_least_0.02",
                                    "target_direction_consistent_all_seeds", "broad_increase_at_most_0.02"))
                 for v in criteria.values())
    decision = {"criteria": criteria, "screening_heuristic_met": passed,
                "threshold_source": "expt_v1/PROTOCOL.md; agent-suggested screening heuristic, not a significance test",
                "broad_interpretation": "Apply the broad-loss tolerance relative to both primary sampling controls",
                "inference": "No optimality, confirmatory significance, or state-of-the-art verdict"}
    (args.run / "screening_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    lines = ["# Complete pretraining screen: loss and perplexity", "",
             "All 27 frozen training cells have audited outcomes. Three paired seeds per method; intervals condition on the fixed corpus and evaluation items.", "",
             "Negative log likelihood (NLL) is in nats per predicted token. Perplexity is exp(mean seed NLL), with endpoints transformed from its 95% Student-t interval; it is not the arithmetic mean of seed perplexities. Absolute means have p-val=n/a: no arbitrary level null was tested.", "",
             "Domain Reweighting with Minimax Optimization (DoReMi) and Domain Reweighting with Generalization Estimation (DoGE) are compact adaptations using the same-size proxy and final models. Their extra reference/proxy costs are included in the measured stage time.", "",
             "| Method | Target NLL [95% interval] | Target perplexity [95% interval] | Seed NLL standard deviation | Broad perplexity [95% interval] | Mean training-stage seconds |",
             "|---|---|---|---:|---|---:|"]
    for name, label in LABELS.items():
        row = arms[name]
        lines.append(f"| {label} | {interval_text(row['target_nll'])} | {interval_text(row['target_nll'], math.exp)} | "
                     f"{row['target_nll']['sd']:.5f} | {interval_text(row['broad_nll'], math.exp)} | {np.mean(costs[name]):.2f} |")
    lines += ["", "The prespecified descriptive screening heuristic is " + ("met" if passed else "not met") +
              ". It requires at least 0.02 lower mean target NLL than population and shuffled controls, a lower target loss in every paired seed, and no more than 0.02 broad-loss increase relative to either control. See screening_decision.json for each component and analysis.md for every paired interval and exact sign-test p-value.", "",
              "All comparisons are exploratory and unadjusted for multiple comparisons. At three non-tied pairs the minimum two-sided sign-test p-value is 0.25. Equal final predicted-token budgets do not imply equal total compute.", "",
              "![Paired seed loss differences](paired_effects.png)", "",
              ("**Zip-Mix " + ("met" if passed else "did not meet") + " the prespecified descriptive screening criterion.** Points are individual seed differences; bars are 95% paired-seed Student-t intervals. Negative values favor Zip-Mix; intervals are conditional on the fixed data and do not establish broad superiority."), ""]
    (args.run / "perplexity_and_screening.md").write_text("\n".join(lines))
    comparators = [k for k in LABELS if k != "zipmix_static"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    for ax, split in zip(axes, ("target", "broad")):
        for y, name in enumerate(comparators):
            pair = summary["zipmix_minus_baseline"][name][split]
            lo, hi = pair["ci95"]
            ax.plot([lo, hi], [y, y], color="#177E89", linewidth=2.5)
            ax.scatter(pair["differences"], np.full(3, y), color="#8D99AE", s=22, zorder=3)
            ax.scatter([pair["mean"]], [y], color="#177E89", marker="D", s=34, zorder=4)
        ax.axvline(0, color="#555555", linewidth=1, linestyle="--")
        ax.set_title("Biomedical target" if split == "target" else "Broad non-target domains", loc="left")
        ax.set_xlabel("Zip-Mix minus comparator NLL (nats/token)\nNegative favors Zip-Mix")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(np.arange(len(comparators)), [LABELS[k] for k in comparators])
    axes[0].invert_yaxis()
    fig.suptitle("Complete 27-model pretraining screen: paired seed differences", fontsize=13)
    fig.subplots_adjust(left=.24, right=.98, top=.86, bottom=.18, wspace=.16)
    fig.savefig(args.run / "paired_effects.png", dpi=180)
    fig.savefig(args.run / "paired_effects.pdf")
    plt.close(fig)
    print(json.dumps({"screening_heuristic_met": passed, "criteria": criteria}))


if __name__ == "__main__":
    main()
