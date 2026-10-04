#!/usr/bin/env python3
"""Plot an actual, complete summary.json emitted by analyze_sft.py.

Example, after the full frozen run has completed and passed the analyzer:
  python plot_results.py --summary /path/to/run/summary.json

Writes accuracy_results.png, accuracy_results.pdf and a provenance receipt beside
summary.json unless --output-dir is supplied. --check-only validates without
rendering. Incomplete, unaudited, or inconsistent summaries fail closed. No
training, analysis-source edits, or imputed values are performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
from scipy.stats import binomtest, t

METHOD_LABELS = {
    "contrastive_zipmix": "Contrastive ZipMix",
    "source_matched": "Source-matched control",
    "contrastive_shuffled": "Shuffled contrast control",
}
SEEDS = (0, 1, 2)
SPLITS = ("target", "retention")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def near(left, right):
    try:
        return bool(np.allclose(left, right, rtol=0, atol=1e-10))
    except (TypeError, ValueError):
        return False


def validate_summary(summary):
    """Check the complete analyzed matrix and recompute plotted seed statistics."""
    require(summary.get("full_procedure_complete") is True, "Full procedure is incomplete; nothing will be plotted")
    require(summary.get("all_cells_verified") is True, "Not every cell passed the analyzer")
    require(summary.get("audit_errors") == [], "Analyzer audit errors are present or absent from the schema")
    require(summary.get("base_status") == "complete", "The single base evaluation is not verified")
    require(set(summary.get("methods", {})) == set(METHOD_LABELS), "Expected all three frozen contrastive methods")
    expected = {(method, seed) for method in METHOD_LABELS for seed in SEEDS}
    require(summary.get("expected_training_cells") == len(expected), "Incorrect declared cell count")
    require(summary.get("verified_training_cells") == len(expected), "Incorrect verified cell count")
    require(summary.get("status_counts") == {"complete": len(expected)}, "Non-complete scientific cell statuses remain")
    run_id = summary.get("run_id")
    require(isinstance(run_id, str) and len(run_id) == 64 and all(c in "0123456789abcdef" for c in run_id),
            "Missing or malformed frozen run identity")
    cells = summary.get("cells", [])
    require(len(cells) == len(expected), "Missing or duplicated cell rows")
    by_key = {}
    for cell in cells:
        key = cell.get("method"), cell.get("seed")
        require(key in expected and key not in by_key, "Unexpected or duplicated cell identity")
        require(cell.get("analysis_status") == "complete" and cell.get("ledger_status") == "complete",
                f"Unverified cell: {key}")
        require(cell.get("id") == f"{key[0]}__seed{key[1]}", "Malformed cell identifier")
        by_key[key] = cell
    require(set(by_key) == expected, "Frozen cell matrix is incomplete")
    parsed, budgets = {}, set()
    for method in METHOD_LABELS:
        entry = summary["methods"][method]
        require(sorted(entry.get("completed_seeds", [])) == list(SEEDS), f"Missing seeds: {method}")
        require(entry.get("missing_seeds") == [], f"Missing seed list is nonempty: {method}")
        tokens = entry.get("supervised_tokens", [])
        require(len(tokens) == len(SEEDS) and all(isinstance(n, int) and n > 0 for n in tokens),
                f"Missing supervision accounting: {method}")
        budgets.update(tokens)
        parsed[method] = {}
        for split in SPLITS:
            metric = entry[split]
            seed_rows = metric.get("per_seed", [])
            require(len(seed_rows) == len(SEEDS), f"Missing per-seed points: {method}/{split}")
            require(sorted(row.get("seed") for row in seed_rows) == list(SEEDS), "Duplicate or missing metric seeds")
            points = []
            for row in sorted(seed_rows, key=lambda item: item["seed"]):
                n, correct, accuracy = row.get("count"), row.get("correct"), row.get("accuracy")
                require(n == 500 and isinstance(correct, int) and 0 <= correct <= n,
                        f"Invalid verified evaluation count: {method}/{split}")
                require(near(accuracy, correct / n), f"Accuracy and count disagree: {method}/{split}")
                require(near(accuracy, by_key[(method, row["seed"])][f"{split}_accuracy"]),
                        f"Cell and method accuracies disagree: {method}/{split}")
                points.append(float(accuracy))
            points = np.asarray(points)
            mean, sd = float(points.mean()), float(points.std(ddof=1))
            half = float(t.ppf(.975, len(SEEDS) - 1) * sd / math.sqrt(len(SEEDS)))
            interval = [mean - half, mean + half]
            aggregate = metric["accuracy"]
            require(aggregate.get("n") == len(SEEDS), f"Incorrect independent-unit count: {method}/{split}")
            require(near(aggregate.get("mean"), mean) and near(aggregate.get("sd"), sd)
                    and near(aggregate.get("interval95"), interval),
                    f"Saved 95% training-seed statistics disagree with seed points: {method}/{split}")
            parsed[method][split] = {"points": points, "mean": mean, "interval95": interval}
    require(len(budgets) == 1, "Supervised-token budgets differ across training cells")
    base = summary.get("base")
    require(isinstance(base, dict), "Missing verified base evaluation")
    for split in SPLITS:
        metric = base[split]
        n, correct = metric.get("count"), metric.get("correct")
        require(n == 500 and isinstance(correct, int) and 0 <= correct <= n, "Invalid base evaluation count")
        require(near(metric.get("accuracy"), correct / n), "Base accuracy disagrees with its verified count")
    require(budgets == {2048}, "Supervision differs from the frozen 2,048-token budget")
    contrasts = summary.get("contrastive_minus_baseline", {})
    controls = set(METHOD_LABELS) - {"contrastive_zipmix"}
    require(set(contrasts) == controls, "Expected both planned target contrasts")
    family = summary.get("target_comparison_family", {})
    require(family.get("size") == 2 and set(family.get("controls", [])) == controls
            and family.get("primary") == "contrastive_zipmix", "Incorrect declared target-comparison family")
    raw = {}
    for control in controls:
        for split in SPLITS:
            value = contrasts[control][split]
            primary_points = parsed["contrastive_zipmix"][split]["points"]
            control_points = parsed[control][split]["points"]
            differences = primary_points - control_points
            mean, sd = float(differences.mean()), float(differences.std(ddof=1))
            half = float(t.ppf(.975, 2) * sd / math.sqrt(3))
            nonzero = differences[differences != 0]
            p = float(binomtest(int((nonzero > 0).sum()), len(nonzero), p=.5).pvalue) if len(nonzero) else 1.0
            require(value.get("paired_seeds") == list(SEEDS) and value.get("n") == 3,
                    "Missing planned paired seeds")
            require(near(value.get("differences"), differences) and near(value.get("mean"), mean)
                    and near(value.get("sd"), sd) and near(value.get("interval95"), [mean-half, mean+half])
                    and near(value.get("method_mean"), primary_points.mean())
                    and near(value.get("baseline_mean"), control_points.mean())
                    and near(value.get("p_value"), p), "Paired contrast does not reproduce seed data")
            if split == "target":
                raw[control] = p
    running = 0.0
    for rank, control in enumerate(sorted(raw, key=raw.get)):
        running = min(1.0, max(running, (2-rank)*raw[control]))
        require(near(contrasts[control]["target"].get("p_value_holm"), running),
                "Holm-adjusted target p-value does not reproduce the fixed two-test family")
    return parsed, next(iter(budgets))


def save_figure_atomic(figure, path, file_format):
    descriptor, name = tempfile.mkstemp(prefix=path.stem + ".", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        figure.savefig(name, format=file_format, dpi=220, facecolor="white")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def render(summary_path, output_dir=None, check_only=False):
    summary_path = Path(summary_path)
    content = summary_path.read_bytes()
    summary = json.loads(content)
    parsed, token_budget = validate_summary(summary)
    if check_only:
        return {"valid": True, "run_id": summary["run_id"], "verified_training_cells": 9,
                "rendered": False}
    # Matplotlib is imported only after every validation gate passes.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    output_dir = Path(output_dir) if output_dir else summary_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titlesize": 13, "axes.labelsize": 11,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    figure, axes = plt.subplots(1, 2, figsize=(13.8, 6.7), sharey=True)
    figure.subplots_adjust(left=.245, right=.975, top=.745, bottom=.40, wspace=.16)
    markers = ("o", "s", "^")
    offsets = (-.17, 0, .17)
    primary, neutral, control = "#087F8C", "#455565", "#8A6E9F"
    titles = {"target": "SciQ · target accuracy\n500 fixed test items",
              "retention": "CommonsenseQA · retention accuracy\n500 fixed validation items"}
    for axis, split in zip(axes, SPLITS):
        bounds = [100 * summary["base"][split]["accuracy"]]
        for y, method in enumerate(METHOD_LABELS):
            values = parsed[method][split]
            color = primary if method == "contrastive_zipmix" else (control if method == "contrastive_shuffled" else neutral)
            lo, hi = np.asarray(values["interval95"]) * 100
            mean = values["mean"] * 100
            bounds.extend([lo, hi])
            axis.errorbar(mean, y, xerr=[[mean - lo], [hi - mean]], fmt="D", markersize=6.5,
                          capsize=5, elinewidth=2, color=color, zorder=4)
            for seed, marker, offset in zip(SEEDS, markers, offsets):
                value = values["points"][seed] * 100
                bounds.append(value)
                axis.scatter(value, y + offset, marker=marker, s=28, facecolors="white",
                             edgecolors=color, linewidths=1.25, zorder=5)
            if method == "contrastive_zipmix":
                axis.axhspan(y - .42, y + .42, color=primary, alpha=.07, zorder=0)
        base_accuracy = 100 * summary["base"][split]["accuracy"]
        axis.axvline(base_accuracy, color="#777777", linestyle="--", linewidth=1.2, zorder=1)
        axis.set_title(titles[split], pad=13)
        axis.set_xlabel("Accuracy (%) · higher is better")
        axis.set_yticks(range(len(METHOD_LABELS)), list(METHOD_LABELS.values()))
        axis.set_ylim(len(METHOD_LABELS) - .55, -.55)
        span = max(bounds) - min(bounds)
        padding = max(1.0, span * .09)
        axis.set_xlim(min(bounds) - padding, max(bounds) + padding)
        axis.grid(axis="x", color="#E1E6EA", linewidth=.7)
        axis.set_axisbelow(True)
        axis.tick_params(axis="y", length=0, pad=12)
        for spine in ("top", "right", "left"):
            axis.spines[spine].set_visible(False)
        axis.spines["bottom"].set_color("#CAD1D8")
    handles = [Line2D([], [], marker=marker, linestyle="none", markerfacecolor="white",
                      markeredgecolor=neutral, label=f"Seed {seed}") for seed, marker in zip(SEEDS, markers)]
    handles.extend([Line2D([], [], color=neutral, marker="D", label="Mean + 95% training-seed interval"),
                    Line2D([], [], color="#777777", linestyle="--", label="Base checkpoint · evaluated once")])
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .855),
                  ncol=5, frameon=False, fontsize=9.5, handlelength=1.7, columnspacing=1.2)
    figure.suptitle("Contrastive ZipMix: compact three-seed mechanism screen", x=.52, y=.965,
                   fontsize=17, fontweight="bold", color="#1C2933")
    figure.text(.52, .912, f"3 methods × 3 matched training seeds · {token_budget:,} supervised answer-label tokens per run",
                ha="center", fontsize=11, color="#485762")
    figure.text(.245, .29, "Planned target contrasts · percentage points (pp), paired 95% intervals",
                fontsize=10, fontweight="bold", color="#1C2933")
    for row, control in enumerate(("source_matched", "contrastive_shuffled")):
        contrast = summary["contrastive_minus_baseline"][control]["target"]
        lo, hi = np.asarray(contrast["interval95"]) * 100
        text = (f"Contrastive − {METHOD_LABELS[control]}: {100*contrast['mean']:+.2f} pp "
                f"[{lo:+.2f}, {hi:+.2f}]; raw p={contrast['p_value']:.3g}; "
                f"Holm p={contrast['p_value_holm']:.3g}")
        figure.text(.245, .245-row*.045, text, fontsize=10, color="#485762")
    figure.text(.245, .155, "Holm adjustment covers the two planned target contrasts; retention is an unadjusted diagnostic.",
                fontsize=9, color="#485762")
    figure.text(.245, .115, "Intervals: 95% Student-t across training seeds; evaluation items are held fixed. Intervals are not clipped.",
                fontsize=9, color="#485762")
    figure.text(.245, .08, "Descriptive screening only. Three pairs cannot yield an exact two-sided sign-test p-value below 0.25.",
                fontsize=9, color="#485762")
    figure.text(.245, .045, "This compact comparison does not establish broad generalization or state-of-the-art superiority.",
                fontsize=9, color="#485762")
    png, pdf = output_dir / "accuracy_results.png", output_dir / "accuracy_results.pdf"
    save_figure_atomic(figure, png, "png")
    save_figure_atomic(figure, pdf, "pdf")
    plt.close(figure)
    receipt = {"run_id": summary["run_id"], "summary_sha256": hashlib.sha256(content).hexdigest(),
               "summary_file": summary_path.name, "verified_training_cells": 9,
               "training_seeds": list(SEEDS), "methods": list(METHOD_LABELS),
               "interval_definition": "95% Student-t interval across 3 training seeds; fixed evaluation items",
               "target_comparison_family": summary["target_comparison_family"],
               "target_contrasts": {m: summary["contrastive_minus_baseline"][m]["target"]
                                    for m in ("source_matched", "contrastive_shuffled")},
               "files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (png, pdf)}}
    receipt_path = output_dir / "accuracy_results_provenance.json"
    temporary = receipt_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    temporary.replace(receipt_path)
    return {"valid": True, "rendered": True, "png": str(png), "pdf": str(pdf),
            "provenance": str(receipt_path), "run_id": summary["run_id"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    try:
        result = render(args.summary, args.output_dir, args.check_only)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"Refusing to plot unverified or incomplete results: {exc}") from exc
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
