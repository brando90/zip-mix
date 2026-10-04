#!/usr/bin/env python3
"""Audit the prospective contrastive supervised fine-tuning (SFT) screen.

Independent units for training comparisons are matched training seeds. Held-out
items are fixed within each seed; they are never treated as extra training runs.
Exact sign tests are exploratory and cannot attain raw p<.25 at n=3.
The two planned target contrasts also receive Holm-adjusted p-values.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy import stats

from common import CONFIG, METHODS, SEEDS, atomic_json, now, stable_hash


def mean_interval(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("estimates must be a finite one-dimensional sequence")
    n = len(values)
    if not n:
        return {"n": 0, "mean": None, "sd": None, "interval95": None}
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if n > 1 else None
    half = float(stats.t.ppf(.975, n - 1) * sd / np.sqrt(n)) if n >= 3 else None
    return {"n": n, "mean": mean, "sd": sd,
            "interval95": [mean - half, mean + half] if half is not None else None}


def paired_estimate(method_values, baseline_values):
    """Map seed->accuracy to paired estimates without filling missing seeds."""
    seeds = sorted(set(method_values) & set(baseline_values))
    differences = np.asarray([method_values[s] - baseline_values[s] for s in seeds])
    summary = mean_interval(differences)
    nonzero = differences[differences != 0]
    pvalue = float(stats.binomtest(int((nonzero > 0).sum()), len(nonzero), p=.5,
                                  alternative="two-sided").pvalue) if len(nonzero) else (1.0 if seeds else None)
    summary.update(paired_seeds=seeds, differences=differences.tolist(),
                   method_mean=float(np.mean([method_values[s] for s in seeds])) if seeds else None,
                   baseline_mean=float(np.mean([baseline_values[s] for s in seeds])) if seeds else None,
                   p_value=pvalue, ties=int((differences == 0).sum()),
                   test="exact two-sided sign test; null P(method accuracy > baseline accuracy)=0.5 among non-ties",
                   multiplicity="unadjusted exploratory comparisons; no confirmatory claim")
    return summary


def holm_adjust(p_values):
    """Adjust a declared family, retaining missing tests and its full size."""
    present = []
    for name, value in p_values.items():
        if value is not None:
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("p-values must be finite probabilities or None")
            present.append((name, value))
    adjusted = dict.fromkeys(p_values)
    running = 0.0
    for rank, (name, value) in enumerate(sorted(present, key=lambda item: item[1])):
        running = max(running, (len(p_values) - rank) * value)
        adjusted[name] = min(1.0, float(running))
    return adjusted


def display(value, multiplier=1):
    if value["mean"] is None:
        return "not measured"
    mean = value["mean"] * multiplier
    interval = value["interval95"]
    if interval is None:
        return f"{mean:.3f} [interval unavailable: n<3]"
    return f"{mean:.3f} [{interval[0] * multiplier:.3f}, {interval[1] * multiplier:.3f}]"


def p_display(value):
    return "n/a" if value["p_value"] is None else f"{value['p_value']:.4g}"


def audit_receipt(path, ledger, cell=None):
    receipt = json.loads((path / "result.json").read_text())
    if receipt["run_id"] != ledger["run_id"]:
        raise ValueError("receipt belongs to another run")
    if cell is not None:
        if (receipt["method"], receipt["seed"], receipt["cell"]) != (cell["method"], cell["seed"], cell["id"]):
            raise ValueError("receipt cell identity differs from ledger")
        if receipt["manifest_id"] != ledger["identity"]["manifest_id"]:
            raise ValueError("receipt data identity differs from ledger")
        if receipt["supervised_tokens"] != CONFIG["steps"] * CONFIG["batch_size"]:
            raise ValueError("supervised-token budget differs from frozen budget")
        if receipt["steps"] != CONFIG["steps"]:
            raise ValueError("training step count differs from frozen budget")
        for name in ("processed_input_tokens", "training_seconds"):
            if not np.isfinite(receipt[name]) or receipt[name] < 0:
                raise ValueError(f"invalid execution accounting: {name}")
        if not isinstance(receipt["infrastructure_resumes"], list):
            raise ValueError("infrastructure recovery history must be a list")
    signatures = {}
    for split in ("target", "retention"):
        metric = receipt["metrics"][split]
        with np.load(path / f"{split}_predictions.npz", allow_pickle=False) as arrays:
            required = ("ids", "correct", "prediction", "gold", "label_nll")
            values = {key: arrays[key] for key in required}
        n = len(values["ids"])
        if n != CONFIG["evaluation_cap_per_split"]:
            raise ValueError(f"{split}: expected {CONFIG['evaluation_cap_per_split']} evaluation items, found {n}")
        if any(value.shape != (n,) for value in values.values()):
            raise ValueError(f"{split}: malformed prediction arrays")
        if len(set(values["ids"].tolist())) != n:
            raise ValueError(f"{split}: duplicate evaluation identities")
        if not np.isfinite(values["label_nll"]).all() or (values["label_nll"] < 0).any():
            raise ValueError(f"{split}: invalid label negative log-likelihood")
        correct = values["prediction"] == values["gold"]
        if not np.array_equal(values["correct"], correct):
            raise ValueError(f"{split}: stored correctness disagrees with predictions and gold labels")
        if metric["count"] != n or metric["correct"] != int(correct.sum()):
            raise ValueError(f"{split}: recorded count differs from predictions")
        if not np.isclose(metric["accuracy"], correct.mean(), atol=1e-12, rtol=0):
            raise ValueError(f"{split}: recorded accuracy differs from predictions")
        if not np.isclose(metric["label_nll"], values["label_nll"].mean(), atol=1e-8, rtol=0):
            raise ValueError(f"{split}: recorded label loss differs from predictions")
        signatures[split] = stable_hash([values["ids"].tolist(), values["gold"].tolist()])
    return receipt, signatures


def analyze(run, output=None):
    run, output = Path(run), Path(output or run)
    output.mkdir(parents=True, exist_ok=True)
    ledger = json.loads((run / "ledger.json").read_text())
    expected = {(method, seed) for method in METHODS for seed in SEEDS}
    by_key = {}
    for cell in ledger["cells"]:
        key = cell["method"], cell["seed"]
        if key not in expected or key in by_key or cell["id"] != f"{cell['method']}__seed{cell['seed']}":
            raise ValueError("ledger contains unknown, duplicated or incorrectly named cells")
        by_key[key] = cell
    if ledger["expected_training_cells"] != len(expected):
        raise ValueError("ledger expected count differs from frozen method-by-seed matrix")
    rows, cells, errors, signatures = {}, [], [], None
    for method in METHODS:
        for seed in SEEDS:
            cell = by_key.get((method, seed), {"id": f"{method}__seed{seed}", "method": method,
                                             "seed": seed, "status": "missing_ledger_row"})
            audited = {"id": cell["id"], "method": method, "seed": seed,
                       "ledger_status": cell["status"], "analysis_status": cell["status"],
                       "target_accuracy": None, "retention_accuracy": None}
            if cell.get("availability") is not None:
                audited["availability"] = cell["availability"]
            if cell.get("availability") == "unavailable_empty_support":
                # Scheduler-compatible terminal status is failed, but no model
                # training attempt or hypothesis outcome exists for this cell.
                audited["analysis_status"] = "unavailable"
            if cell.get("reason") is not None:
                audited["reason"] = cell["reason"]
            if cell["status"] == "complete" and audited["analysis_status"] != "unavailable":
                try:
                    receipt, current_signatures = audit_receipt(run / cell["id"], ledger, cell)
                    if signatures is not None and current_signatures != signatures:
                        raise ValueError("evaluation items or gold labels differ between cells")
                    signatures = current_signatures
                    audited.update(target_accuracy=receipt["metrics"]["target"]["accuracy"],
                                   retention_accuracy=receipt["metrics"]["retention"]["accuracy"],
                                   infrastructure_resumes=receipt["infrastructure_resumes"])
                    rows[(method, seed)] = receipt
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    audited.update(analysis_status="invalid_or_missing_receipt", error=f"{type(exc).__name__}: {exc}")
                    errors.append({"cell": cell["id"], "error": audited["error"]})
            cells.append(audited)
    base, base_status = None, ledger["base"]["status"]
    if base_status == "complete":
        try:
            base, current_signatures = audit_receipt(run / "base", ledger)
            if signatures is not None and current_signatures != signatures:
                raise ValueError("base evaluation items or gold labels differ from training cells")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            base, base_status = None, "invalid_or_missing_receipt"
            errors.append({"cell": "base", "error": f"{type(exc).__name__}: {exc}"})
    counts = dict(Counter(cell["analysis_status"] for cell in cells))
    summary = {"created_at": now(), "run_id": ledger["run_id"],
               "expected_training_cells": len(expected), "status_counts": counts,
               "ledger_status_counts": dict(Counter(cell["ledger_status"] for cell in cells)),
               "verified_training_cells": len(rows), "all_cells_verified": len(rows) == len(expected),
               "base_status": base_status, "base_ledger_status": ledger["base"]["status"],
               "full_procedure_complete": len(rows) == len(expected) and base is not None,
               "cells": cells, "audit_errors": errors, "methods": {}, "contrastive_minus_baseline": {},
               "primary_method": "contrastive_zipmix", "per_method_reference": "source_matched",
               "interval_method": "95% Student-t interval across independent matched training seeds, n>=3; evaluation items held fixed",
               "minimum_two_sided_sign_test_p_with_three_non_tied_pairs": .25,
               "scope": "descriptive mechanism screen; no confirmatory or state-of-the-art verdict",
               "missing_policy": f"All {len(expected)} rows retained. Unverified metrics are null; comparisons use only explicitly identified complete seed pairs.",
               "unavailable_policy": "An arm with no eligible training examples remains unavailable in every seed; this is not a measured failure of its hypothesis.",
               "selector_condition": "Prospective positive target-minus-wrong-target pack alignment, with source-matched and shuffled controls; the Experiment 05 version 2 pool remains frozen."}
    report = ["# Experiment 06: prospective contrastive supervised fine-tuning screen results", "",
              f"Verified training cells: {len(rows)}/{len(expected)}; statuses: {counts}; base: {base_status}.", "",
              "Intervals are descriptive 95% Student-t intervals across training seeds, conditional on the fixed evaluation examples; intervals are omitted below three seeds. Three seeds provide weak power and do not quantify uncertainty over new tasks or corpora.",
              "Raw p-values are exploratory exact two-sided sign tests: null P(method accuracy > baseline accuracy)=0.5 among non-tied seeds. At three non-tied pairs, the minimum p-value is 0.25. The two planned target contrasts also receive Holm adjustment as a fixed family of two. Retention tests remain unadjusted diagnostics. No confirmatory significance or superiority verdict is made.", "",
              "Positive accuracy differences favor the method named first. Differences are percentage points (pp). Intervals are not clipped to feasible accuracy bounds.", "",
              "All methods reuse the fixed Experiment 05 version 2 candidate pool and equal-byte pack scores. Contrastive ZipMix uses positive target-minus-wrong-target alignment; source-matched and shuffled controls test whether its selection adds value beyond source composition and score ordering. This is a prospective mechanism test, not evidence of semantic alignment by construction.", "",
              "Any method with empty selection support is retained as unavailable, with null scores across its seeds. Availability is a preprocessing result, not measured evidence against the method's learning hypothesis; the full declared matrix remains incomplete.", "",
              "| Method | Verified seeds | SciQ accuracy % [95% interval] | Across-seed standard deviation pp | Retention accuracy % [95% interval] |",
              "|---|---:|---|---:|---|"]
    for method in METHODS:
        available = [(seed, rows[(method, seed)]) for seed in SEEDS if (method, seed) in rows]
        entry = {"completed_seeds": [seed for seed, _ in available],
                 "missing_seeds": [seed for seed in SEEDS if (method, seed) not in rows]}
        for split in ("target", "retention"):
            values = {seed: r["metrics"][split]["accuracy"] for seed, r in available}
            baseline = {seed: rows[("source_matched", seed)]["metrics"][split]["accuracy"]
                        for seed in SEEDS if ("source_matched", seed) in rows}
            paired = paired_estimate(values, baseline)
            entry[split] = {"accuracy": mean_interval(list(values.values())), "paired_difference": paired,
                            "paired_seeds": paired["paired_seeds"],
                            "per_seed": [{"seed": s, **r["metrics"][split]} for s, r in available]}
        for key in ("supervised_tokens", "processed_input_tokens", "training_seconds"):
            entry[key] = [r[key] for _, r in available]
        entry["infrastructure_resumes"] = {str(s): r["infrastructure_resumes"] for s, r in available}
        summary["methods"][method] = entry
        sd = entry["target"]["accuracy"]["sd"]
        sd_text = "n/a" if sd is None else f"{100 * sd:.3f}"
        report.append(f"| {method} | {len(available)}/{len(SEEDS)} | {display(entry['target']['accuracy'], 100)} | {sd_text} | {display(entry['retention']['accuracy'], 100)} |")
    report += ["", "Arm-level means: p-val=n/a (no test against an arbitrary accuracy level).", "",
               "| Contrastive ZipMix minus control | Paired seeds | SciQ difference pp [95% interval] | Target raw p-value | Target Holm p-value | Retention difference pp [95% interval] | Retention raw p-value |",
               "|---|---:|---|---:|---:|---|---:|"]
    for baseline in METHODS:
        if baseline == "contrastive_zipmix":
            continue
        entry = {}
        for split in ("target", "retention"):
            z = {s: rows[("contrastive_zipmix", s)]["metrics"][split]["accuracy"] for s in SEEDS if ("contrastive_zipmix", s) in rows}
            b = {s: rows[(baseline, s)]["metrics"][split]["accuracy"] for s in SEEDS if (baseline, s) in rows}
            entry[split] = paired_estimate(z, b)
        summary["contrastive_minus_baseline"][baseline] = entry
    contrasts = summary["contrastive_minus_baseline"]
    adjusted = holm_adjust({baseline: entry["target"]["p_value"] for baseline, entry in contrasts.items()})
    summary["target_comparison_family"] = {
        "primary": "contrastive_zipmix", "controls": list(contrasts),
        "size": len(contrasts), "adjustment": "Holm step-down; raw p-values retained",
        "missing_policy": "Missing tests stay null and remain in the fixed family size; observed tests are conservatively adjusted.",
        "scope": "descriptive only; retention and other diagnostics are outside this family"}
    for baseline, entry in contrasts.items():
        entry["target"]["p_value_holm"] = adjusted[baseline]
        entry["target"]["multiplicity"] = "Holm-adjusted within the two planned target contrasts; no confirmatory claim"
        adjusted_text = "n/a" if adjusted[baseline] is None else f"{adjusted[baseline]:.4g}"
        report.append(f"| {baseline} | {entry['target']['n']}/{len(SEEDS)} | {display(entry['target'], 100)} | {p_display(entry['target'])} | {adjusted_text} | {display(entry['retention'], 100)} | {p_display(entry['retention'])} |")
    if base is not None:
        report += ["", "Base checkpoint: one deterministic evaluation, reported once; no training-seed interval or p-value applies."]
        for split, result in base["metrics"].items():
            report.append(f"- {split}: {result['correct']}/{result['count']} = {100 * result['accuracy']:.3f}%; interval=n/a; p-val=n/a.")
        summary["base"] = base["metrics"]
    else:
        summary["base"] = None
    report += ["", f"| Frozen cell | Ledger status | Analysis status | SciQ correct / {CONFIG['evaluation_cap_per_split']} | Retention correct / {CONFIG['evaluation_cap_per_split']} |", "|---|---|---|---:|---:|"]
    for cell in cells:
        receipt = rows.get((cell["method"], cell["seed"]))
        target = str(receipt["metrics"]["target"]["correct"]) if receipt else "not measured"
        retention = str(receipt["metrics"]["retention"]["correct"]) if receipt else "not measured"
        report.append(f"| {cell['id']} | {cell['ledger_status']} | {cell['analysis_status']} | {target} | {retention} |")
    if errors:
        report += ["", "Receipt audit issues (no affected value contributes to an estimate):"]
        report += [f"- {item['cell']}: {item['error']}" for item in errors]
    unavailable = [cell for cell in cells if cell["analysis_status"] == "unavailable"]
    if unavailable:
        report += ["", "Unavailable frozen cells (no outcome imputation):"]
        report += [f"- {cell['id']}: {cell.get('reason', 'selection support unavailable; see preparation manifest')}" for cell in unavailable]
    report += ["", "CommonsenseQA training examples are in the candidate pool. Its validation result measures untargeted-task retention/transfer, not an unseen training family. The pretrained backbone may already have benchmark exposure.", "",
               f"Equal supervision is {CONFIG['steps'] * CONFIG['batch_size']:,} answer-label tokens per completed cell. Input-token counts, training time and recovery overhead are recorded separately and need not match. The source-matched control matches contrastive ZipMix's total probability for each source and samples uniformly within that source. This screen does not compare against DoGE, DoReMi, LESS or newer mixing methods.", "",
               "Original infrastructure interruptions remain in per-cell recovery receipts. Successful resumption is not evidence of an uninterrupted run.", "",
               "**TLDR-end:** [zip-mix: contrastive fine-tuning screen] " + ("All frozen training cells and the base evaluation have verified artifacts; competitive superiority remains unresolved." if summary["full_procedure_complete"] else f"The frozen procedure remains incomplete, unavailable in part, or has unverified artifacts; all {len(expected)} training cells remain in the denominator."), "",
               "**Snapshot:**", "```text", f"verified_training_cells={len(rows)} expected_training_cells={len(expected)}",
               f"status_counts={counts}", f"base_status={base_status}", "```", ""]
    atomic_json(output / "summary.json", summary)
    temp = output / "results_summary.md.tmp"
    temp.write_text("\n".join(report))
    temp.replace(output / "results_summary.md")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = analyze(args.run, args.output)
    print(json.dumps({"output": str((args.output or args.run) / "results_summary.md"),
                      "status_counts": summary["status_counts"], "audit_errors": summary["audit_errors"]}, indent=2))


if __name__ == "__main__":
    main()
