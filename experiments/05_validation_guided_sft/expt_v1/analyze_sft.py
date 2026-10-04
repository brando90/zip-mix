#!/usr/bin/env python3
"""Audit and describe the frozen supervised fine-tuning (SFT) screen.

Independent units for training comparisons are matched training seeds. Held-out
items are fixed within each seed; they are never treated as extra training runs.
Exact sign tests are exploratory, unadjusted, and cannot attain p<.25 at n=3.
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
            if cell["status"] == "complete":
                try:
                    receipt, current_signatures = audit_receipt(run / cell["id"], ledger, cell)
                    if signatures is not None and current_signatures != signatures:
                        raise ValueError("evaluation items or gold labels differ between cells")
                    signatures = current_signatures
                    rows[(method, seed)] = receipt
                    audited.update(target_accuracy=receipt["metrics"]["target"]["accuracy"],
                                   retention_accuracy=receipt["metrics"]["retention"]["accuracy"],
                                   infrastructure_resumes=receipt["infrastructure_resumes"])
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
               "cells": cells, "audit_errors": errors, "methods": {}, "zipmix_minus_baseline": {},
               "interval_method": "95% Student-t interval across independent matched training seeds, n>=3; evaluation items held fixed",
               "minimum_two_sided_sign_test_p_with_three_non_tied_pairs": .25,
               "scope": "descriptive mechanism screen; no confirmatory or state-of-the-art verdict",
               "missing_policy": "All 18 rows retained. Unverified metrics are null; comparisons use only explicitly identified complete seed pairs."}
    report = ["# Experiment 05: supervised fine-tuning screen results", "",
              f"Verified training cells: {len(rows)}/{len(expected)}; statuses: {counts}; base: {base_status}.", "",
              "Intervals are descriptive 95% Student-t intervals across training seeds, conditional on the fixed evaluation examples; intervals are omitted below three seeds. Three seeds provide weak power and do not quantify uncertainty over new tasks or corpora.",
              "P-values are unadjusted exploratory exact two-sided sign tests: null P(method accuracy > baseline accuracy)=0.5 among non-tied seeds. At three non-tied pairs, the minimum p-value is 0.25. No confirmatory significance or superiority verdict is made.", "",
              "Positive accuracy differences favor the method named first. Differences are percentage points (pp). Intervals are not clipped to feasible accuracy bounds.", "",
              "| Method | Verified seeds | SciQ accuracy % [95% interval] | Across-seed standard deviation pp | Retention accuracy % [95% interval] |",
              "|---|---:|---|---:|---|"]
    for method in METHODS:
        available = [(seed, rows[(method, seed)]) for seed in SEEDS if (method, seed) in rows]
        entry = {"completed_seeds": [seed for seed, _ in available],
                 "missing_seeds": [seed for seed in SEEDS if (method, seed) not in rows]}
        for split in ("target", "retention"):
            values = {seed: r["metrics"][split]["accuracy"] for seed, r in available}
            baseline = {seed: rows[("sample_proportional", seed)]["metrics"][split]["accuracy"]
                        for seed in SEEDS if ("sample_proportional", seed) in rows}
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
               "| ZipMix minus baseline | Paired seeds | SciQ difference pp [95% interval] | Target p-value | Retention difference pp [95% interval] | Retention p-value |",
               "|---|---:|---|---:|---|---:|"]
    for baseline in METHODS:
        if baseline == "zipmix_static":
            continue
        entry = {}
        for split in ("target", "retention"):
            z = {s: rows[("zipmix_static", s)]["metrics"][split]["accuracy"] for s in SEEDS if ("zipmix_static", s) in rows}
            b = {s: rows[(baseline, s)]["metrics"][split]["accuracy"] for s in SEEDS if (baseline, s) in rows}
            entry[split] = paired_estimate(z, b)
        summary["zipmix_minus_baseline"][baseline] = entry
        report.append(f"| {baseline} | {entry['target']['n']}/{len(SEEDS)} | {display(entry['target'], 100)} | {p_display(entry['target'])} | {display(entry['retention'], 100)} | {p_display(entry['retention'])} |")
    if base is not None:
        report += ["", "Base checkpoint: one deterministic evaluation, reported once; no training-seed interval or p-value applies."]
        for split, result in base["metrics"].items():
            report.append(f"- {split}: {result['correct']}/{result['count']} = {100 * result['accuracy']:.3f}%; interval=n/a; p-val=n/a.")
        summary["base"] = base["metrics"]
    else:
        summary["base"] = None
    report += ["", "| Frozen cell | Ledger status | Analysis status | SciQ correct / 500 | Retention correct / 500 |", "|---|---|---|---:|---:|"]
    for cell in cells:
        receipt = rows.get((cell["method"], cell["seed"]))
        target = str(receipt["metrics"]["target"]["correct"]) if receipt else "not measured"
        retention = str(receipt["metrics"]["retention"]["correct"]) if receipt else "not measured"
        report.append(f"| {cell['id']} | {cell['ledger_status']} | {cell['analysis_status']} | {target} | {retention} |")
    if errors:
        report += ["", "Receipt audit issues (no affected value contributes to an estimate):"]
        report += [f"- {item['cell']}: {item['error']}" for item in errors]
    report += ["", "CommonsenseQA training examples are in the candidate pool. Its validation result measures untargeted-task retention/transfer, not an unseen training family. The pretrained backbone may already have benchmark exposure.", "",
               f"Equal supervision is {CONFIG['steps'] * CONFIG['batch_size']:,} answer-label tokens per completed cell. Input-token counts, training time and recovery overhead are recorded separately and need not match. Compel's long-document compression band is a transfer baseline on short questions. This screen does not compare against DoGE, DoReMi, LESS or newer mixing methods.", "",
               "Original infrastructure interruptions remain in per-cell recovery receipts. Successful resumption is not evidence of an uninterrupted run.", "",
               "**TLDR-end:** [zip-mix: supervised fine-tuning screen] " + ("All frozen training cells and the base evaluation have verified artifacts; competitive superiority remains unresolved." if summary["full_procedure_complete"] else "The frozen procedure remains incomplete or has unverified artifacts; all 18 training cells remain in the denominator."), "",
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
