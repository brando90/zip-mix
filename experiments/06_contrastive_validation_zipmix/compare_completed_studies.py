#!/usr/bin/env python3
"""Compare the two complete, prospectively frozen fine-tuning studies.

This analysis never launches training and refuses incomplete matrices. It audits
saved predictions, paired evaluation identities and the shared experimental
inputs before producing explicitly exploratory cross-experiment contrasts.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, t


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def load_study(home, version, expected):
    source = home / version
    measured = source / "results/measured"
    manifest = json.loads((source / "data_manifest.json").read_text())
    identity = manifest.pop("manifest_id")
    if stable_hash(manifest) != identity:
        raise ValueError("Data manifest identity mismatch")
    manifest["manifest_id"] = identity
    ledger = json.loads((measured / "ledger.json").read_text())
    summary = json.loads((measured / "summary.json").read_text())
    if not (summary["full_procedure_complete"] and
            summary["verified_training_cells"] == summary["expected_training_cells"] == expected and
            ledger["expected_training_cells"] == expected and ledger["base"]["status"] == "complete"):
        raise ValueError("Both complete matrices and base evaluations are required")
    if summary["run_id"] != ledger["run_id"] or stable_hash(ledger["identity"]) != ledger["run_id"]:
        raise ValueError("Run identity mismatch")
    if ledger["identity"]["manifest_id"] != identity or ledger["identity"]["config"] != manifest["config"]:
        raise ValueError("Run/data binding mismatch")
    for name, value in ledger["identity"]["code"].items():
        if digest(source / name) != value:
            raise ValueError(f"Frozen analysis/training source changed: {name}")
    keys = {(m, s) for m in manifest["methods"] for s in manifest["seeds"]}
    if len(keys) != expected or len(ledger["cells"]) != expected:
        raise ValueError("Unexpected matrix size")
    rows, signature, provenance = {}, None, {}
    all_cells = [(None, measured / "base")]
    for cell in ledger["cells"]:
        key = cell["method"], cell["seed"]
        if key not in keys or key in rows or cell["status"] != "complete":
            raise ValueError("Duplicate, missing or incomplete training cell")
        rows[key] = None
        all_cells.append((cell, measured / cell["id"]))
    base = None
    for cell, folder in all_cells:
        receipt = json.loads((folder / "result.json").read_text())
        if receipt["run_id"] != ledger["run_id"]:
            raise ValueError("Receipt run differs")
        current, values = {}, {}
        for split in ("target", "retention"):
            path = folder / f"{split}_predictions.npz"
            with np.load(path, allow_pickle=False) as a:
                ids, gold, prediction, correct, losses = (a[k] for k in
                    ("ids", "gold", "prediction", "correct", "label_nll"))
                n = manifest["config"]["evaluation_cap_per_split"]
                if any(v.shape != (n,) for v in (ids, gold, prediction, correct, losses)):
                    raise ValueError("Malformed prediction arrays")
                if len(set(ids.tolist())) != n or not np.array_equal(correct, prediction == gold):
                    raise ValueError("Duplicate items or inconsistent accuracy")
                if not np.isfinite(losses).all() or (losses < 0).any():
                    raise ValueError("Invalid saved losses")
                metric = receipt["metrics"][split]
                if metric["count"] != n or metric["correct"] != int(correct.sum()):
                    raise ValueError("Prediction counts differ from receipt")
                if not np.isclose(metric["accuracy"], correct.mean(), atol=1e-12, rtol=0):
                    raise ValueError("Accuracy differs from saved predictions")
                if not np.isclose(metric["label_nll"], losses.mean(), atol=1e-8, rtol=0):
                    raise ValueError("Label loss differs from saved predictions")
                current[split] = stable_hash([ids.tolist(), gold.tolist()])
                values[split] = float(correct.mean())
            provenance[str(path.relative_to(home))] = digest(path)
        if signature is not None and signature != current:
            raise ValueError("Evaluation examples or gold labels differ")
        signature = current
        if cell is None:
            base = values
        else:
            if (receipt["method"], receipt["seed"], receipt["cell"], receipt["manifest_id"]) != (
                    cell["method"], cell["seed"], cell["id"], identity):
                raise ValueError("Cell identity mismatch")
            if receipt["steps"] != manifest["config"]["steps"] or receipt["supervised_tokens"] != (
                    manifest["config"]["steps"] * manifest["config"]["batch_size"]):
                raise ValueError("Training budget mismatch")
            rows[cell["method"], cell["seed"]] = values
        provenance[str((folder / "result.json").relative_to(home))] = digest(folder / "result.json")
    if set(rows) != keys:
        raise ValueError("Missing method-by-seed cell")
    return manifest, ledger, rows, signature, base, provenance


def estimate(values):
    values = np.asarray(values, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("Exactly three finite paired seeds are required")
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    width = float(t.ppf(.975, 2) * sd / np.sqrt(3))
    return {"n": 3, "mean": mean, "sd": sd, "interval95": [mean-width, mean+width],
            "per_seed": values.tolist()}


def contrast(primary, baseline):
    delta = np.asarray(primary) - baseline
    result = estimate(delta)
    n = int(np.count_nonzero(delta))
    result.update(primary_mean=float(np.mean(primary)), baseline_mean=float(np.mean(baseline)),
                  positive=int((delta > 0).sum()), negative=int((delta < 0).sum()),
                  ties=int((delta == 0).sum()),
                  p_value=float(binomtest(int((delta > 0).sum()), n, .5).pvalue) if n else 1.)
    return result


def compare(old_home, new_home, output):
    old = load_study(old_home, "expt_v2", 21)
    new = load_study(new_home, "expt_v1", 9)
    a, b = old[0], new[0]
    if b["parent"]["manifest_id"] != a["manifest_id"] or old[3] != new[3]:
        raise ValueError("Studies do not share the declared inputs and evaluation labels")
    shared = ["train.npz", "target.npz", "retention.npz"] + sorted(k for k in a["files"] if k.startswith("tokenizer/"))
    if any(a["files"][k] != b["files"][k] for k in shared):
        raise ValueError("Data/tokenizer snapshots differ")
    for key in ("model", "model_revision", "seeds", "label_tokens", "pad_token_id"):
        if a[key] != b[key]:
            raise ValueError(f"Study setting differs: {key}")
    changes = {k: [a["config"].get(k), b["config"].get(k)] for k in set(a["config"]) | set(b["config"])
               if a["config"].get(k) != b["config"].get(k)}
    if changes != {"checkpoint_every": [16, 64], "selector_permutation_seed": [None, 1606]}:
        raise ValueError("Unexpected training-setting change")
    if old[1]["identity"]["code"]["train_sft.py"] != new[1]["identity"]["code"]["train_sft.py"]:
        raise ValueError("Trainer implementations differ")
    seeds = a["seeds"]
    result = {"scope": "Exploratory cross-experiment comparison; shared held-out examples, not independent replication",
              "training_cells": 30, "base_evaluations": 2, "independent_training_seeds_per_method": len(seeds),
              "configuration_differences": changes, "evaluation_identity": old[3], "shared_input_files": shared,
              "run_ids": {"05": old[1]["run_id"], "06": new[1]["run_id"]},
              "bases": {"05": old[4], "06": new[4]}, "methods": {}, "contrasts": {},
              "interval": "95% Student-t across the same three paired training seeds, conditional on fixed items",
              "null": "Exact two-sided sign test: P(contrastive accuracy > comparator accuracy)=0.5 among non-tied seeds",
              "family": "Seven exploratory target comparisons against every Experiment 05 method; raw and Holm-adjusted p-values. Retention is an unadjusted diagnostic.",
              "provenance_sha256": {"05": old[5], "06": new[5]}}
    for label, study in (("05", old), ("06", new)):
        for method in study[0]["methods"]:
            result["methods"][f"{label}/{method}"] = {
                split: estimate([study[2][method, s][split] for s in seeds]) for split in ("target", "retention")}
    for method in a["methods"]:
        result["contrasts"][method] = {split: contrast(
            [new[2]["contrastive_zipmix", s][split] for s in seeds],
            [old[2][method, s][split] for s in seeds]) for split in ("target", "retention")}
    running = 0.
    ordered = sorted(result["contrasts"], key=lambda m: result["contrasts"][m]["target"]["p_value"])
    for rank, method in enumerate(ordered):
        row = result["contrasts"][method]["target"]
        running = min(1., max(running, (len(ordered)-rank)*row["p_value"]))
        row["p_value_holm"] = running
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")
    def fmt(e):
        return f"{100*e['mean']:.3f} [{100*e['interval95'][0]:.3f}, {100*e['interval95'][1]:.3f}]"
    lines = ["# Completed fine-tuning studies: exploratory cross-experiment comparison", "",
             "**Doc link:** <https://github.com/brando90/zip-mix/blob/main/experiments/06_contrastive_validation_zipmix/cross_experiment/comparison.md>", "",
             "**TLDR:** All 30 training cells and two base evaluations are verified. These are three paired seeds per method on shared held-out questions, not six independent seeds or a new benchmark replication.", "",
             result["interval"]+". "+result["null"]+". "+result["family"], "",
             "Accuracy is percent; differences and standard deviations are percentage points. Positive differences favor contrastive ZipMix. Means have p-val=n/a; no test against an arbitrary accuracy level was run.", "",
             "| Study / method | Target accuracy [95% interval] | Target standard deviation | Retention accuracy [95% interval] | Retention standard deviation |",
             "|---|---:|---:|---:|---:|"]
    for method, entry in result["methods"].items():
        lines.append(f"| {method} | {fmt(entry['target'])} | {100*entry['target']['sd']:.3f} | {fmt(entry['retention'])} | {100*entry['retention']['sd']:.3f} |")
    lines += ["", "| Contrastive ZipMix minus Experiment 05 comparator | Target difference [95% interval] | Raw p-value | Holm p-value | Retention difference [95% interval] | Raw p-value |",
              "|---|---:|---:|---:|---:|---:|"]
    for method, entry in result["contrasts"].items():
        x, y = entry["target"], entry["retention"]
        lines.append(f"| {method} | {fmt(x)} | {x['p_value']:.3g} | {x['p_value_holm']:.3g} | {fmt(y)} | {y['p_value']:.3g} |")
    lines += ["", "The source-matched and shuffled controls within Experiment 06 remain its two separately declared primary/secondary target contrasts; see that study's report. These cross-experiment comparisons were planned as exploratory before either complete fine-tuning matrix was inspected. Three seeds give a minimum raw sign-test p-value of 0.25; intervals and sign tests address different summaries and should not be read as interchangeable significance decisions.", "",
              "The model revision, candidate order, tokenizer, evaluation arrays, initialization seeds and learning settings match. Checkpoint cadence differs (16 versus 64 updates), so elapsed-time differences cannot identify selector efficiency. The same nominal seed pairs initial conditions; different mixtures intentionally lead to different sampled questions. Fine-tuning supervises one answer label per question; these results do not establish free-form reasoning or reinforcement-learning improvement.", "",
              "Every individual seed, source hash and comparison is in `comparison.json`. No model fitting, additional evaluation, checkpoint selection or training rerun is performed by this analysis.", ""]
    (output / "comparison.md").write_text("\n".join(lines))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-home", type=Path, default=Path(__file__).parents[1] / "05_validation_guided_sft")
    parser.add_argument("--new-home", type=Path, default=Path(__file__).parent)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "cross_experiment")
    args = parser.parse_args()
    compare(args.old_home, args.new_home, args.output)
