#!/usr/bin/env python3
"""Analyze the 10-21-2026 Exp 07 diagnostic run (3 selectors x 5 seeds at one explicit learning rate).

Reads a train_exp07.py --mode diagnostic output directory (ledger.json, base/ and per-cell
*_predictions.npz) and writes analysis.json plus a Markdown table. Every statistic is computed
from the run's own receipts; missing cells stay visible. Paired units are training seeds; the
exact sign test and the Student-t interval are the conventions of the original Stage B analysis.
Pooled discordant-item counts across seeds are a descriptive secondary readout (items within a
seed are correlated with each other, so they are not an independent test).
"""
from __future__ import annotations

import argparse
import itertools
import json
from math import comb
from pathlib import Path

import numpy as np
from scipy import stats

ALIGNED = "conditional_aligned"
CONTROLS = ("stratum_uniform", "conditional_shuffled")


def sign_test(differences):
    d = [x for x in differences if x != 0]
    if not d:
        return 1.0
    n, k = len(d), sum(x > 0 for x in d)
    tail = sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return float(min(1.0, 2 * tail))


def t_interval(differences):
    d = np.asarray(differences, dtype=float)
    if len(d) < 2:
        return [None, None]
    half = stats.t.ppf(.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return [float(d.mean() - half), float(d.mean() + half)]


def holm(pvalues):
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def predictions(path, split, key):
    with np.load(path / f"{split}_predictions.npz", allow_pickle=False) as arrays:
        return arrays[key]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None, help="directory for analysis.json and analysis.md (default: --run)")
    args = parser.parse_args()
    out_dir = args.out or args.run
    ledger = json.loads((args.run / "ledger.json").read_text())
    splits = ledger["eval_splits"]
    methods = list(dict.fromkeys(c["method"] for c in ledger["cells"]))
    seeds = sorted({c["seed"] for c in ledger["cells"]})
    done = {(c["method"], c["seed"]): c for c in ledger["cells"] if c["status"] == "complete"}
    missing = [[m, s] for m in methods for s in seeds if (m, s) not in done]
    result = {"run_id": ledger["run_id"], "mode": ledger["mode"], "learning_rate": ledger["identity"]["calibration"]["selected_learning_rate"],
              "methods": methods, "seeds": seeds, "eval_splits": splits, "expected_cells": len(ledger["cells"]),
              "complete_cells": len(done), "missing_cells": missing, "base": {}, "per_method": {}, "contrasts": {}, "runtime": ledger.get("runtime")}
    base_metrics = ledger["base"]["metrics"]
    for split in splits:
        result["base"][split] = {"correct": base_metrics[split]["correct"], "count": base_metrics[split]["count"], "accuracy": base_metrics[split]["accuracy"]}
    lines = [f"# Diagnostic analysis: lr {result['learning_rate']:g}, {len(done)}/{len(ledger['cells'])} cells complete", ""]
    for split in splits:
        base_pred = predictions(args.run / "base", split, "prediction")
        base_corr = predictions(args.run / "base", split, "correct")
        n = len(base_pred)
        lines += [f"## {split} (N={n}; base {base_metrics[split]['correct']}/{n} = {100*base_metrics[split]['accuracy']:.2f}%)", "",
                  "| method | seeds | mean % | sd pp | by seed % | flip vs base | fixes base-wrong % | breaks base-right % |", "|---|---:|---:|---:|---|---:|---:|---:|"]
        acc, corr, pred = {}, {}, {}
        for m in methods:
            cells = [done[(m, s)] for s in seeds if (m, s) in done]
            if not cells:
                lines.append(f"| {m} | 0 | not measured | | | | | |")
                continue
            acc[m] = {c["seed"]: c["metrics"][split]["accuracy"] for c in cells}
            corr[m] = {c["seed"]: predictions(args.run / c["id"], split, "correct") for c in cells}
            pred[m] = {c["seed"]: predictions(args.run / c["id"], split, "prediction") for c in cells}
            values = np.array([acc[m][s] for s in sorted(acc[m])])
            flip = float(np.mean([(pred[m][s] != base_pred).mean() for s in pred[m]]))
            fixes = float(np.mean([(corr[m][s][base_corr == 0] == 1).mean() for s in corr[m]]))
            breaks = float(np.mean([(corr[m][s][base_corr == 1] == 0).mean() for s in corr[m]]))
            seed_dis = float(np.mean([(pred[m][a] != pred[m][b]).mean() for a, b in itertools.combinations(sorted(pred[m]), 2)])) if len(pred[m]) > 1 else None
            result["per_method"].setdefault(m, {})[split] = {
                "seeds": sorted(acc[m]), "by_seed": [acc[m][s] for s in sorted(acc[m])], "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)) if len(values) > 1 else None, "interval95": t_interval(values) if len(values) >= 2 else [None, None],
                "prediction_flip_vs_base": flip, "fixes_base_wrong": fixes, "breaks_base_right": breaks, "same_arm_cross_seed_disagreement": seed_dis}
            sd = f"{100*values.std(ddof=1):.2f}" if len(values) > 1 else "n/a"
            lines.append(f"| {m} | {len(values)} | {100*values.mean():.2f} | {sd} | {', '.join(f'{100*v:.1f}' for v in values)} | {100*flip:.1f}% | {100*fixes:.1f} | {100*breaks:.1f} |")
        lines += ["", "| contrast (aligned minus control) | paired seeds | mean pp [95% t-interval] | positive / ties | exact sign p | Holm p (2 target contrasts) | item-paired SE pp | discordant items win/loss (pooled seeds) | binomial p |", "|---|---:|---|---|---:|---:|---:|---|---:|"]
        pvals, rows = [], []
        for c in CONTROLS:
            if ALIGNED not in acc or c not in acc:
                rows.append((c, None)); pvals.append(1.0); continue
            shared = sorted(set(acc[ALIGNED]) & set(acc[c]))
            d = [acc[ALIGNED][s] - acc[c][s] for s in shared]
            item = np.mean([corr[ALIGNED][s] for s in shared], axis=0) - np.mean([corr[c][s] for s in shared], axis=0)
            wins = int(sum(((corr[ALIGNED][s] == 1) & (corr[c][s] == 0)).sum() for s in shared))
            losses = int(sum(((corr[ALIGNED][s] == 0) & (corr[c][s] == 1)).sum() for s in shared))
            arm_dis = float(np.mean([(pred[ALIGNED][s] != pred[c][s]).mean() for s in shared]))
            entry = {"paired_seeds": shared, "differences": d, "mean": float(np.mean(d)) if d else None, "interval95": t_interval(d),
                     "positive_pairs": int(sum(x > 0 for x in d)), "ties": int(sum(x == 0 for x in d)), "sign_p": sign_test(d),
                     "item_paired_se": float(item.std(ddof=1) / np.sqrt(len(item))) if len(shared) else None,
                     "discordant_win_loss": [wins, losses], "binomial_p": float(stats.binomtest(wins, wins + losses).pvalue) if wins + losses else None,
                     "same_seed_cross_arm_disagreement": arm_dis}
            result["contrasts"].setdefault(c, {})[split] = entry
            pvals.append(entry["sign_p"]); rows.append((c, entry))
        adjusted = holm(pvals)
        for (c, entry), hp in zip(rows, adjusted):
            if entry is None:
                lines.append(f"| {ALIGNED} − {c} | 0 | not measured | | | | | | |"); continue
            if split == splits[0]:
                entry["holm_p"] = hp
            lo, hi = entry["interval95"]
            ci = f"{100*entry['mean']:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}]" if lo is not None else f"{100*entry['mean']:+.2f} [n<2]"
            lines.append(f"| {ALIGNED} − {c} | {len(entry['paired_seeds'])} | {ci} | {entry['positive_pairs']} / {entry['ties']} | {entry['sign_p']:.4f} | {hp:.4f} | {100*entry['item_paired_se']:.3f} | {entry['discordant_win_loss'][0]}/{entry['discordant_win_loss'][1]} | {entry['binomial_p']:.3f} |")
        lines.append("")
    advance_like = all(result["contrasts"].get(c, {}).get(splits[0], {}).get("mean", -1) >= 0.02
                       and result["contrasts"].get(c, {}).get(splits[0], {}).get("positive_pairs", 0) >= 4 for c in CONTROLS)
    result["original_stage_b_advance_rule_on_primary_split"] = {"split": splits[0], "gain_at_least_2pp_and_4_of_5_positive_for_both_controls": bool(advance_like)}
    lines.append(f"Original Stage B advance rule applied descriptively to `{splits[0]}` (≥2 pp mean gain and ≥4/5 positive pairs against both controls): **{'met' if advance_like else 'not met'}**.")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (out_dir / "analysis.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
