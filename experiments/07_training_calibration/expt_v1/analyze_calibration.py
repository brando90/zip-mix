#!/usr/bin/env python3
"""Apply Experiment 07's predeclared rules to a finished run ledger.

--stage calibration writes decision.json: the selected learning rate and the
Stage A gate. --stage stage_b writes analysis.json: paired seed contrasts on
the fresh holdout, the advance rule and the recipe-transfer check. Every rule
here is fixed in PROTOCOL.md before any run; nothing is tuned on outcomes.
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np
from scipy import stats

from calib_common import STAGE_A, STAGE_B, atomic_json, now, stable_hash


def sign_test(differences):
    """Exact two-sided sign test of win probability 0.5 among non-ties."""
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


def stage_a_decision(base, cells, config=STAGE_A):
    """base: {set: {"accuracy", "prediction"}}; cells: list of dicts with
    learning_rate, seed, status and metrics/prediction per set.

    Every declared cell must be complete; an incomplete grid is an
    incomplete experiment, never a decision.
    """
    expected = {(lr, s) for lr in config["learning_rates"] for s in config["seeds"]}
    done = {(c["learning_rate"], c["seed"]): c for c in cells if c["status"] == "complete"}
    missing = sorted(expected - set(done))
    if missing:
        return {"complete": False, "missing_cells": [list(x) for x in missing],
                "gate": {"passed": False, "reason": "incomplete grid; no decision"}}
    t0, r0 = base["calibration_target"]["accuracy"], base["calibration_retention"]["accuracy"]
    rows = []
    for lr in config["learning_rates"]:
        group = [done[(lr, s)] for s in config["seeds"]]
        target = [c["calibration_target"]["accuracy"] for c in group]
        retention = [c["calibration_retention"]["accuracy"] for c in group]
        change = [float(np.mean(np.asarray(c["calibration_target"]["prediction"]) != np.asarray(base["calibration_target"]["prediction"])))
                  for c in group]
        rows.append({"learning_rate": lr, "target_mean": float(np.mean(target)), "target_by_seed": target,
                     "target_sd": float(np.std(target, ddof=1)) if len(target) > 1 else None, "retention_mean": float(np.mean(retention)),
                     "retention_by_seed": retention, "prediction_change_mean": float(np.mean(change)),
                     "seeds_at_or_above_base": int(sum(t >= t0 for t in target)),
                     "retention_eligible": bool(np.mean(retention) >= r0 - config["retention_tolerance"])})
    eligible = [r for r in rows if r["retention_eligible"]]
    selected = None
    if eligible:
        best = max(r["target_mean"] for r in eligible)
        # Ties within one item in 500 go to the larger learning rate: more movement.
        tied = [r for r in eligible if r["target_mean"] >= best - config["tie_tolerance"]]
        selected = max(tied, key=lambda r: r["learning_rate"])
    checks = {"retention_eligible_exists": bool(eligible)}
    if selected:
        checks.update(target_mean_at_or_above_base=selected["target_mean"] >= t0,
                      seeds_at_or_above_base=selected["seeds_at_or_above_base"] >= config["minimum_seeds_at_or_above_base"],
                      recipe_moves_predictions=selected["prediction_change_mean"] >= config["minimum_prediction_change"])
    passed = all(checks.values())
    return {"complete": True, "base": {"calibration_target": t0, "calibration_retention": r0}, "grid": rows,
            "selected_learning_rate": selected["learning_rate"] if selected else None,
            "gate": {"passed": bool(passed), "checks": checks,
                     "reason": "selected recipe matches the unchanged base on fresh calibration validation" if passed
                     else "no grid recipe met every predeclared check; Stage B not admitted"}}


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    return {"differences": d, "mean": float(np.mean(d)), "interval95": t_interval(d), "sign_p": sign_test(d),
            "positive_pairs": int(sum(x > 0 for x in d))}


def stage_b_analysis(base, cells, config=STAGE_B):
    methods, seeds = config["methods"], config["seeds"]
    done = {(c["method"], c["seed"]): c for c in cells if c["status"] == "complete"}
    missing = sorted({(m, s) for m in methods for s in seeds} - set(done))
    if missing:
        return {"complete": False, "missing_cells": [list(x) for x in missing], "advance": False,
                "reason": "incomplete manifest; infrastructure incompletion is not a scientific result"}
    acc = lambda m, name, ss=seeds: [done[(m, s)][name]["accuracy"] for s in ss]
    aligned, controls = methods[0], methods[1:]
    out = {"complete": True, "base": {k: v["accuracy"] for k, v in base.items()}, "methods": {}, "contrasts": {}}
    for m in methods:
        out["methods"][m] = {name: {"mean": float(np.mean(acc(m, name))), "by_seed": acc(m, name)}
                             for name in ("holdout_target", "holdout_retention", "calibration_target")}
    pvalues = []
    for c in controls:
        target = paired(acc(aligned, "holdout_target"), acc(c, "holdout_target"))
        retention = paired(acc(aligned, "holdout_retention"), acc(c, "holdout_retention"))
        primary = paired(acc(aligned, "holdout_target", config["primary_seeds"]), acc(c, "holdout_target", config["primary_seeds"]))
        out["contrasts"][c] = {"target": target, "retention": retention, "target_seeds_0_2": primary}
        pvalues.append(target["sign_p"])
    for c, adjusted in zip(controls, holm(pvalues)):
        out["contrasts"][c]["target"]["holm_p"] = adjusted
    advance = all(out["contrasts"][c]["target"]["mean"] >= config["advance_gain"]
                  and out["contrasts"][c]["target"]["positive_pairs"] >= config["minimum_positive_pairs"]
                  and out["contrasts"][c]["retention"]["mean"] >= -config["retention_tolerance"] for c in controls)
    transfer = out["methods"]["uniform_stratum"]["calibration_target"]["mean"] >= base["calibration_target"]["accuracy"] - config["recipe_transfer_tolerance"]
    out.update(advance=bool(advance), recipe_transfer_ok=bool(transfer),
               reason=("advance to a larger independently held-out confirmation" if advance else "stop scaling this exact formulation")
               + ("" if transfer else "; recipe did not transfer to the Stage B pool, so the screen is exploratory only"))
    return out


def load_run(run):
    ledger = json.loads((run / "ledger.json").read_text())

    def predictions(path, name):
        with np.load(path / f"{name}_predictions.npz", allow_pickle=False) as arrays:
            return arrays["prediction"].tolist()

    sets = list(ledger["base"]["metrics"])
    base = {n: {**ledger["base"]["metrics"][n], "prediction": predictions(run / "base", n)} for n in sets}
    cells = []
    for c in ledger["cells"]:
        row = dict(c)
        if c["status"] == "complete":
            receipt = json.loads((run / c["id"] / "result.json").read_text())
            if receipt["run_id"] != ledger["run_id"]:
                raise ValueError(f"Receipt from another run: {c['id']}")
            for n in sets:
                row[n] = {**receipt["metrics"][n], "prediction": predictions(run / c["id"], n)}
        cells.append(row)
    return ledger, base, cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--stage", choices=("calibration", "stage_b"), required=True)
    args = parser.parse_args()
    ledger, base, cells = load_run(args.run)
    if ledger["stage"] != args.stage:
        raise SystemExit(f"Ledger is a {ledger['stage']} run")
    # A smoke ledger declares a reduced grid; the same rules apply to it.
    present = lambda key: sorted({c[key] for c in cells})
    if args.stage == "calibration":
        config = {**STAGE_A, "learning_rates": present("learning_rate"), "seeds": present("seed")} if ledger["smoke"] else STAGE_A
        result = stage_a_decision(base, cells, config)
        result.update(run_id=ledger["run_id"], smoke=ledger["smoke"], created_at=now())
        result["decision_id"] = stable_hash(result)
        atomic_json(args.run / "decision.json", result)
    else:
        config = {**STAGE_B, "seeds": present("seed"), "primary_seeds": present("seed")} if ledger["smoke"] else STAGE_B
        result = stage_b_analysis(base, cells, config)
        result.update(run_id=ledger["run_id"], smoke=ledger["smoke"], created_at=now())
        atomic_json(args.run / "analysis.json", result)
    print(json.dumps({k: result[k] for k in result if k in ("complete", "selected_learning_rate", "gate", "advance", "recipe_transfer_ok", "reason", "missing_cells")}, indent=2))


if __name__ == "__main__":
    main()
