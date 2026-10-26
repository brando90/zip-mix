#!/usr/bin/env python3
"""Exp07b analysis (exploratory): per-arm means and paired-seed 95% t intervals vs uniform, from exp07b result.json cells.

Usage: python analyze_exp07b.py --runs ~/zipmix_exp07/runs/exp07b --out results_exp07b.md
"""
import argparse, json
from pathlib import Path
import numpy as np

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
ARMS = ["uniform", "global_tilt", "oracle_sciq", "anti_sciq"]
METRICS = ["holdout_target", "holdout_retention"]


def paired(a, b):
    d = np.asarray(a) - np.asarray(b); n = len(d)
    se = d.std(ddof=1) / np.sqrt(n); h = T975[n - 1] * se
    return d.mean(), d.mean() - h, d.mean() + h


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cells = [json.loads(p.read_text()) for p in sorted(args.runs.glob("*__lr*__seed*/result.json"))]
    base = json.loads((args.runs / "base" / "result.json").read_text())
    acc = {}  # (arm, lr, metric) -> {seed: accuracy}
    extra = {}
    for c in cells:
        for m in METRICS: acc.setdefault((c["arm"], c["lr"], m), {})[c["seed"]] = c["metrics"][m]["accuracy"]
        extra.setdefault((c["arm"], c["lr"]), []).append((c["sciq_share_drawn"], c["mean_train_loss"], c["unique_drawn"]))
    lrs = sorted({c["lr"] for c in cells})
    L = ["# Exp07b debug sweep (EXPLORATORY, post hoc, not pre-registered)", "",
         f"{len(cells)} cells; base model: " + ", ".join(f"{m} {base[m]['accuracy']:.4f}" for m in METRICS), "",
         "Per-arm mean accuracy over seeds (n = seeds) and paired-seed 95% t interval of the difference vs uniform (same seed).", ""]
    for lr in lrs:
        L += [f"## LR {lr:g}", "", "| arm | n | target mean | target diff vs uniform [95% CI] | retention mean | retention diff vs uniform [95% CI] | SciQ share drawn | train loss |",
              "|---|---|---|---|---|---|---|---|"]
        for arm in ARMS:
            row = [arm]
            ts = acc.get((arm, lr, METRICS[0]), {})
            if not ts: continue
            row.append(str(len(ts)))
            for m in METRICS:
                a = acc[(arm, lr, m)]; u = acc.get(("uniform", lr, m), {})
                seeds = sorted(set(a) & set(u))
                row.append(f"{np.mean(list(a.values())):.4f}")
                if arm == "uniform" or len(seeds) < 2: row.append("-")
                else:
                    d, lo, hi = paired([a[s] for s in seeds], [u[s] for s in seeds])
                    row.append(f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]")
            e = np.array(extra[(arm, lr)])
            row += [f"{e[:, 0].mean():.3f}", f"{e[:, 1].mean():.3f}"]
            L.append("| " + " | ".join(row) + " |")
        L.append("")
        L += ["Paired contrasts:", ""]
        for a1, a2, name in [("oracle_sciq", "anti_sciq", "oracle_sciq - anti_sciq (recipe sensitivity)"),
                             ("global_tilt", "uniform", "global_tilt - uniform")]:
            for m in METRICS:
                a, b = acc.get((a1, lr, m), {}), acc.get((a2, lr, m), {}); seeds = sorted(set(a) & set(b))
                if len(seeds) < 2: continue
                d, lo, hi = paired([a[s] for s in seeds], [b[s] for s in seeds])
                verdict = "excludes 0" if lo > 0 or hi < 0 else "includes 0"
                L.append(f"- {name}, {m}: {d:+.4f} [{lo:+.4f}, {hi:+.4f}] ({verdict}, n={len(seeds)})")
        L.append("")
    args.out.write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
