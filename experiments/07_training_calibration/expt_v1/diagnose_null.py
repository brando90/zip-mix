#!/usr/bin/env python3
"""Exp07 Stage B null diagnosis (post-hoc, CPU, no new training). Usage:
python diagnose_null.py --data <data/stage_b> --runs <runs/stage_b>"""
import argparse, numpy as np
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--data", type=Path, required=True); ap.add_argument("--runs", type=Path, required=True)
a = ap.parse_args()
M = ["conditional_signed_rank", "permuted_stratum_rank", "uniform_stratum"]
s = np.load(a.data / "selection.npz"); ps = np.load(a.data / "pack_scores.npz"); dom = np.load(a.data / "train.npz")["domain"]
mx = ps["maximum"][s["pack_index"]]
print("1) Source mix per selector (identical by design):")
for m in M:
    w = s[m] / s[m].sum(); print(" ", m, {d: round(float(w[dom == d].sum()), 4) for d in np.unique(dom)})
print("2) Mean target-alignment (maximum) of the drawn data, pool sd %.4f:" % ps["maximum"].std())
for m in M: print(" ", m, round(float((s[m] / s[m].sum() * mx).sum()), 5))
print("3) Alignment vs domain: corr(item alignment, is_sciq) = %.3f; sciq share bottom/top quartile = %.2f / %.2f" % (
    np.corrcoef(mx, dom == "sciq")[0, 1], (dom[mx <= np.quantile(mx, .25)] == "sciq").mean(), (dom[mx >= np.quantile(mx, .75)] == "sciq").mean()))
for split in ["holdout_target", "holdout_retention"]:
    P = {m: np.array([np.load(a.runs / f"{m}__seed{k}/{split}_predictions.npz")["prediction"] for k in range(5)]) for m in M}
    C = {m: np.array([np.load(a.runs / f"{m}__seed{k}/{split}_predictions.npz")["correct"] for k in range(5)]) for m in M}
    base = np.load(a.runs / f"base/{split}_predictions.npz")["prediction"]
    seed_dis = np.mean([(P[M[0]][i] != P[M[0]][j]).mean() for i in range(5) for j in range(i + 1, 5)])
    arm_dis = np.mean([(P[M[0]][k] != P[M[1]][k]).mean() for k in range(5)])
    q = np.mean([(C[M[0]][k] != C[M[1]][k]).mean() for k in range(5)])
    d = C[M[0]].mean(0) - C[M[1]].mean(0)
    print(f"4) {split}: flip vs base {np.mean([(P[m] != base).mean() for m in M]):.3f}; same-arm seed disagreement {seed_dis:.3f}; "
          f"aligned-vs-shuffled same-seed disagreement {arm_dis:.3f}; single-seed SE {np.sqrt(q/500):.4f}; "
          f"5-seed item-paired SE {d.std(ddof=1)/np.sqrt(500):.4f}; 80%-power MDE (5 seeds) {2.8*d.std(ddof=1)/np.sqrt(500):.4f}")
