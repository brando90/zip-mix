#!/usr/bin/env python3
"""Phase 2 (Experiment 02): CR distribution analysis on the committed 20k-doc pilot CSVs (CPU only, seconds).

Answers Q1 (shape of CR per corpus) and Q6 (do corpora differ enough). Protocol: README.md Phase 2.
Inputs : results/cr_pilot_{fineweb_edu,dclm,pile}.csv
Outputs: results/phase2_summary.csv, phase2_ks.csv, phase2_quartiles_vs_compel.csv, phase2_buckets.csv,
         results/phase2_cr_hist.png, phase2_cr_cdf.png, results/phase2_report.md
Usage  : python 02_cr_distribution.py
"""
import itertools
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(EXP_DIR, "results")
CORPORA = ["fineweb_edu", "dclm", "pile"]
COMPEL = {"fineweb_edu": (0.67, 0.73, 0.78), "dclm": (0.69, 0.75, 0.82)}  # Q1, median, Q3 reported by Compel
BAND = (0.65, 0.80)
BUCKET_M = (3, 5, 7, 10)

df = {c: pd.read_csv(os.path.join(RES, f"cr_pilot_{c}.csv")) for c in CORPORA}
cr = {c: df[c]["cr"].to_numpy() for c in CORPORA}

summ = pd.DataFrame({c: {"n": len(v), "mean": v.mean(), "std": v.std(ddof=1), "q1": np.quantile(v, .25),
                         "median": np.median(v), "q3": np.quantile(v, .75), "iqr": np.quantile(v, .75) - np.quantile(v, .25),
                         "skew": stats.skew(v), "frac_in_band": float(((v >= BAND[0]) & (v <= BAND[1])).mean())}
                     for c, v in cr.items()}).T
summ.to_csv(os.path.join(RES, "phase2_summary.csv"), float_format="%.4f")

rows = []
for c, (q1, med, q3) in COMPEL.items():
    s = summ.loc[c]
    rows.append(dict(corpus=c, q1_ours=s.q1, q1_compel=q1, med_ours=s["median"], med_compel=med, q3_ours=s.q3, q3_compel=q3))
pd.DataFrame(rows).to_csv(os.path.join(RES, "phase2_quartiles_vs_compel.csv"), index=False, float_format="%.4f")

ks = []
for a, b in itertools.combinations(CORPORA, 2):
    D, p = stats.ks_2samp(cr[a], cr[b])
    sp = np.sqrt((cr[a].var(ddof=1) + cr[b].var(ddof=1)) / 2)
    ks.append(dict(a=a, b=b, ks_D=D, p=p, cohens_d=(cr[a].mean() - cr[b].mean()) / sp))
ks = pd.DataFrame(ks)
ks.to_csv(os.path.join(RES, "phase2_ks.csv"), index=False, float_format="%.4g")

# bucket populations: uniform-width boundaries on [0,1.1] and quantile boundaries pooled over the 3 corpora
pooled = np.concatenate(list(cr.values()))
brow = []
for M in BUCKET_M:
    for kind, edges in (("uniform", np.linspace(0, 1.1, M + 1)), ("quantile", np.quantile(pooled, np.linspace(0, 1, M + 1)))):
        edges = edges.copy(); edges[0], edges[-1] = -np.inf, np.inf
        for c in CORPORA:
            cnt = np.histogram(cr[c], bins=edges)[0]
            brow.append(dict(M=M, kind=kind, corpus=c, counts=" ".join(map(str, cnt)), min_share=cnt.min() / len(cr[c])))
pd.DataFrame(brow).to_csv(os.path.join(RES, "phase2_buckets.csv"), index=False, float_format="%.4f")

fig, ax = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True)
for a, c in zip(ax, CORPORA):
    a.hist(cr[c], bins=100, range=(0, 1.1), color="#4c72b0")
    a.axvspan(*BAND, color="orange", alpha=.2, label="Compel [0.65, 0.80]")
    a.set_title(f"{c} (n={len(cr[c])})"); a.set_xlabel("LZ4 compression ratio")
ax[0].set_ylabel("documents"); ax[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(RES, "phase2_cr_hist.png"), dpi=140); plt.close(fig)
fig, a = plt.subplots(figsize=(5, 3.6))
for c in CORPORA:
    x = np.sort(cr[c]); a.plot(x, np.arange(1, len(x) + 1) / len(x), label=c)
a.axvspan(*BAND, color="orange", alpha=.15); a.set_xlabel("LZ4 compression ratio"); a.set_ylabel("CDF"); a.legend()
fig.tight_layout(); fig.savefig(os.path.join(RES, "phase2_cr_cdf.png"), dpi=140); plt.close(fig)

with open(os.path.join(RES, "phase2_report.md"), "w") as f:
    f.write("# Exp 02 Phase 2: CR distributions (20k-doc pilots, committed CSVs)\n\n")
    f.write("Scope: pilot n=20,000 per corpus (README plans 500K); one seed (42); LZ4 frame default; docs < 512 B dropped.\n\n")
    f.write("## Summary\n\n" + summ.round(4).to_markdown() + "\n\n## Quartiles vs Compel\n\n")
    f.write(pd.DataFrame(rows).round(3).to_markdown(index=False) + "\n\n## Pairwise KS (D = effect size) and Cohen's d\n\n")
    f.write(ks.round(4).to_markdown(index=False) + "\n\n## Bucket populations (counts per bucket)\n\n")
    f.write(pd.DataFrame(brow).round(4).to_markdown(index=False) + "\n")
print(summ.round(3)); print(ks.round(4))
