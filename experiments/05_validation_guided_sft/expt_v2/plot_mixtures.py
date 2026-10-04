#!/usr/bin/env python3
"""Plot exact frozen-pool sampling probabilities; no training outcomes."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parent
    public = root / "data_manifest.json"
    manifest = json.loads((public if public.exists() else root / "data/manifest.json").read_text())
    methods = ["sample_proportional", "zipmix_static", "shuffled_zipmix", "wrongtarget_zipmix"]
    labels = ["Population", "Science target", "Shuffled scores", "Wrong target"]
    colors = ["#8D99AE", "#177E89", "#B091BC", "#DD6E42"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    values = [100 * manifest["mixtures"][m]["domain_mass"]["sciq"] for m in methods]
    bars = axes[0].barh(labels, values, color=colors, height=.6)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 43)
    axes[0].set_xlabel("Probability of sampling a SciQ question (%)")
    axes[0].set_title("Target choice changes the training mixture", loc="left", fontsize=11)
    for bar, value in zip(bars, values):
        axes[0].text(value + .6, bar.get_y() + bar.get_height()/2,
                     f"{value:.2f}%", va="center", fontsize=10)
    x = np.arange(5)
    for method, label, color in zip(methods, labels, colors):
        axes[1].plot(x, np.array(manifest["mixtures"][method]["bucket_mass"]) * 100,
                     marker="o", label=label, color=color, linewidth=2)
    axes[1].set_xticks(x, [str(i + 1) for i in x])
    axes[1].set_ylim(0, 32)
    axes[1].set_xlabel("Compression bin (low to high ratio)")
    axes[1].set_ylabel("Sampling probability (%)")
    axes[1].set_title("Bucket weights retain target information", loc="left", fontsize=11)
    axes[1].legend(frameon=False, fontsize=9, loc="lower left")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Fixed-size compression inputs produce different target-conditioned mixtures", fontsize=13)
    fig.text(.5, .025, "351 source packs; 8 science + 8 wrong-target views; every view is 4,096 real bytes.\n"
             "Exact finite-pool probabilities, not measured generalization gains. Direct selection retains only science questions.",
             ha="center", fontsize=9, color="#444444")
    fig.subplots_adjust(left=.145, right=.975, top=.82, bottom=.22, wspace=.32)
    output = root / "results"
    output.mkdir(exist_ok=True)
    fig.savefig(output / "mixture_diagnostics.png", dpi=180)
    fig.savefig(output / "mixture_diagnostics.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
