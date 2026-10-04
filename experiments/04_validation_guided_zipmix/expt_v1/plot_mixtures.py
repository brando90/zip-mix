#!/usr/bin/env python3
"""Render descriptive frozen-pool mixture probabilities, not performance."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    version = Path(__file__).resolve().parent
    pretraining = json.loads((version / 'results/mixture_diagnostics.json').read_text())
    sft = json.loads((version.parents[1] / '05_validation_guided_sft/expt_v1/data_manifest.json').read_text())
    lookup = {row['method']: row['biomedical_token_mass'] * 100 for row in pretraining['rows']}
    panels = [
        ([lookup[k] for k in ['token_proportional', 'zipmix_static', 'direct_zipfit']],
         ['Token-proportional', 'Zip-Mix buckets', 'Direct mean-score sampling'],
         'Pretraining pool: biomedical tokens'),
        ([sft['mixtures'][k]['domain_mass']['sciq'] * 100
          for k in ['sample_proportional', 'zipmix_static', 'direct_zipfit']],
         ['Sample-proportional', 'Zip-Mix buckets', 'Direct mean-score top 25%'],
         'Fine-tuning pool: science questions'),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for ax, (values, labels, title) in zip(axes, panels):
        bars = ax.barh(labels, values, color=['#8D99AE', '#177E89', '#DD6E42'], height=.6)
        ax.invert_yaxis()
        ax.set_xlim(0, 65)
        ax.set_xlabel('Sampling probability (%)')
        ax.set_title(title, fontsize=11)
        ax.spines[['top', 'right']].set_visible(False)
        for bar, value in zip(bars, values):
            ax.text(value + .7, bar.get_y() + bar.get_height()/2,
                    f'{value:.2f}%', va='center', fontsize=10)
    fig.suptitle('Prepared mixtures: Zip-Mix changes target-source sampling only slightly', fontsize=12)
    fig.savefig(version / 'results/mixture_diagnostics.png', dpi=180)


if __name__ == '__main__':
    main()
