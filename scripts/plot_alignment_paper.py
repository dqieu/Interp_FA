#!/usr/bin/env python3
"""
Re-plot gradient alignment from saved per-step tracking data.
Bigger panels, single shared legend, renamed layers.

Expects: save_cifar101/{method}/tracking_data.json
  (created by trainer.py after training with track_alignment=True)
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SAVE_DIR = Path('save_cifar101')
OUT_DIR = Path('figures')

METHODS = [
    ('FA',           'FA (Random)',   '#ff7f0e'),
    ('FA_toeplitz',  'FA (Toeplitz)', '#2ca02c'),
    ('FA_uSF_init',  'uSF Init',     '#d62728'),
    ('FA_uSF_sn',    'uSF SN',       '#9467bd'),
]

LAYER_RENAME = {
    'layer_0': 'Conv2',
    'layer_1': 'FC1',
    'layer_2': 'FC2',
    'layer_3': 'FC3',
}

LAYER_ORDER = ['Conv2', 'FC1', 'FC2', 'FC3']


def ema(values, alpha=0.05):
    out = np.empty(len(values))
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def load_tracking(method_key):
    path = SAVE_DIR / method_key / 'tracking_data.json'
    if not path.exists():
        print(f'  Warning: {path} not found')
        return {}
    with open(path) as f:
        return json.load(f)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load alignment data, renaming layers
    all_alignment = {}
    for method_key, display_name, color in METHODS:
        data = load_tracking(method_key)
        alignment = data.get('alignment', {})
        renamed = {}
        for ln, vals in alignment.items():
            renamed[LAYER_RENAME.get(ln, ln)] = vals
        all_alignment[display_name] = renamed
        n_steps = {k: len(v) for k, v in renamed.items()}
        print(f'{display_name}: {n_steps}')

    # Plot — 2×2 grid for bigger panels
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), squeeze=False)

    for idx, label in enumerate(LAYER_ORDER):
        ax = axes[idx // 2, idx % 2]
        for method_key, display_name, color in METHODS:
            data = all_alignment.get(display_name, {})
            if label not in data or not data[label]:
                continue
            raw = np.array(data[label])
            smoothed = ema(raw)
            ax.plot(smoothed, color=color, linewidth=1.5, label=display_name)
            ax.plot(raw, alpha=0.12, color=color, linewidth=0.5)
        ax.set_title(label, fontsize=13)
        ax.set_xlabel('Step')
        ax.set_ylabel('Angle (degrees)')
        ax.grid(True, alpha=0.3)

    # Single shared legend at top
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(METHODS),
               fontsize='medium', bbox_to_anchor=(0.5, 1.02))

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_DIR / 'alignment_paper.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {OUT_DIR / "alignment_paper.png"}')


if __name__ == '__main__':
    main()
