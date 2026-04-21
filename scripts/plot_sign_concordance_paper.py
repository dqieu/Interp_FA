#!/usr/bin/env python3
"""
Re-plot sign concordance figures.
Prefers per-step data from tracking_data.json (saved by trainer.py).
Falls back to per-epoch data from checkpoints if JSON not available.
Main figure: Conv2 and FC1 layers only.
Supplementary figure: FC2 and FC3 layers.
"""

import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SAVE_DIR = Path('save_cifar101')
OUT_DIR = Path('figures')

METHODS = {
    'FA': 'FA (Random)',
    'FA_toeplitz': 'FA (Toeplitz)',
}

COLOR_MAP = {
    'FA (Random)': '#ff7f0e',
    'FA (Toeplitz)': '#2ca02c',
}

LAYER_RENAME = {
    'layer_0': 'Conv2',
    'layer_1': 'FC1',
    'layer_2': 'FC2',
    'layer_3': 'FC3',
}

# For checkpoint fallback
LAYER_INFO = [
    ('conv2', 'B_kernel', 'layer.weight', 'Conv2'),
    ('fc1',   'B',        'layer.weight', 'FC1'),
    ('fc2',   'B',        'layer.weight', 'FC2'),
    ('fc3',   'B',        'layer.weight', 'FC3'),
]


def load_from_json():
    """Load per-step sign agreement from tracking_data.json files."""
    results = {}
    any_found = False
    for method_key, display_name in METHODS.items():
        path = SAVE_DIR / method_key / 'tracking_data.json'
        if not path.exists():
            results[display_name] = {}
            continue
        any_found = True
        with open(path) as f:
            data = json.load(f)
        sign_data = data.get('sign_agreement', {})
        renamed = {}
        for ln, vals in sign_data.items():
            renamed[LAYER_RENAME.get(ln, ln)] = vals
        results[display_name] = renamed
        print(f'  {display_name}: loaded per-step data from JSON')
    return results if any_found else None


def load_from_checkpoints():
    """Fallback: compute sign agreement per epoch from checkpoints."""
    print('  Falling back to per-epoch checkpoint computation...')
    results = {}
    for method_key, display_name in METHODS.items():
        method_dir = SAVE_DIR / method_key
        epochs = sorted(method_dir.glob('epoch_*.pt'))
        results[display_name] = {}
        for layer_prefix, b_key, w_key, label in LAYER_INFO:
            agreements = []
            for ckpt_path in epochs:
                state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
                full_b = f'{layer_prefix}.{b_key}'
                full_w = f'{layer_prefix}.{w_key}'
                if full_b not in state or full_w not in state:
                    continue
                B, W = state[full_b], state[full_w]
                if B.shape != W.shape:
                    continue
                agreements.append((W.sign() == B.sign()).float().mean().item())
            if agreements:
                results[display_name][label] = agreements
    return results


def ema(values, alpha=0.05):
    out = np.empty(len(values))
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def plot_sign_figure(results, layer_labels, save_path, per_step=False):
    """Plot sign concordance subplots for the given layers."""
    n = len(layer_labels)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 3.5), squeeze=False)
    axes = axes[0]

    for ax, label in zip(axes, layer_labels):
        for method, data in results.items():
            if label not in data or not data[label]:
                continue
            vals = np.array(data[label])
            c = COLOR_MAP.get(method)
            if per_step:
                smoothed = ema(vals)
                ax.plot(smoothed, color=c, linewidth=1.5, label=method)
                ax.plot(vals, alpha=0.12, color=c, linewidth=0.5)
            else:
                ax.plot(np.arange(len(vals)), vals, color=c, linewidth=1.5,
                        label=method)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='chance')
        ax.set_title(label, fontsize=11)
        ax.set_xlabel('Step' if per_step else 'Epoch')
        ax.set_ylabel('Sign Concordance')
        ax.set_ylim(0.45, 0.65)
        ax.legend(fontsize='small')
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {save_path}')


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print('Loading sign concordance data...')

    results = load_from_json()
    per_step = results is not None
    if not per_step:
        results = load_from_checkpoints()

    for method, layers in results.items():
        for label, vals in layers.items():
            if vals:
                print(f'  {method} / {label}: {vals[0]:.4f} -> {vals[-1]:.4f} ({len(vals)} points)')
            else:
                print(f'  {method} / {label}: (empty)')

    plot_sign_figure(results, ['Conv2', 'FC1'],
                     OUT_DIR / 'sign_concordance_main.png', per_step=per_step)
    plot_sign_figure(results, ['FC2', 'FC3'],
                     OUT_DIR / 'sign_concordance_supp.png', per_step=per_step)


if __name__ == '__main__':
    main()
