"""Alignment tracking utilities for FA/DFA layers."""

from pathlib import Path

import torch

from .fa import FALayer
from .dfa import DFALayer


def enable_alignment_tracking(model):
    """Turn on alignment tracking for all FA/DFA layers and clear history."""
    idx = 0
    for m in model.modules():
        if isinstance(m, (FALayer, DFALayer)):
            m.track_alignment = True
            m.alignment_history = []
            if not m.layer_name:
                m.layer_name = f"layer_{idx}"
                idx += 1


def disable_alignment_tracking(model):
    """Turn off alignment tracking for all FA/DFA layers."""
    for m in model.modules():
        if isinstance(m, (FALayer, DFALayer)):
            m.track_alignment = False


def collect_alignment_data(model):
    """Return {layer_name: [angles]} for all tracked FA/DFA layers."""
    data = {}
    for m in model.modules():
        if isinstance(m, (FALayer, DFALayer)) and m.layer_name:
            data[m.layer_name] = list(m.alignment_history)
    return data


def plot_alignment(all_results, save_path, ema_alpha=0.05):
    """Plot alignment angles: one subplot per layer, overlaying methods.

    all_results: dict  method_name -> {layer_name: [angles]}
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    # Collect all layer names across methods
    layer_names = sorted({ln for res in all_results.values() for ln in res})
    if not layer_names:
        print("No alignment data to plot.")
        return

    n = len(layer_names)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)
    axes = axes[0]

    def _ema(values, alpha):
        out = np.empty(len(values))
        out[0] = values[0]
        for i in range(1, len(values)):
            out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
        return out

    for ax, ln in zip(axes, layer_names):
        for method, res in all_results.items():
            if ln not in res or not res[ln]:
                continue
            raw = np.array(res[ln])
            smoothed = _ema(raw, ema_alpha)
            ax.plot(smoothed, label=method, alpha=0.85)
            ax.plot(raw, alpha=0.15, color=ax.get_lines()[-1].get_color())
        ax.set_title(ln)
        ax.set_xlabel('Step')
        ax.set_ylabel('Angle (degrees)')
        ax.legend(fontsize='small')
        ax.grid(True, alpha=0.3)

    fig.suptitle('BP vs Feedback Gradient Alignment During Training', fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Alignment plot saved to {save_path}")


def enable_sign_tracking(model):
    """Turn on sign agreement tracking for all FALayer modules and clear history."""
    idx = 0
    for m in model.modules():
        if isinstance(m, FALayer):
            m.track_sign_agreement = True
            m.sign_agreement_history = []
            if not m.layer_name:
                m.layer_name = f"layer_{idx}"
                idx += 1


def collect_sign_data(model):
    """Return {layer_name: [agreement_values]} for all tracked FALayer modules."""
    data = {}
    for m in model.modules():
        if isinstance(m, FALayer) and m.layer_name:
            data[m.layer_name] = list(m.sign_agreement_history)
    return data


def plot_sign_agreement(all_results, save_path, ema_alpha=0.05):
    """Plot sign agreement: one subplot per layer, overlaying methods.

    all_results: dict  method_name -> {layer_name: [agreement_values]}
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    layer_names = sorted({ln for res in all_results.values() for ln in res})
    if not layer_names:
        print("No sign agreement data to plot.")
        return

    n = len(layer_names)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)
    axes = axes[0]

    def _ema(values, alpha):
        out = np.empty(len(values))
        out[0] = values[0]
        for i in range(1, len(values)):
            out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
        return out

    for ax, ln in zip(axes, layer_names):
        for method, res in all_results.items():
            if ln not in res or not res[ln]:
                continue
            raw = np.array(res[ln])
            smoothed = _ema(raw, ema_alpha)
            ax.plot(smoothed, label=method, alpha=0.85)
            ax.plot(raw, alpha=0.15, color=ax.get_lines()[-1].get_color())
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='chance')
        ax.set_title(ln)
        ax.set_xlabel('Step')
        ax.set_ylabel('Sign Agreement')
        ax.set_ylim(0, 1)
        ax.legend(fontsize='small')
        ax.grid(True, alpha=0.3)

    fig.suptitle('Sign Agreement: sign(W) == sign(B)', fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Sign agreement plot saved to {save_path}")


def plot_val_acc(all_results, save_path):
    """Plot validation accuracy over epochs.

    all_results: {method_name: [val_acc_per_epoch]}
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for method, accs in all_results.items():
        epochs = list(range(1, len(accs) + 1))
        ax.plot(epochs, accs, label=method, marker='o', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val Accuracy (%)')
    ax.set_title('Validation Accuracy Over Training')
    ax.legend(fontsize='small')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Val accuracy plot saved to {save_path}")


def verify_B_constancy(save_dir, method):
    """Check that B matrices remain constant across training checkpoints."""
    method_dir = Path(save_dir) / method
    epoch0 = torch.load(method_dir / 'epoch_000.pt', map_location='cpu', weights_only=True)

    # Find last epoch checkpoint
    ckpts = sorted(method_dir.glob('epoch_*.pt'))
    last_ckpt = [c for c in ckpts if c.name != 'epoch_000.pt' and c.name != 'best.pt'][-1]
    last = torch.load(last_ckpt, map_location='cpu', weights_only=True)

    print(f"\n  B-matrix verification ({method}): epoch_000 vs {last_ckpt.name}")
    b_keys = [k for k in epoch0 if '.B' in k and not k.endswith('_initialized')]
    all_match = True
    for k in b_keys:
        match = torch.equal(epoch0[k], last[k])
        print(f"    {k}: identical = {match}")
        all_match = all_match and match
    print(f"    All B matrices identical: {all_match}")
    return all_match
