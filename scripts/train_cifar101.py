#!/usr/bin/env python3
"""
train_cifar101.py — Train CIFAR101 (Moskovitz Arch 1) on CIFAR-10.

Architecture (24×24×3 input):
  Conv1: 3→64, k5 → 64×20×20 → Pool(2) → 64×10×10
  Conv2: 64→64, k5 → 64×6×6  → Pool(2) → 64×3×3
  FC1: 576→384 → FC2: 384→192 → FC3: 192→10

Methods: BP, FA (Random), FA_toeplitz, FA_uSF_init, FA_uSF_sn
"""

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms as T

from src.data import get_device
from src.models import CIFAR101
from src.trainer import hp_search, train_and_evaluate
from src.analysis import verify_B_constancy, plot_alignment, plot_sign_agreement, plot_val_acc

DEVICE = get_device()
SAVE_DIR = Path('save_cifar101')
IN_CHANNELS = 3


def get_cifar10_24x24_loaders(batch_size=128, val_batch_size=1000, data_dir='./data'):
    """CIFAR-10 with 24×24 crops as required by CIFAR101 architecture."""
    train_transform = T.Compose([
        T.RandomCrop(24),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2470, 0.2435, 0.2616)),
    ])
    val_transform = T.Compose([
        T.CenterCrop(24),
        T.ToTensor(),
        T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2470, 0.2435, 0.2616)),
    ])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, transform=train_transform, download=True)
    val_dataset = datasets.CIFAR10(root=data_dir, train=True, transform=val_transform, download=True)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, transform=val_transform, download=True)

    n = len(train_dataset)
    indices = np.random.permutation(n)
    split = int(0.9 * n)

    train_dataset = Subset(train_dataset, indices[:split])
    val_dataset = Subset(val_dataset, indices[split:])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def model_fn(method, cfg):
    return CIFAR101(num_classes=10, learn=method,
                    fa_scale=cfg.get('scale', 0.1))


def main():
    print(f"Device: {DEVICE}")

    torch.manual_seed(42)
    np.random.seed(42)
    train_loader, val_loader, test_loader = get_cifar10_24x24_loaders()

    methods = ['BP', 'FA', 'FA_toeplitz', 'FA_uSF_init', 'FA_uSF_sn']

    # --- Phase 1: HP search (3 epochs) ---
    best_cfgs = {}
    for method in methods:
        best_cfgs[method] = hp_search(model_fn, method, train_loader, val_loader, DEVICE)

    # --- Phase 2: Full training until convergence ---
    MAX_EPOCHS = 50
    print("\n" + "=" * 60)
    print(f"FULL TRAINING ({MAX_EPOCHS} epochs)")
    print("=" * 60)

    all_res = {}
    for method in methods:
        all_res[method] = train_and_evaluate(
            model_fn, method, best_cfgs[method],
            train_loader, val_loader, test_loader, DEVICE,
            save_dir=SAVE_DIR, max_epochs=MAX_EPOCHS,
            track_alignment=True)

    # --- B-matrix verification ---
    print("\n" + "=" * 60)
    print("B-MATRIX VERIFICATION")
    print("=" * 60)
    for method in ['FA', 'FA_toeplitz']:
        verify_B_constancy(SAVE_DIR, method)
    # Skip B-constancy for uSF methods — B changes by design

    # --- Alignment plot ---
    alignment_results = {}
    for method in ['FA', 'FA_toeplitz', 'FA_uSF_init', 'FA_uSF_sn']:
        if 'alignment' in all_res[method]:
            alignment_results[method] = all_res[method]['alignment']

    if alignment_results:
        os.makedirs('figures', exist_ok=True)
        plot_alignment(alignment_results, 'figures/online_alignment_cifar101.png')

    display_names = {
        'BP': 'Backprop', 'FA': 'FA (Random)', 'FA_toeplitz': 'FA (Toeplitz)',
        'FA_uSF_init': 'uSF Init', 'FA_uSF_sn': 'uSF SN',
    }

    # --- Sign agreement plot ---
    sign_results = {}
    for method in ['FA', 'FA_toeplitz']:
        if 'sign_agreement' in all_res[method]:
            sign_results[method] = all_res[method]['sign_agreement']
    if sign_results:
        os.makedirs('figures', exist_ok=True)
        plot_sign_agreement(sign_results, 'figures/sign_agreement_cifar101.png')

    # --- Val accuracy plot ---
    os.makedirs('figures', exist_ok=True)
    val_acc_results = {display_names[m]: all_res[m]['val_acc'] for m in methods}
    plot_val_acc(val_acc_results, 'figures/val_acc_cifar101.png')

    # --- Summary ---
    os.makedirs('results', exist_ok=True)

    header = f"{'Method':<16} {'Config':<35} {'Best Val':<10} {'Test Acc':<10} {'Best Ep'}"
    sep = "-" * 80
    lines = [header, sep]

    for method in methods:
        res = all_res[method]
        line = (f"{display_names[method]:<16} {str(best_cfgs[method]):<35} "
                f"{res['best_val_acc']:>6.2f}%   {res['test_acc']:>6.2f}%   "
                f"{res['best_epoch']}")
        lines.append(line)

    summary = "\n".join(lines)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n{summary}")

    results_path = 'results/cifar101_results.txt'
    with open(results_path, 'w') as f:
        f.write(summary + "\n")
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
