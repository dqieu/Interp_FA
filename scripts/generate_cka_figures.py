#!/usr/bin/env python3
"""
Generate CKA comparison figures: BP vs each FA method, on 3 data subsets.
Saves to figures/CKA/.
"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms as T
import matplotlib.pyplot as plt

from src.data import get_device
from src.models import CIFAR101
from CKA.cka import CKACalculator

DEVICE = get_device()
SAVE_DIR = Path('save_cifar101')
OUT_DIR = Path('figures/CKA')
OUT_DIR.mkdir(parents=True, exist_ok=True)

METHODS = ['FA', 'FA_toeplitz', 'FA_uSF_init', 'FA_uSF_sn']

DISPLAY_NAMES = {
    'FA': 'FA (Random)',
    'FA_toeplitz': 'FA (Toeplitz)',
    'FA_uSF_init': 'uSF Init',
    'FA_uSF_sn': 'uSF SN',
}

LAYER_LABELS = ['conv1', 'conv2', 'fc1', 'fc2', 'fc3']


def get_val_test_dataset():
    """Return combined val + test dataset with CenterCrop(24), matching training split."""
    val_transform = T.Compose([
        T.CenterCrop(24),
        T.ToTensor(),
        T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2470, 0.2435, 0.2616)),
    ])

    # Replicate the val split from training (seed=42, last 10% of train)
    train_full = datasets.CIFAR10(root='./data', train=True, transform=val_transform, download=True)
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=val_transform, download=True)

    rng = np.random.RandomState(42)
    indices = rng.permutation(len(train_full))
    split = int(0.9 * len(train_full))
    val_indices = indices[split:]

    val_dataset = Subset(train_full, val_indices)
    combined = ConcatDataset([val_dataset, test_dataset])
    return combined


def load_model(method):
    """Load best checkpoint for a method."""
    model = CIFAR101(num_classes=10, learn=method)
    ckpt_path = SAVE_DIR / method / 'best.pt'
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    model = model.to(DEVICE).eval()
    return model


@torch.no_grad()
def get_predictions(model, dataset, batch_size=512):
    """Return per-sample predictions and correctness mask."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_preds = []
    all_labels = []
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        logits = model(imgs)
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(labels)
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    correct = (preds == labels)
    return preds, labels, correct


def make_subset_loader(dataset, indices, batch_size=256):
    """Create a DataLoader for a subset of indices, with drop_last for CKA."""
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)


def compute_cka(model_bp, model_other, dataloader, num_epochs=5):
    """Compute CKA matrix between BP model and another model."""
    hook_layers = (nn.Conv2d, nn.Linear)
    calculator = CKACalculator(
        model1=model_bp,
        model2=model_other,
        dataloader=dataloader,
        hook_layer_types=hook_layers,
        num_epochs=num_epochs,
        device=DEVICE,
    )
    cka_matrix = calculator.calculate_cka_matrix()
    calculator.reset()
    return cka_matrix.cpu().numpy()


def plot_cka(cka_matrix, title, save_path):
    """Plot and save a CKA heatmap."""
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cka_matrix, cmap='inferno', vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(LAYER_LABELS)))
    ax.set_xticklabels(LAYER_LABELS, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(LAYER_LABELS)))
    ax.set_yticklabels(LAYER_LABELS, fontsize=9)
    ax.set_xlabel("BP layers")
    ax.set_ylabel("Method layers")
    ax.set_title(title, fontsize=11)

    # Annotate cells with values
    for i in range(cka_matrix.shape[0]):
        for j in range(cka_matrix.shape[1]):
            val = cka_matrix[i, j]
            color = 'white' if val < 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color=color)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def main():
    print(f"Device: {DEVICE}")

    # Load data
    print("Loading val + test dataset...")
    dataset = get_val_test_dataset()
    print(f"Total samples: {len(dataset)}")

    # Load BP model and get predictions
    print("Loading BP model...")
    model_bp = load_model('BP')
    bp_preds, labels, bp_correct = get_predictions(model_bp, dataset)
    print(f"BP accuracy: {bp_correct.float().mean():.4f}")

    # For each method, compute CKA on 3 subsets
    for method in METHODS:
        display = DISPLAY_NAMES[method]
        print(f"\n{'='*50}")
        print(f"Method: {display}")
        print(f"{'='*50}")

        model_other = load_model(method)
        _, _, other_correct = get_predictions(model_other, dataset)
        print(f"  {display} accuracy: {other_correct.float().mean():.4f}")

        # Define 3 subsets
        all_indices = list(range(len(dataset)))

        both_correct_mask = bp_correct & other_correct
        both_correct_indices = torch.where(both_correct_mask)[0].tolist()

        bp_right_other_wrong_mask = bp_correct & ~other_correct
        bp_right_other_wrong_indices = torch.where(bp_right_other_wrong_mask)[0].tolist()

        subsets = {
            'all': (all_indices, 'All'),
            'both_correct': (both_correct_indices, 'Both correct'),
            'bp_only': (bp_right_other_wrong_indices, 'BP correct, method wrong'),
        }

        for subset_key, (indices, subset_label) in subsets.items():
            n = len(indices)
            print(f"\n  Subset: {subset_label} (n={n})")
            if n < 256:
                print(f"  Skipping — too few samples for CKA (need ≥256)")
                continue

            loader = make_subset_loader(dataset, indices)
            cka = compute_cka(model_bp, model_other, loader)

            title = f"CKA: BP vs {display}\n{subset_label} (n={n})"
            fname = f"{method}_{subset_key}.png"
            plot_cka(cka, title, OUT_DIR / fname)

        # Cleanup model from GPU
        del model_other
        torch.mps.empty_cache() if DEVICE.type == 'mps' else None

    print(f"\nAll figures saved to {OUT_DIR}/")


if __name__ == '__main__':
    main()
