#!/usr/bin/env python3
"""
Generate packed conv2 feature-vis exemplar grids for the paper.
For each method: top 3 channels, 9 images per channel.
Layout: 3 rows (channels) × 9 columns (images), no labels, no borders.
"""

import re
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.data import get_device
from src.models import CIFAR101

DEVICE = get_device()
SAVE_DIR = Path('save_cifar101')
OUT_DIR = Path('figures/feature_vis_exemplars')
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)
METHODS = ['BP', 'FA', 'FA_toeplitz', 'FA_uSF_init', 'FA_uSF_sn']
DISPLAY_MAP = {
    'BP': 'BP', 'FA': 'FA (Random)', 'FA_toeplitz': 'FA (Toeplitz)',
    'FA_uSF_init': 'uSF Init', 'FA_uSF_sn': 'uSF SN',
}
TARGET_CLASS = 5  # dog
LAYER_NAME = 'conv2'
TOP_K_CHANNELS = 3
TOP_K_IMAGES = 9


def remap_to_bp(state_dict):
    new = {}
    for k, v in state_dict.items():
        if re.search(r'\.B(_kernel)?$', k):
            continue
        new[k.replace('.layer.', '.')] = v
    return new


def load_bp_shell(method):
    ckpt = SAVE_DIR / method / 'best.pt'
    state = torch.load(ckpt, map_location=DEVICE, weights_only=True)
    if method != 'BP':
        state = remap_to_bp(state)
    shell = CIFAR101(num_classes=10, learn='BP')
    shell.load_state_dict(state)
    shell.to(DEVICE).eval()
    return shell


def get_test_dataset():
    transform = T.Compose([
        T.CenterCrop(24), T.ToTensor(),
        T.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
    ])
    return datasets.CIFAR10(root='./data', train=False,
                            transform=transform, download=True)


def denormalize(img_tensor):
    mean = torch.tensor(CIFAR_MEAN).view(3, 1, 1)
    std  = torch.tensor(CIFAR_STD).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def compute_mean_dog_importance(model, dataset):
    """GradCAM-style channel importance averaged over dog images."""
    target_module = getattr(model, LAYER_NAME)
    importance_sum = None
    count = 0

    for i in range(len(dataset)):
        img, label = dataset[i]
        if label != TARGET_CLASS:
            continue

        activations = {}
        gradients = {}

        def fwd_hook(module, inp, out):
            activations['val'] = out

        def bwd_hook(module, grad_in, grad_out):
            gradients['val'] = grad_out[0].detach()

        h_fwd = target_module.register_forward_hook(fwd_hook)
        h_bwd = target_module.register_full_backward_hook(bwd_hook)

        image = img.unsqueeze(0).to(DEVICE)
        model.zero_grad()
        logits = model(image)
        pred_class = logits.argmax(dim=1).item()
        logits[0, pred_class].backward()

        h_fwd.remove()
        h_bwd.remove()

        act = activations['val'].detach().squeeze(0)
        grad = gradients['val'].squeeze(0)
        imp = (grad.mean(dim=(1, 2)) * act.mean(dim=(1, 2))).abs()

        if importance_sum is None:
            importance_sum = imp
        else:
            importance_sum += imp
        count += 1

    return importance_sum / count


@torch.no_grad()
def find_top_activating_images(model, channel_indices, dataset,
                               k=TOP_K_IMAGES, batch_size=256):
    target_module = getattr(model, LAYER_NAME)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    ch_list = channel_indices.tolist()
    topk = {ch: [] for ch in ch_list}

    global_idx = 0
    for images, _ in loader:
        images = images.to(DEVICE)
        bs = images.size(0)

        activation = {}
        def fwd_hook(module, inp, out):
            activation['val'] = out
        h = target_module.register_forward_hook(fwd_hook)
        _ = model(images)
        h.remove()

        act = activation['val']
        for ch in ch_list:
            ch_act = act[:, ch].mean(dim=(1, 2))
            for i in range(bs):
                val = ch_act[i].item()
                idx = global_idx + i
                if len(topk[ch]) < k:
                    topk[ch].append((val, idx))
                    topk[ch].sort(key=lambda x: x[0], reverse=True)
                elif val > topk[ch][-1][0]:
                    topk[ch][-1] = (val, idx)
                    topk[ch].sort(key=lambda x: x[0], reverse=True)
        global_idx += bs

    return topk


def make_combined_grid(dataset, all_topk, all_channels):
    """Single figure: 3 channels (cols) × 5 methods (rows), each cell is a 3×3 image grid."""
    n_methods = len(METHODS)
    n_channels = TOP_K_CHANNELS
    grid_size = 3  # 3×3 images per cell

    # Gap rows/cols between the 3×3 sub-grids
    gap = 1
    total_rows = n_methods * grid_size + (n_methods - 1) * gap
    total_cols = n_channels * grid_size + (n_channels - 1) * gap

    cell = 0.55
    # Use GridSpec with half-width gap rows/cols
    height_ratios = []
    for m_idx in range(n_methods):
        height_ratios.extend([1] * grid_size)
        if m_idx < n_methods - 1:
            height_ratios.append(0.5)
    width_ratios = []
    for ch_idx in range(n_channels):
        width_ratios.extend([1] * grid_size)
        if ch_idx < n_channels - 1:
            width_ratios.append(0.5)
    fig, axes = plt.subplots(total_rows, total_cols,
                             figsize=(total_cols * cell, total_rows * cell),
                             gridspec_kw={'height_ratios': height_ratios,
                                          'width_ratios': width_ratios})
    fig.subplots_adjust(wspace=0.03, hspace=0.03,
                        left=0, right=1, top=0.94, bottom=0.01)

    # Hide all axes first
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    # Fill in the 3×3 sub-grids
    for m_idx, method in enumerate(METHODS):
        topk = all_topk[method]
        ch_indices = all_channels[method]
        row_offset = m_idx * (grid_size + gap)

        for ch_rank, ch_idx in enumerate(ch_indices.tolist()):
            col_offset = ch_rank * (grid_size + gap)
            entries = topk[ch_idx]

            for i in range(grid_size * grid_size):
                r, c = divmod(i, grid_size)
                ax = axes[row_offset + r, col_offset + c]
                if i < len(entries):
                    _, ds_idx = entries[i]
                    img, _ = dataset[ds_idx]
                    ax.imshow(denormalize(img))

    # Hide gap rows and columns
    gap_rows = set()
    for m_idx in range(n_methods - 1):
        gap_rows.add((m_idx + 1) * grid_size + m_idx * gap)
    gap_cols = set()
    for ch_idx in range(n_channels - 1):
        gap_cols.add((ch_idx + 1) * grid_size + ch_idx * gap)
    for r in range(total_rows):
        for c in range(total_cols):
            if r in gap_rows or c in gap_cols:
                axes[r, c].set_visible(False)

    # Method labels on the left
    for m_idx, method in enumerate(METHODS):
        row_offset = m_idx * (grid_size + gap)
        mid_row = row_offset + grid_size // 2
        axes[mid_row, 0].set_ylabel(DISPLAY_MAP[method], fontsize=9,
                                     rotation=90, labelpad=8)

    # Channel labels on top
    for ch_rank in range(n_channels):
        col_offset = ch_rank * (grid_size + gap) + grid_size // 2
        axes[0, col_offset].set_title(f'Ch {ch_rank+1}', fontsize=9)

    fname = f'exemplars_combined.png'
    fig.savefig(OUT_DIR / fname, dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f'Saved {OUT_DIR / fname}')


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print('Loading test dataset...')
    dataset = get_test_dataset()

    all_topk = {}
    all_channels = {}

    for method in METHODS:
        print(f'\n--- {method} / {LAYER_NAME} ---')
        model = load_bp_shell(method)
        importance = compute_mean_dog_importance(model, dataset)
        _, top_indices = torch.topk(importance, TOP_K_CHANNELS)
        print(f'  Top {TOP_K_CHANNELS} channels: {top_indices.tolist()}')

        topk = find_top_activating_images(model, top_indices, dataset)
        all_topk[method] = topk
        all_channels[method] = top_indices

    make_combined_grid(dataset, all_topk, all_channels)
    print('\nDone!')


if __name__ == '__main__':
    main()
