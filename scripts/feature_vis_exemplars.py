#!/usr/bin/env python3
"""
feature_vis_exemplars.py — For each model's top 9 conv channels,
find the 9 real test images that activate each channel the most.
"""

import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.data import get_device
from src.models import CIFAR101

# ── Constants ─────────────────────────────────────────────────────────
DEVICE = get_device()
SAVE_DIR = Path('save_cifar101')
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
FA_SCALE = {'BP': 0.1, 'FA': 0.01, 'FA_toeplitz': 0.01,
            'FA_uSF_init': 0.1, 'FA_uSF_sn': 0.1}
METHODS = ['BP', 'FA', 'FA_toeplitz', 'FA_uSF_init', 'FA_uSF_sn']
TARGET_CLASS = 5  # dog
OUT_DIR = Path('figures/feature_vis_exemplars')


# ── Model loading ─────────────────────────────────────────────────────
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


# ── Channel importance (GradCAM-style, averaged over all dog images) ──
def compute_mean_dog_importance(model, dataset, layer_name):
    """Compute per-channel importance averaged over all dog (class 5) images.
    Uses each model's own predicted class as the GradCAM target."""
    target_module = getattr(model, layer_name)
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

    print(f'  Averaged over {count} dog images')
    return importance_sum / count


# ── Find top-activating images for a set of channels ─────────────────
@torch.no_grad()
def find_top_activating_images(model, layer_name, channel_indices, dataset,
                               k=9, batch_size=256):
    """For each channel in channel_indices, find the k images from dataset
    that produce the highest mean activation.
    Returns dict: channel_idx -> list of (dataset_idx, activation_value)."""
    target_module = getattr(model, layer_name)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    num_channels = len(channel_indices)
    ch_list = channel_indices.tolist()

    # Track top-k per channel using sorted lists
    # Each entry: (activation_value, global_index)
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

        act = activation['val']  # [B, C, H, W]

        for ch in ch_list:
            # Mean activation per image for this channel
            ch_act = act[:, ch].mean(dim=(1, 2))  # [B]
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


# ── Find target dog image ────────────────────────────────────────────
@torch.no_grad()
def find_dog_image(dataset, bp_model, fat_model):
    best = None
    for i in range(len(dataset)):
        img, label = dataset[i]
        if label != TARGET_CLASS:
            continue
        x = img.unsqueeze(0).to(DEVICE)
        bp_probs = F.softmax(bp_model(x), dim=1)[0]
        fat_probs = F.softmax(fat_model(x), dim=1)[0]
        bp_pred = bp_probs.argmax().item()
        fat_pred = fat_probs.argmax().item()
        if bp_pred == TARGET_CLASS and fat_pred != TARGET_CLASS:
            bp_conf = bp_probs[bp_pred].item()
            if best is None or bp_conf > best[3]:
                best = (i, img, bp_pred, bp_conf, fat_pred, fat_probs[fat_pred].item())
    return best


# ── Figure generation ────────────────────────────────────────────────
DISPLAY_MAP = {'FA_toeplitz': 'FA Toep', 'FA_uSF_init': 'uSF Init', 'FA_uSF_sn': 'uSF SN'}


def make_exemplar_grid(dataset, topk_results, model_name, layer_name,
                       importances, channel_indices):
    """Grid: rows = top channels, cols = 9 top-activating images."""
    n_channels = len(channel_indices)
    n_images = 9
    fig, axes = plt.subplots(n_channels, n_images, figsize=(n_images * 1.2, n_channels * 1.4))
    display_name = DISPLAY_MAP.get(model_name, model_name)
    fig.suptitle(f'{display_name} — {layer_name}: Top 9 Channels × Top 9 Activating Images',
                 fontsize=11)

    for row, ch_idx in enumerate(channel_indices.tolist()):
        imp = importances[ch_idx].item()
        entries = topk_results[ch_idx]
        for col in range(n_images):
            ax = axes[row, col]
            if col < len(entries):
                val, ds_idx = entries[col]
                img, label = dataset[ds_idx]
                ax.imshow(denormalize(img))
                ax.set_xlabel(f'{CLASS_NAMES[label]}', fontsize=6)
            else:
                ax.axis('off')
                continue
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(f'Ch {ch_idx}\n({imp:.3f})', fontsize=7)

    fig.tight_layout()
    fname = f'{model_name}_{layer_name}_exemplars.png'
    fig.savefig(OUT_DIR / fname, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {fname}')


def make_summary_figure(dataset, idx, predictions):
    img, label = dataset[idx]
    fig, ax = plt.subplots(1, 1, figsize=(3, 4.5))
    ax.imshow(denormalize(img))
    lines = [f'Test image #{idx} — True: {CLASS_NAMES[label]}']
    for method, (pred, conf) in predictions.items():
        name = DISPLAY_MAP.get(method, method)
        lines.append(f'{name}: {CLASS_NAMES[pred]} ({conf:.1%})')
    ax.set_title('\n'.join(lines), fontsize=8)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'summary.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved summary.png')


# ── Main ─────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print('Loading models...')
    models = {m: load_bp_shell(m) for m in METHODS}

    print('Loading test dataset...')
    dataset = get_test_dataset()

    print('Finding target dog image...')
    result = find_dog_image(dataset, models['BP'], models['FA_toeplitz'])
    if result is None:
        print('ERROR: No dog image found where BP correct and FA_toeplitz wrong')
        return
    idx, img_tensor, _, _, _, _ = result

    # Get predictions from all models
    predictions = {}
    with torch.no_grad():
        x = img_tensor.unsqueeze(0).to(DEVICE)
        for method, model in models.items():
            probs = F.softmax(model(x), dim=1)[0]
            pred = probs.argmax().item()
            conf = probs[pred].item()
            predictions[method] = (pred, conf)
            print(f'  {method}: {CLASS_NAMES[pred]} ({conf:.1%})')

    make_summary_figure(dataset, idx, predictions)

    for method, model in models.items():
        for layer_name in ['conv1', 'conv2']:
            print(f'\n--- {method} / {layer_name} ---')
            importance = compute_mean_dog_importance(model, dataset, layer_name)
            _, top_indices = torch.topk(importance, 9)
            print(f'  Top channels: {top_indices.tolist()}')

            print(f'  Scanning dataset for top-activating images...')
            topk = find_top_activating_images(
                model, layer_name, top_indices, dataset, k=9
            )

            # Print class distribution for each channel
            for ch in top_indices.tolist():
                classes = [CLASS_NAMES[dataset[e[1]][1]] for e in topk[ch]]
                print(f'    Ch {ch}: {classes}')

            make_exemplar_grid(dataset, topk, method, layer_name,
                               importance, top_indices)

    print('\nDone! Check figures/feature_vis_exemplars/')


if __name__ == '__main__':
    main()
