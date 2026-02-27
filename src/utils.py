from math import sqrt

import numpy as np
import torch


def random_conv_matrix_2d(H, W, kH, kW, padding=0):
    """
    Build the doubly block-Toeplitz matrix for a 2D conv (stride=1).
    Output shape: (out_H*out_W, H*W)
    """
    pH, pW = H + 2 * padding, W + 2 * padding
    out_H, out_W = pH - kH + 1, pW - kW + 1
    K = np.random.randn(kH, kW)              # random kernel

    full = np.zeros((out_H * out_W, H * W))
    for i in range(out_H):
        for j in range(out_W):
            patch = np.zeros((pH, pW))
            patch[i:i+kH, j:j+kW] = K
            # Extract the non-padded region that maps to original input
            unpadded = patch[padding:padding+H, padding:padding+W]
            full[i * out_W + j] = unpadded.ravel()
    return torch.from_numpy(full)

def random_square_conv_matrix_2d(sz, ksz):
    H = W = sz
    kH = kW = ksz
    return random_conv_matrix_2d(H, W, kH, kW)