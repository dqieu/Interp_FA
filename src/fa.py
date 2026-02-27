import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def _angle_deg_mean(a, b):
    a = a.reshape(a.size(0), -1).float()
    b = b.reshape(b.size(0), -1).float()
    cos = F.cosine_similarity(a, b, dim=1).clamp(-1, 1)
    return torch.acos(cos).mean().item() * (180.0 / math.pi)


class FALayer(nn.Module):
    """Wraps any nn.Module for Feedback Alignment.

    Uses register_full_backward_hook to replace the gradient flowing
    to the previous layer with a fixed random projection,
    instead of grad_output @ W (standard backprop).
    Weight and bias gradients are computed normally.

    For Conv2d layers with use_toeplitz=True, uses conv_transpose2d with
    a random kernel instead of a dense matmul — equivalent to replacing
    the conv kernel with a random one in the backward pass.
    """
    def __init__(self, layer, scale=0.1, use_toeplitz=True, mode='fa'):
        super().__init__()
        self.layer = layer

        self.scale = scale
        self.use_toeplitz = use_toeplitz
        self.mode = mode  # 'fa', 'usf_init', 'usf_sn'
        self._B_initialized = False
        self._use_conv_backward = False
        self._conv_padding = getattr(layer, 'padding', (0, 0))
        self._conv_stride = getattr(layer, 'stride', (1, 1))

        self.track_alignment = False
        self.alignment_history = []
        self.track_sign_agreement = False
        self.sign_agreement_history = []

        self.layer_name = ""
        self.layer.register_full_backward_hook(self._fa_hook)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys, error_msgs):
        # Pre-register feedback buffers so the base class can load them
        b_key = prefix + 'B'
        bk_key = prefix + 'B_kernel'
        if bk_key in state_dict:
            if not hasattr(self, 'B_kernel'):
                self.register_buffer('B_kernel', state_dict[bk_key])
            self._use_conv_backward = True
            self._B_initialized = True
        elif b_key in state_dict:
            if not hasattr(self, 'B'):
                self.register_buffer('B', state_dict[b_key])
            self._B_initialized = True
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    def _init_B(self, x, out):
        if self.use_toeplitz and isinstance(self.layer, nn.Conv2d):
            C_in = self.layer.in_channels
            C_out = self.layer.out_channels
            kH, kW = self.layer.kernel_size
            # conv_transpose2d weight shape: (C_out, C_in, kH, kW)
            B_kernel = torch.randn(C_out, C_in, kH, kW, device=out.device) * self.scale
            self.register_buffer('B_kernel', B_kernel)
            self._conv_padding = self.layer.padding
            self._conv_stride = self.layer.stride
            self._use_conv_backward = True
        else:
            in_dim = x.reshape(x.size(0), -1).size(1)
            out_dim = out.reshape(out.size(0), -1).size(1)
            B = torch.randn(out_dim, in_dim, device=out.device) * self.scale
            self.register_buffer('B', B)

        self._B_initialized = True

    @torch.no_grad()
    def _get_effective_B(self):
        """Compute effective feedback matrix based on mode (conv and linear)."""
        B0 = self.B_kernel if self._use_conv_backward else self.B
        if self.mode == 'fa':
            return B0
        W = self.layer.weight
        if self.mode == 'usf_init':
            return B0.abs() * W.sign()
        if self.mode == 'usf_sn':
            sign_W = W.sign()
            return W.norm(2) * sign_W / (sign_W.norm(2) + 1e-8)
        raise ValueError(f"Unknown mode: {self.mode}")

    def _fa_hook(self, module, grad_input, grad_output):
        """Replace the gradient sent to the previous layer with a random projection.
        Fires after the gradient wrt module input is calculated
        """
        if not self._B_initialized or grad_input[0] is None:
            return

        B_eff = self._get_effective_B()
        if self._use_conv_backward:
            go = grad_output[0]
            new_grad = F.conv_transpose2d(go, B_eff,
                                          padding=self._conv_padding,
                                          stride=self._conv_stride)
        else:
            go = grad_output[0].reshape(grad_output[0].size(0), -1)
            new_grad = (go @ B_eff).reshape_as(grad_input[0])

        if self.track_alignment and grad_input[0] is not None:
            self.alignment_history.append(
                _angle_deg_mean(grad_input[0].detach(), new_grad.detach()))

        if self.track_sign_agreement:
            W = self.layer.weight
            B0 = self.B_kernel if self._use_conv_backward else self.B
            if W.shape == B0.shape:
                agreement = (W.sign() == B0.sign()).float().mean().item()
                self.sign_agreement_history.append(agreement)

        return (new_grad,)

    def forward(self, x):
        out = self.layer(x)
        if not self._B_initialized:
            self._init_B(x, out)
        return out

def init_sequential(layers, fa_scale=0.1, use_toeplitz=True, mode='fa'):
    # first layer doesnt need to send any grad past itself
    fa_layers = []
    for i, layer in enumerate(layers[1:]):
        fa_layers.append(FALayer(layer, fa_scale, use_toeplitz=use_toeplitz, mode=mode))
    return nn.Sequential(layers[0], *fa_layers)

