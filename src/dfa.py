import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import random_conv_matrix_2d, random_square_conv_matrix_2d


@torch.no_grad()
def _angle_deg_mean(a, b):
    a = a.reshape(a.size(0), -1).float()
    b = b.reshape(b.size(0), -1).float()
    cos = F.cosine_similarity(a, b, dim=1).clamp(-1, 1)
    return torch.acos(cos).mean().item() * (180.0 / 3.141592653589793)


class DFAContext:
    """Holds the output error shared by all DFA layers in a model."""
    def __init__(self):
        # e in the DFA paper
        self.output_error = None


class DFALayer(nn.Module):
    """Wraps any nn.Module for DFA.

    During training, a hook on the wrapped layer's output replaces the incoming
    gradient with (output_error @ B).view_as(grad), where B is a fixed random
    matrix and output_error comes from the shared DFAContext.
    """
    def __init__(self, layer, num_classes, ctx, scale=0.1):
        super().__init__()
        self.layer = layer
        self.ctx = ctx
        self.num_classes = num_classes
        self.scale = scale
        self._B_initialized = False
        self.track_alignment = False
        self.alignment_history = []
        self.layer_name = ""

    def _init_B(self, out):
        flat_dim = out[0].numel()
        self.register_buffer('B', torch.randn(self.num_classes, flat_dim,
                                               device=out.device) * self.scale)
        self._B_initialized = True

    def forward(self, x):
        out = self.layer(x)
        if not self._B_initialized:
            self._init_B(out)
        if self.training:
            ctx, B = self.ctx, self.B
            dfa_layer = self

            def _dfa_hook(grad, ctx=ctx, B=B):
                new_grad = (ctx.output_error @ B).view_as(grad)
                if dfa_layer.track_alignment:
                    dfa_layer.alignment_history.append(
                        _angle_deg_mean(grad.detach(), new_grad.detach()))
                return new_grad

            out.register_hook(_dfa_hook)
        return out

def init_sequential(layers: nn.Sequential, num_classes, dfa_scale=0.1):
    ctx = DFAContext()
    dfa_layers = [DFALayer(layer, num_classes, ctx, dfa_scale) for layer in layers[:-1]]

    # last layer receives e, no need to replace
    return ctx, nn.Sequential(*dfa_layers, layers[-1])

class DFAModel(nn.Module):
    """Example model using DFALayer."""
    def __init__(self, num_classes=10, dfa_scale=0.1):
        super().__init__()
        self.ctx = DFAContext()
        self.layer1 = DFALayer(nn.Linear(784, 128), num_classes, self.ctx, dfa_scale)
        self.layer2 = DFALayer(nn.Linear(128, 64), num_classes, self.ctx, dfa_scale)
        self.output_layer = nn.Linear(64, num_classes)  # output layer uses true gradient

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.output_layer(x)
        if self.training:
            ctx = self.ctx
            x.register_hook(lambda grad, ctx=ctx: setattr(ctx, 'output_error', grad.detach()) or grad)
        return x