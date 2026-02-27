"""
Verify FA hook-based gradients in a 3-layer MLP against manual computation.

MLP: input(1) -> Linear(1,2)+ReLU [FA] -> Linear(2,2)+ReLU [FA] -> Linear(2,2) [FA] -> output(2)

In FA, every layer (including output) is wrapped in FALayer. The output layer's
weight gradient is identical to BP (it receives the true dL/d(logits) from the loss),
but it sends output_error @ B3 backward instead of output_error @ W3^T.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.fa import FALayer

torch.manual_seed(0)

num_classes = 2

# --- Build layers ---
hidden_layer1 = nn.Linear(1, 2, bias=False)   # W1: (2, 1)
hidden_layer2 = nn.Linear(2, 2, bias=False)   # W2: (2, 2)
output_linear = nn.Linear(2, 2, bias=False)   # W3: (2, 2)

# Fix weights for reproducibility
with torch.no_grad():
    hidden_layer1.weight.copy_(torch.tensor([[0.5], [0.3]]))
    hidden_layer2.weight.copy_(torch.tensor([[0.7, 0.2], [0.4, 0.9]]))
    output_linear.weight.copy_(torch.tensor([[0.2, 0.4], [0.1, 0.6]]))

# Wrap all layers in FALayer
fa_layer1 = FALayer(hidden_layer1, scale=1.0)
fa_layer2 = FALayer(hidden_layer2, scale=1.0)
fa_output = FALayer(output_linear, scale=1.0)

# Scalar input, target
x = torch.tensor([[1.0]])   # shape (1, 1)
target = torch.tensor([0])  # class 0

print('=== Weights ===')
print(f'W1: {hidden_layer1.weight.data}')
print(f'W2: {hidden_layer2.weight.data}')
print(f'W3: {output_linear.weight.data}')

# --- First forward pass to initialize B matrices ---
h1 = torch.relu(fa_layer1(x))
h2 = torch.relu(fa_layer2(h1))
logits = fa_output(h2)

# Force B matrices to known values
B1 = torch.tensor([[1.0], [-0.5]])               # shape (2, 1) — out_dim x in_dim of layer1
B2 = torch.tensor([[0.8, 0.1], [-0.2, 0.6]])    # shape (2, 2) — out_dim x in_dim of layer2
B3 = torch.tensor([[0.4, -0.3], [0.9, 0.2]])    # shape (2, 2) — out_dim x in_dim of output

with torch.no_grad():
    fa_layer1.B.copy_(B1)
    fa_layer2.B.copy_(B2)
    fa_output.B.copy_(B3)

print(f'\nB1: {fa_layer1.B}')
print(f'B2: {fa_layer2.B}')
print(f'B3: {fa_output.B}')

# --- Clean forward pass with known B ---
# Zero any existing grads
for p in [hidden_layer1.weight, hidden_layer2.weight, output_linear.weight]:
    if p.grad is not None:
        p.grad.zero_()

h1 = torch.relu(fa_layer1(x))
h2 = torch.relu(fa_layer2(h1))
logits = fa_output(h2)

print(f'\nh1 (after ReLU): {h1.data}')
print(f'h2 (after ReLU): {h2.data}')
print(f'logits: {logits.data}')

loss = F.cross_entropy(logits, target)
print(f'Loss: {loss.item():.6f}')

# --- Backward pass ---
loss.backward()

print(f'\n=== Autograd Gradients ===')
print(f'W3 grad: {output_linear.weight.grad}')
print(f'W2 grad: {hidden_layer2.weight.grad}')
print(f'W1 grad: {hidden_layer1.weight.grad}')

# --- Manual verification ---
print(f'\n=== Manual Verification ===')

# Step 1: output_error = softmax(logits) - one_hot(target)
softmax = torch.softmax(logits.detach(), dim=1)
output_error = softmax.clone()
output_error[0, target[0]] -= 1
print(f'output_error (dL/d_logits): {output_error}')

h1_det = h1.detach()
h2_det = h2.detach()

# Step 2: W3 grad = output_error^T @ h2 (same as BP)
w3_grad_manual = output_error.T @ h2_det
print(f'W3 grad (manual): {w3_grad_manual}')

# Step 3: Output layer sends output_error @ B3 backward (FA replaces W3^T with B3)
signal_from_output = output_error @ B3  # shape (1, 2)

# ReLU mask for layer2
relu_mask2 = (h2_det > 0).float()
delta2 = signal_from_output * relu_mask2
print(f'delta2 (after ReLU mask): {delta2}')

# W2 grad = delta2^T @ h1
w2_grad_manual = delta2.T @ h1_det
print(f'W2 grad (manual): {w2_grad_manual}')

# Step 4: Layer2 sends delta2 @ B2 backward
signal_from_layer2 = delta2 @ B2  # shape (1, 1)

# ReLU mask for layer1
relu_mask1 = (h1_det > 0).float()
delta1 = signal_from_layer2 * relu_mask1
print(f'delta1 (after ReLU mask): {delta1}')

# W1 grad = delta1^T @ x
w1_grad_manual = delta1.T @ x
print(f'W1 grad (manual): {w1_grad_manual}')

# --- Compare ---
print(f'\n=== Match Check ===')
w3_match = torch.allclose(output_linear.weight.grad, w3_grad_manual, atol=1e-5)
w2_match = torch.allclose(hidden_layer2.weight.grad, w2_grad_manual, atol=1e-5)
w1_match = torch.allclose(hidden_layer1.weight.grad, w1_grad_manual, atol=1e-5)
print(f'W3 grad matches: {w3_match}')
print(f'W2 grad matches: {w2_match}')
print(f'W1 grad matches: {w1_match}')

if w1_match and w2_match and w3_match:
    print('\nAll gradients verified!')
else:
    print('\nMISMATCH DETECTED!')
    if not w3_match:
        print(f'  W3 diff: {(output_linear.weight.grad - w3_grad_manual).abs().max():.2e}')
    if not w2_match:
        print(f'  W2 diff: {(hidden_layer2.weight.grad - w2_grad_manual).abs().max():.2e}')
    if not w1_match:
        print(f'  W1 diff: {(hidden_layer1.weight.grad - w1_grad_manual).abs().max():.2e}')
    sys.exit(1)
