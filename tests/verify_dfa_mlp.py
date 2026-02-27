"""
Verify DFA hook-based gradients in a simple scalar 2-layer MLP
against manual computation.

MLP: input(1) -> Linear(1,2) + ReLU [DFA-wrapped] -> Linear(2,2) -> output(2)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.dfa import DFAContext, DFALayer

torch.manual_seed(0)

num_classes = 2

# Build model manually with DFALayer
ctx = DFAContext()

hidden_layer1 = nn.Linear(1, 2, bias=False)  # W1: (2, 1)
hidden_layer2 = nn.Linear(2, 2, bias=False)  # W2: (2, 2)
output_layer = nn.Linear(2, 2, bias=False)  # W3: (2, 2)

# Fix weights for reproducibility
with torch.no_grad():
    hidden_layer1.weight.copy_(torch.tensor([[0.5], [0.3]]))
    hidden_layer2.weight.copy_(torch.tensor([[0.7, 0.2], [0.4, 0.9]]))
    output_layer.weight.copy_(torch.tensor([[0.2, 0.4], [0.1, 0.6]]))

# Wrap hidden layer in DFALayer with a known B
dfa_hidden1 = DFALayer(nn.Sequential(hidden_layer1, nn.ReLU()), num_classes, ctx, scale=1.0)
dfa_hidden2 = DFALayer(nn.Sequential(hidden_layer2, nn.ReLU()), num_classes, ctx, scale=1.0)

# Scalar input, target
x = torch.tensor([[1.0]])  # shape (1, 1)
target = torch.tensor([0])  # class 0

print('=== Weights ===')
print(f'W1 (hidden): {hidden_layer1.weight.data}')
print(f'W2 (hidden): {hidden_layer2.weight.data}')
print(f'W3 (output): {output_layer.weight.data}')

# --- Forward pass ---
h1 = dfa_hidden1(x)  # hidden + relu output
h2 = dfa_hidden2(h1)  # hidden + relu output
logits = output_layer(h2)

# Register logit hook to capture output_error
ctx_ref = ctx
logits.register_hook(lambda grad, c=ctx_ref: setattr(c, 'output_error', grad.detach()) or grad)

print(f'\nHidden output 1 (after ReLU): {h1.data}')
print(f'Hidden output 2 (after ReLU): {h2.data}')
print(f'Logits: {logits.data}')

# Force B to a known value for manual verification
with torch.no_grad():
    B_known = torch.tensor([[1.0, -0.5], [0.3, 0.7]])  # shape (num_classes, hidden_dim)
    dfa_hidden1.B.copy_(B_known)
    dfa_hidden2.B.copy_(B_known)  # same B for both layers for simplicity
print(f'B matrix: {dfa_hidden1.B}, in both layers? {torch.allclose(dfa_hidden1.B, dfa_hidden2.B)}')

# Re-run forward with the known B
dfa_hidden1._B_initialized = True  # already initialized
dfa_hidden2._B_initialized = True
h1_dfa = dfa_hidden1(x)
h2_dfa = dfa_hidden2(h1_dfa)
logits_dfa = output_layer(h2_dfa)
print(f'\nHidden output 1 (after ReLU, second pass): {h1_dfa.data}')
print(f'Hidden output 2 (after ReLU, second pass): {h2_dfa.data}')
logits_dfa.register_hook(lambda grad, c=ctx_ref: setattr(c, 'output_error', grad.detach()) or grad)

loss = F.cross_entropy(logits_dfa, target)
print(f'\nLoss: {loss.item():.6f}')

# --- Backward pass ---
loss.backward()

print(f'\n=== Results ===')
print(f'Output error (dL/d_logits): {ctx.output_error}')
print(f'W3 grad (autograd):         {output_layer.weight.grad}')
print(f'W2 grad (DFA + autograd):   {hidden_layer2.weight.grad}')
print(f'W1 grad (DFA + autograd):   {hidden_layer1.weight.grad}')

# --- Manual verification ---
print(f'\n=== Manual Verification ===')
softmax = torch.softmax(logits_dfa.detach(), dim=1)
output_error = softmax.clone()
output_error[0, target[0]] -= 1
print(f'Manual output_error: {output_error}')

# W3 grad: output_error^T @ h_out  (standard BP for output layer)
w3_grad_manual = output_error.T @ h2_dfa.detach()
print(f'W3 grad (manual):    {w3_grad_manual}')

# DFA hook replaces grad at hidden output with (output_error @ B)
dfa_signal = output_error @ B_known  # shape (1, 2)
print(f'DFA signal at hidden output: {dfa_signal}')

# ReLU backward: multiply by (h > 0)
h2_pre_relu = hidden_layer1(x)  # before ReLU
relu_mask = (h2_pre_relu > 0).float()
dfa_through_relu2 = dfa_signal * relu_mask.detach()
print(f'delta_h2 After ReLU backward: {dfa_through_relu2}')

# W2 grad: dfa_through_relu^T @ x
w2_grad_manual = dfa_through_relu2.T @ h2_pre_relu
print(f'W2 grad (manual):    {w2_grad_manual}')

# ReLU backward: multiply by (h > 0)
h1_pre_relu = hidden_layer1(x)  # before ReLU
relu_mask = (h1_pre_relu > 0).float()
dfa_through_relu = dfa_signal * relu_mask.detach()
print(f'delta_h1 After ReLU backward: {dfa_through_relu}')

# W1 grad: dfa_through_relu^T @ x
w1_grad_manual = dfa_through_relu.T @ x
print(f'W1 grad (manual):    {w1_grad_manual}')

# Compare
print(f'\n=== Match Check ===')
w3_match = torch.allclose(output_layer.weight.grad, w3_grad_manual, atol=1e-5)
w2_match = torch.allclose(hidden_layer2.weight.grad, w2_grad_manual, atol=1e-5)
w1_match = torch.allclose(hidden_layer1.weight.grad, w1_grad_manual, atol=1e-5)
print(f'W3 grad matches: {w3_match}')
print(f'W2 grad matches: {w2_match}')
print(f'W1 grad matches: {w1_match}')
print(f'\nAll gradients verified!' if (w1_match and w2_match and w3_match) else '\nMISMATCH DETECTED!')
