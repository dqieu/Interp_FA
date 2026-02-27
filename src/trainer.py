import copy
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += output.argmax(1).eq(target).sum().item()
        total += target.size(0)
    return total_loss / len(loader), 100. * correct / total


def evaluate(model, loader, device):
    model.eval()
    test_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            correct += output.argmax(1).eq(target).sum().item()
            total += target.size(0)
    return test_loss / total, 100. * correct / total


def run_config(model_fn, method, cfg, train_loader, val_loader, device, epochs):
    torch.manual_seed(42)
    model = model_fn(method, cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'],
                           weight_decay=cfg.get('wd', 0.0))
    for _ in range(epochs):
        train_epoch(model, train_loader, optimizer, device)
    _, val_acc = evaluate(model, val_loader, device)
    return val_acc


def hp_search(model_fn, method, train_loader, val_loader, device, epochs=3, configs=None):
    print(f"\n--- HP Search: {method} ({epochs} epochs each) ---")
    if configs is None:
        if method == 'BP':
            configs = [{'lr': lr, 'wd': wd}
                       for lr in [0.0005, 0.001, 0.003]
                       for wd in [0.0, 1e-4]]
        elif method == 'FA_uSF_sn':
            # scale is irrelevant for uSF SN (B fully determined by W)
            configs = [{'lr': lr, 'scale': 0.1}
                       for lr in [0.001, 0.003, 0.01, 0.03]]
        else:
            configs = [{'lr': lr, 'scale': sc}
                       for lr in [0.001, 0.003, 0.01]
                       for sc in [0.01, 0.05, 0.1]]

    best_acc, best_cfg = -1, configs[0]
    for cfg in configs:
        val_acc = run_config(model_fn, method, cfg, train_loader, val_loader, device, epochs)
        marker = "  <-- best so far" if val_acc > best_acc else ""
        if val_acc > best_acc:
            best_acc, best_cfg = val_acc, cfg
        print(f"  {cfg}  =>  val acc: {val_acc:.2f}%{marker}")

    print(f"\n  Best config: {best_cfg}  (val acc: {best_acc:.2f}%)")
    return best_cfg


def train_and_evaluate(model_fn, method, cfg, train_loader, val_loader, test_loader, device,
                       save_dir, max_epochs=50, track_alignment=False):
    print(f"\n{'='*60}")
    print(f"Final Training — {method}  |  config: {cfg}")
    print(f"  max_epochs={max_epochs}")
    print(f"{'='*60}")

    torch.manual_seed(42)
    model = model_fn(method, cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'],
                           weight_decay=cfg.get('wd', 0.0))

    if track_alignment and method != 'BP':
        from .analysis import enable_alignment_tracking
        enable_alignment_tracking(model)

    if track_alignment and method in ('FA', 'FA_toeplitz'):
        from .analysis import enable_sign_tracking
        enable_sign_tracking(model)

    method_dir = Path(save_dir) / method
    os.makedirs(method_dir, exist_ok=True)

    # Epoch 0: save pre-training checkpoint
    # Dummy forward to initialize B buffers for FA/DFA
    was_training = model.training
    model.eval()
    with torch.no_grad():
        batch = next(iter(train_loader))
        dummy = batch[0][:2].to(device)
        model(dummy)
    if was_training:
        model.train()
    torch.save(model.state_dict(), method_dir / 'epoch_000.pt')
    print(f"  Saved epoch 0 (pre-training) checkpoint")

    best_val_acc, best_state, best_epoch = -1, None, 0

    results = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(max_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)

        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['val_loss'].append(val_loss)
        results['val_acc'].append(val_acc)

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            marker = " *"

        torch.save(model.state_dict(), method_dir / f'epoch_{epoch+1:03d}.pt')

        print(f"  Epoch {epoch+1}/{max_epochs} | "
              f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%{marker}")

    torch.save(best_state, method_dir / 'best.pt')
    model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"\n  Best Val Acc: {best_val_acc:.2f}% (epoch {best_epoch})")
    print(f"  Test Accuracy: {test_acc:.2f}%  (Test Loss: {test_loss:.4f})")
    print(f"  Checkpoints saved to {method_dir}/")

    results['test_acc'] = test_acc
    results['test_loss'] = test_loss
    results['best_epoch'] = best_epoch
    results['best_val_acc'] = best_val_acc

    if track_alignment and method != 'BP':
        from .analysis import collect_alignment_data
        results['alignment'] = collect_alignment_data(model)

    if track_alignment and method in ('FA', 'FA_toeplitz'):
        from .analysis import collect_sign_data
        results['sign_agreement'] = collect_sign_data(model)

    return results
