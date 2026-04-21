# Interpretability Analysis of Feedback Alignment on CNN

Code accompanying the paper **"Interpretability Analysis of Feedback Alignment on CNN"** by Jake Lance and Larry Kieu. The paper PDF and LaTeX sources are in [`paper/`](paper/).

We compare five learning rules for a convolutional network trained on CIFAR-10, where the methods differ only in how the error signal is propagated backward:

- **BP** — standard backpropagation
- **FA (Random)** — Feedback Alignment with a dense random matrix for convolutional layers
- **FA (Toeplitz)** — Feedback Alignment with a random convolutional kernel (transposed-convolution feedback)
- **uSF Init** — unsigned sign-flip using the initial random magnitudes
- **uSF SN** — unsigned sign-flip with spectral-norm scaling

We then analyse the resulting networks with gradient-alignment / sign-concordance tracking, Centered Kernel Alignment (CKA), and Grad-CAM-based channel importance plus top-activating exemplars.

## Setup

Python 3.11 recommended. PyTorch MPS/CUDA accelerators are auto-detected.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install matplotlib
```

Additional runtime deps used by the plotting scripts: `matplotlib`.

## Reproducing the paper end-to-end

All commands are run from the repo root. CIFAR-10 downloads automatically into `data/`.

```bash
# 1. Train all 5 methods (HP search + 50-epoch training, per-epoch checkpoints).
#    Writes save_cifar101/{method}/epoch_*.pt + tracking_data.json,
#           figures/val_acc_cifar101.png,
#           results/cifar101_results.txt.
python -m scripts.train_cifar101

# 2. Gradient-alignment figure.
python -m scripts.plot_alignment_paper

# 3. Sign-concordance figures (main + supplementary).
python -m scripts.plot_sign_concordance_paper

# 4. CKA heatmaps for each FA variant vs BP, on 3 data subsets.
python -m scripts.generate_cka_figures

# 5. Feature-visualization exemplar grid.
python -m scripts.plot_exemplars_paper

```

## Repository layout

```
src/
  fa.py          FALayer — FA backward-hook, Toeplitz and uSF variants
  models.py      CIFAR101 — Moskovitz Arch 1 for 24x24x3 inputs
  trainer.py     HP search, training loop, checkpointing
  analysis.py    Alignment / sign-agreement tracking and plotting helpers
  data.py        get_device()
scripts/
  train_cifar101.py              Full training pipeline
  plot_alignment_paper.py        Alignment figure
  plot_sign_concordance_paper.py Sign-concordance figures
  generate_cka_figures.py        CKA heatmaps
  plot_exemplars_paper.py        Feature-vis exemplar grid
tests/
  verify_fa_mlp.py               FA correctness test on a small MLP
CKA/                             Third-party CKA implementation (see CKA/LICENSE)
report.pdf                       Paper
results/                         Published-numbers summary
```

## Citation

```bibtex
@misc{lance_kieu_fa_interp_2026,
  title  = {Interpretability Analysis of Feedback Alignment on CNN},
  author = {Lance, Jake and Kieu, Larry},
  year   = {2026}
}
```

## License

Project code: see ArXiv license. Third-party `CKA/` retains its own license (`CKA/LICENSE`).
