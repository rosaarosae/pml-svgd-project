# Training Energy-Based Models with SVGD

This project studies how the choice of sampler affects the training of a
multimodal **energy-based model (EBM)**. We compare Stein Variational Gradient
Descent (SVGD) with Langevin dynamics when generating the negative samples used
in contrastive training.

## Research question

Can the repulsive interaction between SVGD particles improve mode coverage and
the learned density of a two-dimensional EBM compared with Langevin dynamics
under a comparable computational budget?

## Objectives

- Implement modern, reusable SVGD and Langevin samplers.
- Validate both samplers on a known Gaussian mixture target.
- Train the same neural EBM with contrastive divergence, changing only the
  negative-sample generator.
- Compare density fit, mode coverage, sample quality, convergence, and runtime.
- Study the effect of the number of particles and report results across multiple
  random seeds.

The existing one-dimensional Gaussian mixture is an initial validation example.
The main experiment will use a multimodal two-dimensional Gaussian mixture so
that the true density and learned energy landscape can be evaluated directly.

## Main references

- Q. Liu and D. Wang, “Stein Variational Gradient Descent: A General Purpose
  Bayesian Inference Algorithm,” *Advances in Neural Information Processing
  Systems 29*, 2016. [Paper](https://proceedings.neurips.cc/paper/2016/hash/b3ba8f1bee1238a2f37603d90b58898d-Abstract.html)
- Y. Song and D. P. Kingma, “How to Train Your Energy-Based Models,” 2021.
  [Paper](https://arxiv.org/abs/2101.03288)
- P. Jaini, L. Holdijk, and M. Welling, “Learning Equivariant Energy Based
  Models with Equivariant Stein Variational Gradient Descent,” 2021.
  [Paper](https://arxiv.org/abs/2106.07832)

The complete course and project bibliography is available in
[references/README.md](references/README.md).

## Structure

- `experiments/`: validation, model training, comparisons, and results.
- `notes/`: concise mathematical background needed to understand the project.
- `presentation/`: LaTeX Beamer presentation.
- `references/`: official course material, textbooks, and project papers.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```
