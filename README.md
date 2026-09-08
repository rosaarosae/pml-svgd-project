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

Progress is tracked in the [project checklist](TASKS.md).

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

- `experiments/1D/`: completed preliminary sampler validation.
- `experiments/2D/`: main target, sampler, and EBM experiments.
- `notes/`: concise mathematical background needed to understand the project.
- `presentation/`: LaTeX Beamer presentation.
- `references/`: official course material, textbooks, and project papers.
- `TASKS.md`: project-wide checklist and completion criteria.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the experiments

With the environment activated, run the complete validation from the repository
root:

```bash
python experiments/1D/score_energy_check.py
python experiments/1D/langevin_gmm_1d.py
python experiments/1D/svgd_gmm_1d.py
python experiments/1D/langevin_stepsize_1d.py
python experiments/1D/svgd_stepsize_1d.py
python experiments/1D/compare_svgd_langevin_1d.py
```

Each experiment is reproducible from a fixed seed. Numerical diagnostics are
printed in the terminal and generated figures are stored in
`experiments/1D/results/`. See the documentation for the
[one-dimensional experiments](experiments/1D/README.md) and the
[two-dimensional experiments](experiments/2D/README.md).

Run the current two-dimensional target and sampler validation from the
repository root:

```bash
python experiments/2D/gmm_2d.py
python experiments/2D/visualize_gmm_2d.py
python experiments/2D/langevin_2d.py
python experiments/2D/visualize_langevin_2d.py
python experiments/2D/langevin_stepsize_2d.py
python experiments/2D/svgd_2d.py
python experiments/2D/visualize_svgd_2d.py
python experiments/2D/svgd_stepsize_2d.py
python experiments/2D/metrics_2d.py
python experiments/2D/compare_svgd_langevin_2d.py
```

The 2D target now provides its complete density, exact energy, analytic score,
and direct sampler. Langevin and SVGD have both been validated on this known
target, their step sizes have been studied across five random seeds, and the
direct comparison uses shared initial particles and distribution-level metrics.
SVGD most accurately reproduces energy and within-mode geometry, while Langevin
better reproduces global mixture weights and is substantially faster. The next
task is to implement the neural energy function.
