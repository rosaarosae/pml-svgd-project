# Training a One-Dimensional Energy-Based Model with SVGD

This project studies Stein Variational Gradient Descent (SVGD) as a sampler for
training a neural energy-based model (EBM). The main experiment uses the
one-dimensional unequal Gaussian mixture from Liu and Wang (2016), for which
the exact density and moments are known. This makes it possible to evaluate the
learned EBM quantitatively rather than relying only on sample visualizations.

The project has two main stages:

1. reproduce the published 1D SVGD toy experiment and validate the sampler;
2. train the same neural EBM twice, using either SVGD or Langevin dynamics to
   generate the negative samples.

The existing 2D work is retained as an optional sampler study and possible
future extension. It is not the main EBM experiment and is not a reproduction
of a published 2D experiment.

## Research question

When training the same neural EBM on a known bimodal 1D distribution, how does
the choice between SVGD and Langevin negative samples affect the learned
density, mode proportions, stability, and computational cost?

## Experimental design

The data distribution is the target used in Section 5 of the original SVGD
paper:

```text
p_data(x) = (1/3) N(-2, 1) + (2/3) N(2, 1).
```

The neural model defines

```text
p_theta(x) = exp(-E_theta(x)) / Z_theta.
```

Its score is available without evaluating the partition function:

```text
grad_x log p_theta(x) = -grad_x E_theta(x).
```

Training uses positive data samples and negative samples from the current EBM.
The primary method generates the negatives with SVGD; the required baseline
uses Langevin dynamics. The architecture, data, optimizer, initialization,
number of particles, training schedule, and evaluation remain fixed between
the two runs. Only the negative sampler changes.

## Scope and attribution

- The 1D target, original SVGD update, RBF kernel, and published toy results
  come from Liu and Wang (2016).
- The use of SVGD to generate negative samples for EBM training is motivated by
  prior work on learning energy models with Stein variational methods.
- The small 1D neural architecture, confining term, seeds, training budget, and
  evaluation protocol are explicit project choices, not settings reported in
  the original 1D paper.
- Langevin is a controlled baseline chosen for this project; it is not part of
  Figures 1 or 2 of Liu and Wang (2016).
- The 2D directory is a paper-inspired extension created by the project and is
  optional.

## Main references

- Q. Liu and D. Wang, “Stein Variational Gradient Descent: A General Purpose
  Bayesian Inference Algorithm,” NeurIPS 2016.
  [Paper](https://arxiv.org/abs/1608.04471)
- Q. Liu and D. Wang, “Learning Deep Energy Models: Contrastive Divergence vs.
  Amortized MLE,” 2017. [Paper](https://arxiv.org/abs/1707.00797)
- Y. Song and D. P. Kingma, “How to Train Your Energy-Based Models,” 2021.
  [Paper](https://arxiv.org/abs/2101.03288)
- P. Jaini, L. Holdijk, and M. Welling, “Learning Equivariant Energy Based
  Models with Equivariant Stein Variational Gradient Descent,” 2021.
  [Paper](https://arxiv.org/abs/2106.07832)

The full bibliography and the role of each reference are documented in
[references/README.md](references/README.md).

## Repository structure

- `experiments/1D/`: main project: paper reproduction, neural EBM, SVGD, and
  Langevin baseline.
- `experiments/2D/`: optional analytic sampler extension; not the main EBM.
- `notes/`: mathematical foundations for the project.
- `presentation/`: plan and future source files for the final Beamer slides.
- `references/`: course material and project bibliography.
- `TASKS.md`: current project status and completion criteria.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the completed 1D paper reproduction

From the repository root:

```bash
python experiments/1D/gmm_1d.py
python experiments/1D/score_energy_check.py
python experiments/1D/svgd_gmm_1d.py
python experiments/1D/svgd_expectations_1d.py
```

The generated figures and numerical table are stored in
`experiments/1D/results/`. See
[experiments/1D/RESULTS.md](experiments/1D/RESULTS.md) for the verified results
and limitations.

## Run the current SVGD- and Langevin-trained EBMs

Both training paths now complete a 500-epoch single-seed run with persistent
negative particles, numerically normalize the learned density, report density
error and left/right mass, and save a comparison figure:

```bash
python experiments/1D/energy_model.py
python experiments/1D/svgd_ebm_1d.py
python experiments/1D/svgd_ebm_stepsize_1d.py
python experiments/1D/train_ebm_svgd_1d.py
python experiments/1D/langevin_ebm_1d.py
python experiments/1D/langevin_ebm_stepsize_1d.py
python experiments/1D/train_ebm_langevin_1d.py
```

The shared single-seed configuration uses 200 positive samples, 200 persistent
particles, 20 sampler steps per epoch, Adam learning rate `1e-3`, and 500
epochs. The selected sampler step sizes are `0.05` for SVGD and `0.1` for
Langevin. SVGD obtains integrated squared density error `0.00205` and mass
`0.318 / 0.682`; Langevin obtains error `0.00272` and mass `0.358 / 0.642`,
compared with exact grid mass `0.3409 / 0.6591`. These are development results
from one seed, not final multi-seed comparison statistics.

The main outputs are `experiments/1D/results/ebm_svgd_1d.png` and
`experiments/1D/results/ebm_langevin_1d.png`. Additional files with
configuration suffixes preserve the controlled checks used to select the
current SVGD configuration.

## Supplementary sampler checks

The following 1D programs are supporting analyses rather than reproductions of
the paper's Figures 1 and 2:

```bash
python experiments/1D/langevin_gmm_1d.py
python experiments/1D/langevin_stepsize_1d.py
python experiments/1D/svgd_stepsize_1d.py
python experiments/1D/compare_svgd_langevin_1d.py
```

Instructions for the optional 2D study are kept in
[experiments/2D/README.md](experiments/2D/README.md).

## Current status

The exact 1D paper reproduction and the first evaluated, single-seed neural EBM
runs with SVGD and Langevin are complete. The shared development configuration
is frozen. The next milestone is to compare both learned densities over at
least five random seeds and report aggregate metrics and runtime.
