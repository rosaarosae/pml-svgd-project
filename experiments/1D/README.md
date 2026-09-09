# Main one-dimensional experiments

This directory contains both the completed reproduction of the original SVGD
toy experiment and the main neural energy-based model study.

The common target is

```text
p_data(x) = (1/3) N(-2, 1) + (2/3) N(2, 1).
```

Using the same analytic target for every stage gives the project a controlled
progression: first validate the samplers when the exact score is known, then
use those samplers to generate negative examples while learning an unknown
neural energy from data samples.

## Part A: reproduction of Liu and Wang (2016)

Section 5 of [Liu and Wang (2016)](https://arxiv.org/abs/1608.04471)
initializes 100 particles from

```text
q0(x) = N(-10, 1)
```

and transports them towards the unequal mixture. The reproduction uses the
paper's RBF kernel and adaptive median bandwidth, together with the AdaGrad
update in the authors' released implementation.

### Reproduction files

- `gmm_1d.py`: exact target density, energy, score, samplers, and moments.
- `score_energy_check.py`: numerical check of
  `score(x) = -d energy(x) / dx`.
- `svgd_gmm_1d.py`: reproduction of the six snapshots in Figure 1.
- `svgd_expectations_1d.py`: reproduction of the expectation comparison in
  Figure 2, including its CSV output.
- `RESULTS.md`: verified configuration, results, interpretation, and limits.

Run this completed stage from the repository root:

```bash
python experiments/1D/gmm_1d.py
python experiments/1D/score_energy_check.py
python experiments/1D/svgd_gmm_1d.py
python experiments/1D/svgd_expectations_1d.py
```

The generated artifacts are:

- `results/svgd_gmm_1d.png` for Figure 1;
- `results/svgd_expectations_1d.png` for Figure 2;
- `results/svgd_expectations_1d.csv` for the Figure 2 values.

## Part B: main neural EBM experiment

The model learns a scalar energy `E_theta(x)` from samples of the mixture and
defines

```text
p_theta(x) = exp(-E_theta(x)) / Z_theta.
```

The score supplied to both negative samplers is

```text
score_theta(x) = -grad_x E_theta(x).
```

The training gradient contains a positive phase from real data and a negative
phase from the current model. The main run generates the negative particles
with SVGD. The required baseline trains a new copy of the identical EBM using
Langevin dynamics instead. Only the negative sampler may change in the final
comparison.

### Current EBM files

- `energy_model.py`: smooth neural energy, confining term, automatic score, and
  basic checks.
- `svgd_ebm_1d.py`: PyTorch RBF kernel, adaptive bandwidth, SVGD direction, and
  repeated particle updates.
- `train_ebm_svgd_1d.py`: work in progress for positive samples, persistent
  negative particles, and alternating model updates.

Current development checks:

```bash
python experiments/1D/energy_model.py
python experiments/1D/svgd_ebm_1d.py
python experiments/1D/train_ebm_svgd_1d.py
```

Passing these checks currently verifies the components and their tensor shapes;
it does not yet demonstrate a successfully trained EBM.

### Remaining EBM work

1. complete and stabilize SVGD-based EBM training;
2. implement the matching Langevin negative sampler and training run;
3. normalize the learned 1D densities numerically;
4. compare density, energy, mode weights, moments, NLL, error, and runtime;
5. repeat the matched comparison over at least five seeds.

## Supporting sampler analyses

These programs use the exact target score and support the experimental design,
but they are not reproductions of Figures 1 or 2 and are not substitutes for
the learned-EBM comparison:

- `langevin_gmm_1d.py`: Langevin on the known target;
- `langevin_stepsize_1d.py`: Langevin step-size sensitivity;
- `svgd_stepsize_1d.py`: supplementary SVGD step-size sensitivity;
- `compare_svgd_langevin_1d.py`: direct known-target sampler comparison.

## Scope of the 2D work

The sibling `experiments/2D/` directory is an optional analytic sampler
extension. It is not the main neural EBM experiment and should be included in
the final presentation only if the 1D comparison is complete and time permits.
