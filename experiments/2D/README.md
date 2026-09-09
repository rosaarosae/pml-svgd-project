# Optional two-dimensional sampler extension

This directory contains a completed analytic comparison of SVGD and Langevin
on a paper-inspired two-dimensional Gaussian mixture. It is retained as an
optional extension and possible source of supplementary results. The main
project trains and compares neural EBMs in `experiments/1D/`.

The original SVGD paper publishes a one-dimensional Gaussian-mixture toy
experiment and does not define a 2D version. This directory therefore must not
be described as a reproduction of a published 2D experiment.

## Project-defined target

The first coordinate preserves the unequal mixture from Liu and Wang (2016),
while the second coordinate adds an independent standard normal:

```text
p(x1, x2) = [(1/3) N(x1; -2, 1) + (2/3) N(x1; 2, 1)] N(x2; 0, 1).
```

Equivalently, the component means are `(-2, 0)` and `(2, 0)`, the weights are
`1/3` and `2/3`, and both covariance matrices are the identity. Initial
particles use the corresponding project-defined extension

```text
q0(x1, x2) = N((-10, 0), I).
```

The added dimension, this initialization, the Langevin baseline, seeds, and
evaluation metrics are project choices. The 1D mixture, particle count, SVGD
kernel, and update structure are traced to the original paper and its released
implementation.

## Implemented components

- `gmm_2d.py`: exact density, energy, analytic score, and direct sampling.
- `visualize_gmm_2d.py`: target-density and sample visualization.
- `svgd_2d.py` and `visualize_svgd_2d.py`: SVGD sampler and visualization.
- `langevin_2d.py` and `visualize_langevin_2d.py`: Langevin baseline.
- `svgd_stepsize_2d.py` and `langevin_stepsize_2d.py`: five-seed sensitivity
  checks.
- `metrics_2d.py`: distribution-level metrics.
- `compare_svgd_langevin_2d.py`: paired five-seed sampler comparison.
- `RESULTS.md`: numerical results and their limitations.

## Run the optional experiment

From the repository root:

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

Figures and CSV tables are saved in `results/`.

## Role in the final project

The 2D results validate that both samplers can handle a known multidimensional
target and illustrate their different accuracy/runtime trade-offs. They do not
show which sampler trains a better neural EBM. That question is answered only
by the matched 1D EBM experiment.

Recommended presentation priority:

1. include the completed 1D paper reproduction;
2. present the 1D EBM comparison between SVGD and Langevin;
3. show one 2D figure only if time remains;
4. otherwise describe 2D neural EBM training as future work.
