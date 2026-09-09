# One-dimensional SVGD paper reproduction

This directory reproduces the one-dimensional Gaussian-mixture experiment in
Section 5 of Liu and Wang, *Stein Variational Gradient Descent: A General
Purpose Bayesian Inference Algorithm* (NeurIPS 2016):

- Paper: <https://arxiv.org/abs/1608.04471>
- Authors' reference implementation:
  <https://github.com/DartML/Stein-Variational-Gradient-Descent>

Only the one-dimensional experiment is addressed here. The `experiments/2D/`
work is independent and is not changed by this reproduction.

## Published setup

The paper uses the target

```text
p(x) = (1/3) N(-2, 1) + (2/3) N(2, 1)
```

and initializes the particles from

```text
q0(x) = N(-10, 1).
```

Figure 1 transports 100 particles for 500 iterations and displays iterations
0, 50, 75, 100, 150 and 500. The implementation uses the paper's RBF kernel,
adaptive median bandwidth and the AdaGrad update from the authors' released
code.

Figure 2 compares SVGD with independent Monte Carlo samples when estimating
`E[x]`, `E[x^2]` and `E[cos(omega*x + b)]`.

## Primary reproduction files

- `gmm_1d.py` contains the published target, initialization, density, energy,
  score, samplers and exact moments.
- `svgd_gmm_1d.py` implements SVGD and reproduces the six panels in Figure 1.
- `svgd_expectations_1d.py` reproduces the expectation-error comparison in
  Figure 2 and saves both a figure and a CSV table.
- `score_energy_check.py` numerically checks the identity
  `score(x) = -d energy(x)/dx`.
- `RESULTS.md` records the settings, numerical results, interpretation and
  reproduction limitations.

Run the primary reproduction from the repository root:

```bash
python experiments/1D/gmm_1d.py
python experiments/1D/score_energy_check.py
python experiments/1D/svgd_gmm_1d.py
python experiments/1D/svgd_expectations_1d.py
```

The generated artifacts are:

- `results/svgd_gmm_1d.png` for Figure 1;
- `results/svgd_expectations_1d.png` for Figure 2;
- `results/svgd_expectations_1d.csv` for the Figure 2 numerical values.

## Supplementary files

The following scripts are useful additional checks, but they are **not**
presented as reproductions of Figures 1 or 2:

- `langevin_gmm_1d.py`: Langevin baseline on the same target and initialization;
- `compare_svgd_langevin_1d.py`: direct SVGD-Langevin comparison;
- `svgd_stepsize_1d.py`: SVGD step-size sensitivity;
- `langevin_stepsize_1d.py`: Langevin step-size sensitivity.

## Main conclusion

Starting from a distribution with almost no overlap with the target, the SVGD
particles reach both modes and recover their unequal masses. With 100 particles,
the final left/right split is `0.330 / 0.670`, compared with the exact
`1/3 / 2/3`, while the estimated first two moments are close to their analytic
values. The expectation experiment also reproduces the paper's qualitative
finding: the deterministic, interacting SVGD particles estimate all three test
expectations more accurately than the same number of independent Monte Carlo
samples in this experiment.
