# Experiments

## Initial validation

`validate_score_identity.py` verifies numerically that the score of an EBM is
the negative input gradient of its energy and visualises the density, energy,
and score of a Gaussian mixture.

`svgd_gmm_1d.py` is a small one-dimensional Gaussian mixture example used to
validate the initial SVGD implementation. It is not the main project experiment.

`langevin_gmm_1d.py` validates Langevin dynamics on the same target and saves
the same plots and diagnostics for a consistent preliminary comparison.

`langevin_stepsize_1d.py` checks a small, medium, and large Langevin step while
holding the initial particles, noise sequence, and number of updates fixed.

The main experiment will train the same two-dimensional neural EBM twice, using
either Langevin dynamics or SVGD to generate the negative samples. The two
versions will be compared using multiple random seeds and the same computational
budget.

Run it from the repository root with the environment activated:

```bash
python experiments/validate_score_identity.py
python experiments/langevin_gmm_1d.py
python experiments/langevin_stepsize_1d.py
python experiments/svgd_gmm_1d.py
```

The output figures are saved to `experiments/results/`.
