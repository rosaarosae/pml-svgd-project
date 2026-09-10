# One-dimensional experiments

This folder contains the main project and the smaller experiments that helped
me build and check it.

## Start here

The finished experiment is:

```text
compare_ebm_svgd_langevin_1d.py
```

It trains the same neural energy-based model with two different negative
samplers, SVGD and Langevin, over 10 paired random seeds. It then compares the
learned densities, test negative log-likelihood, moments, particle statistics,
stability, and runtime.

Run it from this folder with:

```bash
python compare_ebm_svgd_langevin_1d.py
```

The full comparison may take about 20 minutes. To recreate the figures from
the saved outputs without training again, use:

```bash
python compare_ebm_svgd_langevin_1d.py --plot-only
```

## Files used by the final comparison

- `gmm_1d.py` defines the exact two-component Gaussian mixture.
- `energy_model.py` defines the neural energy model and its score.
- `svgd_ebm_1d.py` implements SVGD updates for PyTorch particles.
- `langevin_ebm_1d.py` implements Langevin updates with drift and noise.
- `compare_ebm_svgd_langevin_1d.py` trains, evaluates, saves, and plots both
  methods under matched conditions.
- `results/ebm_svgd_langevin_neutral_init_1d.csv` stores the 20 final runs.
- `results/ebm_svgd_langevin_comparison_1d.png` compares accuracy and runtime.
- `results/ebm_svgd_langevin_curves_1d.png` compares the exact and learned
  density and energy curves.

## Final setup

Both methods use 1,000 epochs, batches of 200 examples, 500 persistent
particles, 20 sampler steps per epoch, and the same learning-rate schedule. The
SVGD step size is `0.02` and the Langevin step size is `0.05`. These values were
chosen in development tests and frozen before the final ten-seed run.

The final comparison found that SVGD learned the density more accurately,
while Langevin finished about 8.6 times faster. Every run was stable.

## Development and supporting files

The other scripts are useful checks, but they are not separate final results:

- `train_ebm_svgd_1d.py` and `train_ebm_langevin_1d.py` train one method at a
  time and were used while building the final loop.
- Files containing `stepsize` or `tune` explore sampler settings.
- `compare_svgd_langevin_1d.py` compares the samplers directly on the known
  target, without training a neural EBM.
- `score_energy_check.py` checks the relationship between energy and score.
- `svgd_gmm_1d.py` and `svgd_expectations_1d.py` are supporting checks based on
  examples from the original SVGD work.

The saved development figures remain in `results/` so the path from early
experiments to the final design is visible. For a clean folder to send to the
professor, use [`../../professor_submission`](../../professor_submission/).
