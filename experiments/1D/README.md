# Experiments

## One-dimensional validation

All completed preliminary experiments are grouped in `1D/`, with their figures
in `1D/results/`.

`1D/langevin_gmm_1d.py` validates Langevin dynamics on the known target and
saves plots and diagnostics for the preliminary sampler validation.

`1D/svgd_gmm_1d.py` provides the corresponding one-dimensional SVGD validation.

`1D/score_energy_check.py` numerically verifies that the target score equals the
negative input derivative of its energy. `1D/langevin_stepsize_1d.py` and
`1D/svgd_stepsize_1d.py` compare step-size sensitivity. Finally,
`1D/compare_svgd_langevin_1d.py` starts both methods from the same particles
and compares mode balance and mean log density.

The main experiment will train the same two-dimensional neural EBM twice, using
either Langevin dynamics or SVGD to generate the negative samples. The two
versions will be compared using multiple random seeds and the same computational
budget.

Run it from the repository root with the environment activated:

```bash
python experiments/1D/score_energy_check.py
python experiments/1D/langevin_gmm_1d.py
python experiments/1D/svgd_gmm_1d.py
python experiments/1D/langevin_stepsize_1d.py
python experiments/1D/svgd_stepsize_1d.py
python experiments/1D/compare_svgd_langevin_1d.py
```

The scripts print their numerical diagnostics and save figures to
`experiments/1D/results/`. Together, these experiments complete the preliminary
one-dimensional sampler validation. The next experiments will live in a sibling
`2D/` directory, starting with the multimodal target used by the main experiment.
