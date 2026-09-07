# Experiments

## Initial validation

`langevin_gmm_1d.py` validates Langevin dynamics on the same target and saves
plots and diagnostics for the preliminary sampler validation.

`svgd_gmm_1d.py` provides the corresponding one-dimensional SVGD validation.

`score_energy_check.py` numerically verifies that the target score equals the
negative input derivative of its energy. `langevin_stepsize_1d.py` and
`svgd_stepsize_1d.py` compare step-size sensitivity. Finally,
`compare_svgd_langevin_1d.py` starts both methods from the same particles and
compares mode balance and mean log density.

The main experiment will train the same two-dimensional neural EBM twice, using
either Langevin dynamics or SVGD to generate the negative samples. The two
versions will be compared using multiple random seeds and the same computational
budget.

Run it from the repository root with the environment activated:

```bash
python experiments/score_energy_check.py
python experiments/langevin_gmm_1d.py
python experiments/svgd_gmm_1d.py
python experiments/langevin_stepsize_1d.py
python experiments/svgd_stepsize_1d.py
python experiments/compare_svgd_langevin_1d.py
```

The scripts print their numerical diagnostics and save figures to
`experiments/results/`. Together, these experiments complete the preliminary
one-dimensional sampler validation. The next step is to define and validate the
multimodal two-dimensional target used by the main experiment.
