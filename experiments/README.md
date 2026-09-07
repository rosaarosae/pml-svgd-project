# Experiments

## Initial validation

`langevin_gmm_1d.py` validates Langevin dynamics on the same target and saves
plots and diagnostics for the preliminary sampler validation.

The next validation steps are to implement the score–energy identity check,
study Langevin step-size sensitivity, implement SVGD, and compare both samplers
on the same target. Progress is recorded in the project checklist.

The main experiment will train the same two-dimensional neural EBM twice, using
either Langevin dynamics or SVGD to generate the negative samples. The two
versions will be compared using multiple random seeds and the same computational
budget.

Run it from the repository root with the environment activated:

```bash
python experiments/langevin_gmm_1d.py
```

The output figures are saved to `experiments/results/`.
