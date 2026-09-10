# Development tests

This folder is only a guide to the exploratory work in the repository. The
tests themselves remain beside the code they exercise, so there are no copied
or competing versions of the same program.

These checks helped answer practical questions before the final experiment:

- Does the exact Gaussian-mixture density and score behave correctly?
- Do SVGD and Langevin particles move and remain finite?
- Which ranges of step sizes are stable?
- Does each single-method EBM training loop work before combining them?
- How do the samplers behave when the target density is already known?

Most of these scripts are in `experiments/1D` and have names containing
`stepsize`, `tune`, `check`, or `train_ebm`. They are development evidence, not
additional headline results. The final comparison is
`experiments/1D/compare_ebm_svgd_langevin_1d.py`.

The `experiments/2D` folder is a separate optional extension. It is useful as a
further sampler check, but the project's main conclusion comes from the
one-dimensional neural EBM experiment.
