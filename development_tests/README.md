# Development tests and optional experiments

This folder documents code that supports development but is not part of the
final professor submission. The original files remain in place so their imports
and Git history are not broken.

## Single-seed EBM development tests

- `experiments/1D/train_ebm_svgd_1d.py`
- `experiments/1D/train_ebm_langevin_1d.py`
- `experiments/1D/svgd_ebm_stepsize_1d.py`
- `experiments/1D/langevin_ebm_stepsize_1d.py`
- `experiments/1D/tune_ebm_samplers_1d.py`

These scripts were used to build each training path and explore learning rates,
particle counts, sampler steps, and sampler step sizes before the final
multi-seed experiment. Their results must be labelled as development results.

## Known-target sampler tests

- `experiments/1D/langevin_gmm_1d.py`
- `experiments/1D/langevin_stepsize_1d.py`
- `experiments/1D/svgd_stepsize_1d.py`
- `experiments/1D/compare_svgd_langevin_1d.py`

These use the exact analytic target score. They test sampler behavior but do
not train a neural energy-based model.

## Optional extension

Everything in `experiments/2D/` is an optional analytic two-dimensional
extension. It is not part of the main neural EBM comparison and should be shown
only if presentation time permits.

## Generated exploratory files

Figures whose names contain particle counts, learning rates, or sampler-step
counts are retained as an audit trail of development choices. The final
submission uses the explicitly named comparison figure and final CSV instead.
