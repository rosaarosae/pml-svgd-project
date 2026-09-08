# Two-dimensional experiments

This directory contains the main two-dimensional Gaussian-mixture target and
sampler-validation experiments. Generated figures are stored in `results/`.

`gmm_2d.py` defines four equally weighted isotropic Gaussian components and
implements the complete mixture density, exact target energy, analytic score,
direct sampling, and numerical consistency checks. `visualize_gmm_2d.py`
displays the density and direct target samples.

`langevin_2d.py` and `svgd_2d.py` contain reusable two-dimensional samplers.
Their corresponding visualization scripts show the particles before and after
sampling. The step-size scripts compare six candidate values visually and
report mean target energy, mixture-weight error, runtime, and uncertainty across
five seeds.

Run all current 2D experiments from the repository root:

```bash
python experiments/2D/gmm_2d.py
python experiments/2D/visualize_gmm_2d.py
python experiments/2D/langevin_2d.py
python experiments/2D/visualize_langevin_2d.py
python experiments/2D/langevin_stepsize_2d.py
python experiments/2D/svgd_2d.py
python experiments/2D/visualize_svgd_2d.py
python experiments/2D/svgd_stepsize_2d.py
```

The selected validation settings are a step size of `0.01` for Langevin and
`1.0` for SVGD, with 500 particles and 1,000 updates. Under these equal-step
conditions, both methods reach a mean target energy close to the true target
value. SVGD is more consistent in energy but remains slower and shows greater
mixture-weight error. A direct comparison with shared initial particles and
additional distribution-level metrics is the next task.
