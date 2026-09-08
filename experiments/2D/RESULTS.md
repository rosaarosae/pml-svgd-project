# Two-dimensional sampler validation results

## Experimental design

The direct validation compares Langevin dynamics and SVGD against the known
four-component Gaussian mixture before either sampler is used to train a neural
energy-based model.

- Both methods use 500 particles and 1,000 score evaluations per particle.
- Every paired run starts from exactly the same particles.
- Results are repeated with seeds 7, 17, 27, 37, and 47.
- Langevin uses step size 0.01; SVGD uses step size 1.0.
- Step sizes were selected in separate five-seed validation experiments.
- An independent exact GMM sample is used as the distributional reference for
  each seed.
- A second exact sample provides a finite-sample baseline called `Target`.

The numerical expected target energy is 2.837690. It is obtained by integrating
the known density and energy on a dense two-dimensional grid.

## Metrics

- **Mean target energy:** average exact GMM energy of the particles.
- **Energy error:** absolute difference from the numerically integrated expected
  target energy.
- **Mode coverage:** number of modes receiving at least 1% of the particles.
- **Mixture-weight error:** total absolute difference from the four true weights
  of 0.25.
- **Mode-centre error:** average distance between empirical and true mode means.
- **Within-mode variance error:** average error from the true per-coordinate
  variance of 0.25.
- **Sliced Wasserstein:** projection-based distance to an independent exact
  target sample; lower values indicate a better overall distributional match.
- **Runtime:** wall-clock sampler time. Target sampling is shown only as a
  statistical baseline and is not included in the runtime comparison.

## Results

All values are mean ± sample standard deviation across five seeds.

| Method | Mean energy | Energy error | Modes | Weight error |
|---|---:|---:|---:|---:|
| Exact target sample | 2.807 ± 0.056 | 0.053 ± 0.029 | 4.0 | 0.066 ± 0.042 |
| Langevin | 2.841 ± 0.041 | 0.034 ± 0.017 | 4.0 | 0.077 ± 0.056 |
| SVGD | 2.835 ± 0.005 | 0.005 ± 0.003 | 4.0 | 0.122 ± 0.050 |

| Method | Sliced-W2 | Centre error | Variance error | Runtime |
|---|---:|---:|---:|---:|
| Exact target sample | 0.262 ± 0.146 | 0.044 ± 0.016 | 0.021 ± 0.005 | — |
| Langevin | 0.321 ± 0.036 | 0.051 ± 0.018 | 0.016 ± 0.004 | 0.292 ± 0.012 s |
| SVGD | 0.370 ± 0.043 | 0.002 ± 0.001 | 0.002 ± 0.001 | 7.574 ± 0.240 s |

The paired result is consistent across seeds:

- SVGD has lower energy, centre, and within-mode variance errors in all five
  paired runs.
- Langevin has lower mixture-weight error and runtime in all five paired runs.
- Langevin has lower sliced-Wasserstein distance in four of five runs.
- Both methods cover all four modes in every run.

## Interpretation

SVGD gives an extremely regular local approximation. Its particles reproduce
the target energy, mode centres, and within-mode variance more accurately and
with less seed-to-seed variation than independent Langevin particles. This is
consistent with the repulsive interaction organizing particles within each
mode.

The same repulsion does not correct the number of particles assigned to distant
modes. Once separated particle groups form, an RBF kernel couples them only
weakly. Consequently, SVGD preserves more of the global imbalance inherited
from initialization. Langevin noise permits some boundary crossing and produces
better mixture weights in this experiment. The global effect dominates sliced
Wasserstein distance, so Langevin obtains the better overall distributional
score despite its less regular local geometry.

SVGD is also substantially slower for this analytic target because its kernel
requires pairwise particle interactions. Langevin updates particles
independently. The comparison fixes particle count, update count, and score
evaluations, then reports runtime rather than hiding this algorithmic cost.
Runtime values are machine-dependent and should be interpreted as a relative
comparison from the same run.

These results do not establish that Langevin will train the better neural EBM.
Negative-sample diversity and regularity can affect learning differently from
standalone target approximation. The next stage therefore keeps the model,
data, optimizer, initialization, and training budget fixed and changes only the
negative sampler.

## Outputs

- `results/compare_svgd_langevin_2d_particles.png`: representative particles.
- `results/compare_svgd_langevin_2d_metrics.png`: metric means and uncertainty.
- `results/compare_svgd_langevin_2d_per_seed.csv`: every individual result.
- `results/compare_svgd_langevin_2d_summary.csv`: aggregated results.
