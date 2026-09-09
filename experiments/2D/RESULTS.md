# Results of the optional paper-inspired 2D sampler experiment

## Scope

Liu and Wang (2016) publish a 1D Gaussian-mixture example, not a 2D one. This
optional experiment preserves the published mixture in the first coordinate
and adds an independent standard-normal second coordinate. It should therefore
be described as a controlled project extension, never as the paper's Figure 1
or Figure 2. The main learned-EBM comparison is one-dimensional.

## Experimental design

- Target: `(1/3) N((-2,0), I) + (2/3) N((2,0), I)`.
- Initial distribution: `N((-10,0), I)`.
- Both samplers use 100 particles and 500 score evaluations per particle.
- Results use seeds 7, 17, 27, 37 and 47.
- SVGD uses the paper/reference-code configuration: RBF kernel, adaptive median
  bandwidth, AdaGrad and master step size `0.1`.
- Langevin uses step size `0.1`, selected by its separate five-seed experiment.
- Every paired run starts from exactly the same particles.
- Independent exact GMM samples provide the distributional reference and
  finite-sample baseline.

The first-coordinate target, weights, first-coordinate initialization, particle
count, iteration count and SVGD construction come from the paper or authors'
code. The added standard-normal coordinate, Langevin baseline, seeds and metrics
are explicit project choices.

## Metrics

- **Mean target energy:** average exact GMM energy of the particles.
- **Energy error:** absolute difference from the numerically integrated expected
  target energy, `3.417621`.
- **Mode coverage:** number of modes receiving at least 1% of the particles.
- **Mixture-weight error:** total absolute error from weights `1/3` and `2/3`.
- **Mode-centre error:** average distance between empirical and target centres.
- **Within-mode variance error:** error from unit per-coordinate variance.
- **Sliced Wasserstein:** projection-based distance from an independent exact
  target sample; lower values indicate a better global approximation.
- **Runtime:** sampler wall-clock time on the same machine.

## Results

All values are mean ± sample standard deviation across five seeds.

| Method | Mean energy | Energy error | Modes | Weight error |
|---|---:|---:|---:|---:|
| Exact target sample | `3.457 ± 0.095` | `0.084 ± 0.047` | `2.0` | `0.077 ± 0.039` |
| Langevin | `3.379 ± 0.047` | `0.040 ± 0.045` | `2.0` | `0.093 ± 0.071` |
| SVGD | `3.354 ± 0.031` | `0.063 ± 0.031` | `2.0` | `0.019 ± 0.011` |

| Method | Sliced-W2 | Centre error | Variance error | Runtime |
|---|---:|---:|---:|---:|
| Exact target sample | `0.469 ± 0.085` | `0.180 ± 0.056` | `0.153 ± 0.039` | — |
| Langevin | `0.392 ± 0.080` | `0.186 ± 0.099` | `0.096 ± 0.042` | `0.059 ± 0.009 s` |
| SVGD | `0.312 ± 0.076` | `0.046 ± 0.017` | `0.103 ± 0.045` | `0.328 ± 0.047 s` |

The paired outcomes are:

- SVGD has lower mixture-weight error and sliced-Wasserstein distance in all
  five seeds.
- SVGD has lower mode-centre error in four of five seeds.
- Langevin has lower energy error in four of five seeds and lower within-mode
  variance error in three of five.
- Both methods cover both modes in every run.
- Langevin is faster in all five runs because its particles update independently,
  whereas the SVGD kernel requires pairwise interactions.

## Interpretation

The paper-inspired initialization begins far to the left of both target modes.
SVGD successfully transports the interacting particle set across this gap and
recovers the unequal target masses especially accurately. Its mean weight error
is about five times smaller than Langevin's, and its lower sliced-Wasserstein
distance indicates the better overall approximation in this experiment.

Mean energy alone does not determine distributional quality: particles can have
low energy while representing the wrong proportions or being too concentrated.
This explains why Langevin can obtain a smaller mean-energy error while SVGD is
better on the global distributional metrics.

The defensible conclusion is that SVGD performs better on mode proportions,
mode centres and overall distributional distance for this paper-inspired 2D
target, while Langevin is much faster and slightly better on some energy and
variance diagnostics. This is evidence from a small analytic extension, not a
universal ranking and not a result reported by Liu and Wang.

These sampler results also do not establish which method trains a better neural
EBM. The main 1D experiment addresses that separate question by keeping the
neural model, data, optimizer, and training budget fixed while changing only
the negative-particle sampler.
