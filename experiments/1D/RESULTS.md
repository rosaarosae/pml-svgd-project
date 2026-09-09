# Results of the 1D paper reproduction

## Scope

The primary result is a reproduction of Figures 1 and 2 in Section 5 of Liu and
Wang (2016). Langevin and step-size experiments remain available only as clearly
labelled supplementary analyses; they are not part of the published toy
experiment.

## Configuration traced to the paper

| Quantity | Value | Source |
|---|---:|---|
| Target distribution | `(1/3) N(-2,1) + (2/3) N(2,1)` | Section 5 |
| Initial distribution | `N(-10,1)` | Section 5 |
| Number of particles | `100` | Figure 1 |
| Displayed iterations | `0, 50, 75, 100, 150, 500` | Figure 1 |
| Kernel | `exp(-||x-x'||^2 / h)` | Section 3.2 |
| Adaptive bandwidth | median heuristic | Section 3.2 |
| Optimizer | AdaGrad | Section 5 and author code |

The paper writes the median heuristic with `log(n)`, while the released code
uses `log(n + 1)`; the reproduction follows the released implementation. The
paper does not publish the plotting bandwidth of its kernel-density estimate,
so the visualization uses a fixed Silverman bandwidth computed from the initial
particles. This choice affects only the green plotted curve, not SVGD itself.

For Figure 2, the paper specifies 20 random cosine functions with
`omega ~ N(0,1)` and `b ~ Uniform(0,2*pi)`. The sample-size grid
`10, 20, 50, 100, 250`, the 20 repeated trials for the other two functions and
the deterministic seeds are explicit reproduction choices because the exact
random seeds and full numerical protocol are not reported.

## Figure 1: transport of the particles

With seed 7, 100 particles and 500 iterations:

| Diagnostic | SVGD estimate | Exact target value |
|---|---:|---:|
| Mean `E[x]` | `0.6397` | `0.6667` |
| Second moment `E[x^2]` | `4.9514` | `5.0000` |
| Mass left of zero | `0.330` | `0.333` |
| Mass right of zero | `0.670` | `0.667` |

The snapshots reproduce the behavior shown in the paper: the initially distant
cloud moves towards the target, separates into two groups and ends with the
correct unequal mode proportions. The repulsive kernel term is essential here:
without it, the target-score attraction alone could collapse particles instead
of maintaining a useful approximation of the full distribution.

## Figure 2: expectation estimates

At the largest tested sample size (`n = 250`), the mean squared errors are:

| Function | Monte Carlo MSE | SVGD MSE | SVGD / MC |
|---|---:|---:|---:|
| `x` | `1.2899e-2` | `3.5284e-4` | `0.027` |
| `x^2` | `3.6063e-2` | `3.0824e-3` | `0.085` |
| `cos(omega*x+b)` | `1.5505e-3` | `1.4482e-4` | `0.093` |

Thus, for this run, SVGD has approximately 37, 12 and 11 times lower MSE than
independent Monte Carlo sampling for the three functions. More importantly than
any single ratio, the full curves keep the SVGD error below the Monte Carlo error
throughout the tested range, matching the qualitative conclusion of Figure 2.

## Interpretation and limitations

This experiment demonstrates particle transport and expectation approximation;
it does not prove that SVGD always outperforms Monte Carlo or Langevin. The target
is one-dimensional, analytic and deliberately chosen for visualization. The
exact curves can vary with initialization, optimizer settings and finite-trial
randomness. Accordingly, the defensible conclusion is that the implementation
reproduces the paper's reported qualitative behavior under its published toy
setup, not that it establishes a universal ranking of inference algorithms.

The energy used here is exactly `E(x) = -log p(x)`. Its negative derivative is
the score required by SVGD, so the energy-score check connects the mathematical
target to the implementation. It is not a separately trained energy-based
model in this experiment.
