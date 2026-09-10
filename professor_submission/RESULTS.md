# Final one-dimensional EBM results

## Protocol

The same neural energy architecture, batches of 200, 1,000 epochs, 500
persistent particles, and 20 sampler updates per epoch are used for both
methods. Adam starts at learning rate `1e-3` and follows the same cosine
schedule down to `1e-4`. Negative particles start from the same neutral
distribution `N(0, 3^2)`, rather than from the data distribution.

Multiple development sweeps were used to identify stable step sizes, which
were frozen before the final large comparison:

- SVGD: `0.02`;
- Langevin: `0.05`.

The step sizes differ because the two update rules have different numerical
scales. They are empirical project choices rather than universal optima. The
final comparison uses ten paired seeds, and all 20 runs were stable.

## Aggregate results

Values are mean plus or minus sample standard deviation over ten seeds.

| Metric | SVGD | Langevin | Preferred |
|---|---:|---:|---|
| Integrated squared density error | `0.000568 +/- 0.000280` | `0.001349 +/- 0.000582` | SVGD |
| Test negative log-likelihood | `2.005433 +/- 0.006213` | `2.012102 +/- 0.008119` | SVGD |
| Training time (seconds) | `69.789967 +/- 5.306098` | `8.157168 +/- 0.911722` | Langevin |
| Learned left mass | `0.338640 +/- 0.011421` | `0.338482 +/- 0.016518` | Exact: `0.340916` |
| Learned mean | `0.691529 +/- 0.058051` | `0.696487 +/- 0.081874` | Exact: `0.666667` |
| Learned second moment | `4.741096 +/- 0.034284` | `5.137377 +/- 0.025288` | Exact: `5.0` |
| Stable runs | `10/10` | `10/10` | Tie |

SVGD has lower density error and test NLL on all ten paired seeds. Its mean
density error is about 58% lower and both accuracy metrics vary less across
seeds. Langevin is about 8.6 times faster. Both methods recover the unequal
left/right probability mass and mean accurately; Langevin is closer on the
second moment, while SVGD is slightly closer on the mean.

The density figure shows that both learned models recover the two modes and
their unequal heights. The normalized-energy curves agree closely with the
exact target in the high-density region. Their larger differences occur in the
far tails, where the probability density is already close to zero.

## Defensible conclusion

Under this matched one-dimensional protocol, SVGD provides the better average
density fit, while Langevin provides the lower computational cost. The result
supports a quality-versus-speed trade-off; it does not establish that either
sampler is universally superior.

## Limitations

- The target and model are one-dimensional.
- The learned density is normalized numerically on `[-8, 8]`.
- Step sizes are selected from a finite candidate grid.
- Equal sampler-step counts do not mean equal computational cost: the SVGD
  kernel couples all particles and is quadratic in their number.
- Results are empirical over ten seeds and should not be presented as a general
  theorem about SVGD or Langevin dynamics.
