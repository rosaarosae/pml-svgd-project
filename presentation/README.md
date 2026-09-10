# Suggested presentation order

The clearest way to present this project is as one question and one controlled
experiment. Six short parts are enough.

## 1. The question

“When training the same neural energy-based model, what changes if its negative
samples come from SVGD instead of Langevin dynamics?”

## 2. Why use a one-dimensional Gaussian mixture?

Show the target distribution with modes near `-2` and `2`. Explain that its
exact density is known, so the learned model can be checked directly instead
of judged only from samples.

## 3. How the EBM learns

Explain that likely data should receive low energy. Positive samples come from
the target. Persistent negative particles come from the current model. The
network learns from the difference between their average energies.

## 4. What SVGD and Langevin change

SVGD moves particles together using attraction and repulsion. Langevin moves
them using the score plus random noise. Everything else in training is held
fixed. The step sizes differ because the two update equations have different
scales; both were selected in development tests and frozen before comparison.

## 5. Show the two final figures

First show the density and energy curves. Both methods recover the two modes,
but SVGD follows the exact density more closely across seeds.

Then show the paired metric plot. State the main numbers:

- SVGD density error: `0.000568 ± 0.000280`.
- Langevin density error: `0.001349 ± 0.000582`.
- SVGD time: `69.79 ± 5.31` seconds.
- Langevin time: `8.16 ± 0.91` seconds.
- All 20 runs were stable.

## 6. Finish with the trade-off

The honest conclusion is: SVGD was more accurate in this experiment, but
Langevin was about 8.6 times faster. The study is controlled and reproducible,
but it covers one target and one model, so it does not establish a universal
winner.

The two figures to use are:

- `experiments/1D/results/ebm_svgd_langevin_curves_1d.png`
- `experiments/1D/results/ebm_svgd_langevin_comparison_1d.png`
