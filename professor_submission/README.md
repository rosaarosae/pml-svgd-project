# Professor submission: SVGD for neural EBM training

This folder is the self-contained project snapshot intended for submission.
It contains the final one-dimensional experiment, the minimal paper
reproduction needed to validate SVGD, the fixed experimental protocol, and the
final numerical results. Exploratory scripts and optional 2D work are kept
outside this folder.

## Research question

When the same one-dimensional neural energy-based model is trained under
matched conditions, how does using SVGD rather than Langevin dynamics for the
negative particles affect density accuracy, test negative log-likelihood,
stability, and runtime?

The target distribution is

```text
p_data(x) = (1/3) N(-2, 1) + (2/3) N(2, 1).
```

Its exact mean is `2/3`, its exact second moment is `5`, and its two unequal
modes make failures in mode coverage visible.

## Contents

- `code/gmm_1d.py`: exact target density, score, samples, and moments.
- `code/energy_model.py`: neural energy and automatically differentiated score.
- `code/svgd_ebm_1d.py`: SVGD kernel and particle updates.
- `code/langevin_ebm_1d.py`: Langevin particle updates.
- `code/compare_ebm_svgd_langevin_1d.py`: final ten-seed comparison and figure.
- `code/svgd_gmm_1d.py` and `code/svgd_expectations_1d.py`: validation against
  the published one-dimensional SVGD toy experiment.
- `RESULTS.md`: concise interpretation of the final results and limitations.
- `PAPER_REPRODUCTION_RESULTS.md`: settings and results of the published toy
  experiment reproduction.
- `code/results/`: final CSV files and presentation-ready figures.

## Final protocol and step-size choice

Both methods use the same neural architecture, data batches, 1,000 epochs, 500
persistent particles, 20 sampler updates per epoch, neutral particle
initialization, optimizer, and cosine learning-rate schedule. Only the sampler
and its algorithm-specific step size differ.

Several development experiments explored learning rates, particle counts,
sampler-step counts, and candidate step sizes. The final stable choices were
frozen at `0.02` for SVGD and `0.05` for Langevin. Equal numerical step sizes
would not be a meaningful fairness requirement: the SVGD update is a
kernel-averaged deterministic direction, whereas Langevin also contains noise
with scale proportional to the square root of its step size. These values are
empirical project choices, not settings claimed to be universally optimal.

## Reproduce the final experiment

Create an environment from this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the final comparison:

```bash
cd code
python compare_ebm_svgd_langevin_1d.py
```

This trains both methods over ten paired seeds, writes the metrics and curve
data, saves both comparison figures, and displays them with `plt.show()`. The
runtime depends on the computer; the complete run may take around 20 minutes.

To recreate only the figures from the saved CSV and curve archive, without
training again:

```bash
cd code
python compare_ebm_svgd_langevin_1d.py --plot-only
```

## What is not part of the submission

Single-seed development runs, broad step-size checks, direct known-target
sampler comparisons, and the optional 2D extension remain in the main
repository as development tests. They are documented separately in
`development_tests/README.md` and are not presented as final evidence.

## Main references

- Q. Liu and D. Wang, “Stein Variational Gradient Descent: A General Purpose
  Bayesian Inference Algorithm,” NeurIPS 2016.
  <https://arxiv.org/abs/1608.04471>
- Q. Liu and D. Wang, “Learning Deep Energy Models: Contrastive Divergence vs.
  Amortized MLE,” 2017. <https://arxiv.org/abs/1707.00797>
- Y. Song and D. P. Kingma, “How to Train Your Energy-Based Models,” 2021.
  <https://arxiv.org/abs/2101.03288>
