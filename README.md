# Stein Variational Gradient Descent

Project on particle-based approximate Bayesian inference using **Stein Variational
Gradient Descent (SVGD)**.

## Objectives

- Study the main ideas behind SVGD and its particle update.
- Implement a reusable SVGD method.
- Apply SVGD to one- and two-dimensional Gaussian mixture models.
- Use SVGD to approximate the posterior distribution over the weights of a Bayesian
  neural network.
- Analyse predictive performance and uncertainty.

## Main references

- Q. Liu and D. Wang, “Stein Variational Gradient Descent: A General Purpose
  Bayesian Inference Algorithm,” *Advances in Neural Information Processing
  Systems 29*, 2016. [Paper](https://proceedings.neurips.cc/paper/2016/hash/b3ba8f1bee1238a2f37603d90b58898d-Abstract.html)
- DartML, “Stein-Variational-Gradient-Descent.”
  [Reference implementation](https://github.com/DartML/Stein-Variational-Gradient-Descent)

Additional references and documentation are listed in [SOURCES.md](SOURCES.md).

## Structure

- `experiments/`: implementations and results.
- `presentation/`: LaTeX Beamer presentation.
- `SOURCES.md`: references and technical documentation.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```
