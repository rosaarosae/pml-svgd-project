# Experiments

## One-dimensional GMM

`svgd_gmm_1d.py` approximates an equally weighted mixture of two Gaussian
distributions centred at -2 and 2 using 100 SVGD particles.

Run it from the repository root with the environment activated:

```bash
python experiments/svgd_gmm_1d.py
```

The program prints basic metrics and saves the visual comparison to
`experiments/results/svgd_gmm_1d.png`.

The random seed and hyperparameters are fixed at the top of the file to make the
result reproducible.
