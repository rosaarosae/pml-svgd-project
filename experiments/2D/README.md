# Two-dimensional experiments

This directory contains the main two-dimensional experiments, beginning with
the multimodal Gaussian mixture target. Generated figures belong in `results/`.

`gmm_2d.py` currently defines four equally weighted isotropic Gaussian
components and verifies their individual densities at known points.

Run the current validation from the repository root:

```bash
python experiments/2D/gmm_2d.py
```

The next step is to combine the weighted component densities into the complete
mixture density.
