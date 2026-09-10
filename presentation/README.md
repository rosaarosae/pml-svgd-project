# Presentation

This directory is reserved for Rosa's implementation and results slides within
the group Beamer presentation on SVGD.

The presentation narrative should follow the final project scope:

1. Mai presents the theoretical derivation of SVGD.
2. Rosa briefly validates the implementation with the published 1D Gaussian
   mixture experiment.
3. Rosa explains how an EBM defines `p_theta(x)` proportional to
   `exp(-E_theta(x))` and why its score is `-grad_x E_theta(x)`.
4. Rosa presents the main experiment: train the same 1D neural EBM with SVGD
   and Langevin negative samples under matched conditions.
5. The results compare learned density, mode proportions, moments, stability,
   and runtime over multiple seeds.
6. The 2D analytic sampler comparison is optional and should appear only if the
   ten-minute time limit allows it; 2D neural EBM training is future work.

No slide should claim that Langevin or the 2D extension appears in Figures 1 or
2 of Liu and Wang (2016). The neural architecture and training protocol must be
identified as project choices. The completed ten-seed experiment supports a
quality-versus-speed conclusion: SVGD has lower density error and test NLL on
every paired seed, while Langevin is approximately 8.6 times faster.
