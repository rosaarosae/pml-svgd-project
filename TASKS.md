# Project Checklist

The main project is one-dimensional. SVGD is the primary negative sampler,
Langevin is the required baseline, and the existing 2D experiment is optional.
A task is checked only after its output has been executed and understood.

## 1. Theory and published reproduction

- [x] Explain energy, density, score, positive samples, and negative samples.
- [x] Explain the attraction and repulsion terms in SVGD.
- [x] Implement the exact unequal 1D Gaussian mixture from Liu and Wang (2016).
- [x] Verify `score(x) = -d energy(x) / dx` numerically.
- [x] Reproduce the particle transport shown in Figure 1.
- [x] Reproduce the expectation study shown in Figure 2.
- [x] Record the paper settings, implementation choices, results, and limits.

## 2. Supporting sampler validation

- [x] Implement Langevin dynamics on the same known 1D target.
- [x] Check Langevin and SVGD step-size sensitivity.
- [x] Compare both samplers from shared initial particles.
- [x] Keep these checks clearly labelled as supplementary, not paper results.

## 3. Main 1D neural EBM experiment

- [x] Implement a smooth neural energy with a confining quadratic term.
- [x] Compute the learned score with PyTorch automatic differentiation.
- [x] Implement and check the PyTorch RBF kernel and SVGD direction.
- [x] Implement and check repeated fixed-step SVGD particle updates.
- [x] Complete the alternating SVGD-based EBM training loop.
- [x] Maintain persistent SVGD particles across model updates.
- [x] Check whether resets are required; all ten final SVGD runs remain stable
      and no particle resets are used in the final protocol.
- [x] Check whether shared energy regularization is currently required; the
      final ten-seed runs remain finite without it.
- [x] Implement and validate the equivalent Langevin negative sampler in
      PyTorch.
- [x] Train and evaluate a second copy of the same EBM with Langevin negatives.

## 4. Controlled evaluation

- [x] Freeze the shared final configuration for both methods.
- [x] Run both EBM-training methods with ten random seeds.
- [x] Numerically normalize every learned density on a fixed grid.
- [x] Compare the learned densities with the exact density across seeds.
- [x] Compare learned and exact density and normalized-energy curves for both
      training methods across ten seeds.
- [x] Measure left/right mass against the exact target mass across seeds.
- [x] Measure integrated squared density error for both methods across seeds.
- [x] Measure mean, second moment, test NLL, and density error or numerical KL
      for both methods across seeds.
- [x] Compare training stability, negative-sample quality, and runtime.
- [x] Report mean and standard deviation and discuss failed runs honestly.

## 5. Optional 2D extension

- [x] Implement and validate the paper-inspired 2D analytic target.
- [x] Compare analytic-target SVGD and Langevin samplers across five seeds.
- [ ] Include the 2D results only if presentation time permits.
- [ ] Leave neural EBM training in 2D as future work unless the 1D study is
      completely finished.

## 6. Final deliverables

- [x] Produce one clean and reproducible main experiment runner.
- [x] Finalize figures, captions, environment, and run instructions.
- [ ] Integrate the implementation and results into the group Beamer slides.
- [ ] Prepare answers to likely technical questions and rehearse the talk.

## Completion criteria

The project is complete when the same 1D neural EBM has been trained with SVGD
and Langevin negative samples under matched conditions, the learned densities
have been evaluated over multiple seeds against the exact target, all choices
not taken from a paper are identified explicitly, and the results can be
reproduced from the documented environment.
