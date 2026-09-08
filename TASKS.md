# Project Checklist

This checklist tracks the complete path from the initial literature review to
the final presentation. A task is checked only after its output has been
verified and understood.

## Completed setup

- [x] Define a focused research question for the project.
- [x] Review the core SVGD, EBM, score-matching, and course references.
- [x] Create and organise the local and GitHub repositories.

## Foundations and sampler validation

- [x] Understand EBM, score, Langevin dynamics, SVGD, and contrastive negative
      samples.
- [x] Implement Langevin dynamics on the one-dimensional Gaussian mixture and
      explain its drift and diffusion terms.
- [x] Implement a numerical check that the EBM score is the negative input
      gradient of its energy.
- [x] Reproduce and explain the Langevin step-size
      sensitivity check.
- [x] Study and explain the existing one-dimensional SVGD implementation,
      including its attraction and repulsion terms.
- [x] Compare Langevin and SVGD on the same known one-dimensional target.
  - [x] Run both methods from the same initial particles.
  - [x] Compare mode balance and mean log density.

## Main two-dimensional experiment

- [x] Implement the multimodal two-dimensional Gaussian
      mixture.
  - [x] Define its means, equal weights, and shared isotropic covariance.
  - [x] Implement and check the individual component densities.
  - [x] Combine the components into the complete mixture density.
  - [x] Generate target samples and visualise the density in two dimensions.
- [x] Validate its density, energy, and analytic score.
- [x] Implement reusable Langevin and SVGD samplers for two-dimensional
      targets.
- [ ] **Current task:** Validate both samplers against the known target
      distribution with a direct, shared-condition comparison.

## Energy-based model

- [ ] Implement the neural energy function and contrastive training objective.
- [ ] Train the EBM using Langevin negative samples.
- [ ] Train the same EBM using SVGD negative samples.
- [ ] Add persistent particles, controlled resets, and shared regularisation.

## Evaluation

- [ ] Run both methods with at least five random seeds.
- [ ] Measure mode coverage and mixture-weight error.
- [ ] Estimate test negative log-likelihood using numerical normalisation in
      two dimensions.
- [ ] Measure sample quality, convergence, and runtime.
- [ ] Study the effect of the number of particles.
- [ ] Summarise results with uncertainty and discuss limitations.

## Final deliverables

- [ ] Produce one clean and reproducible Python notebook.
- [ ] Finalise figures, captions, environment, and reproduction instructions.
- [ ] Prepare the ten-minute LaTeX Beamer presentation.
- [ ] Prepare answers to likely technical questions and rehearse the
      presentation.

## Completion criteria

The project is complete when the notebook reproduces the main results from a
clean environment, the comparison changes only the negative sampler, reported
results include uncertainty across seeds, and every team member can explain the
method and conclusions.
