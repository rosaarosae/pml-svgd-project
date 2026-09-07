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
- [ ] **Current task:** Implement a numerical check that the EBM score is the
      negative input gradient of its energy.
- [ ] Reproduce and explain the Langevin step-size sensitivity check.
- [ ] Study and explain the existing one-dimensional SVGD implementation,
      including its attraction and repulsion terms.
- [ ] Compare Langevin and SVGD on the same known one-dimensional target.
  - [ ] Run both methods from the same initial particles.
  - [ ] Create a three-panel initial/Langevin/SVGD figure.
  - [ ] Compare mode balance, mean log density, and runtime.
  - [ ] Repeat the comparison across multiple random seeds.

## Main two-dimensional experiment

- [ ] Implement the multimodal two-dimensional Gaussian mixture.
- [ ] Validate its density, energy, and analytic score.
- [ ] Implement reusable Langevin and SVGD samplers for two-dimensional
      targets.
- [ ] Validate both samplers against the known target distribution.

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
