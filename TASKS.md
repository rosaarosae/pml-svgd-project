# Project Checklist

This checklist tracks the complete path from the initial literature review to
the final presentation. A task is checked only after its output has been
verified and understood.

## Completed setup

- [x] Define a focused research question for the project.
- [x] Review the core SVGD, EBM, score-matching, and course references.
- [x] Create and organise the local and GitHub repositories.
- [x] Implement an initial one-dimensional SVGD validation.
- [x] Verify numerically that the EBM score is the negative energy gradient.

## Foundations and sampler validation

- [ ] **Current task:** Understand EBM, score, Langevin dynamics, SVGD, and
      contrastive negative samples. *(Estimated effort: 1 hour)*
- [ ] Implement Langevin dynamics on the one-dimensional Gaussian mixture and
      analyse drift, noise, and step size. *(2 hours)*
- [ ] Compare Langevin and SVGD on the same known one-dimensional target.
      *(2–3 hours)*

## Main two-dimensional experiment

- [ ] Implement the multimodal two-dimensional Gaussian mixture. *(2 hours)*
- [ ] Validate its density, energy, and analytic score. *(1–2 hours)*
- [ ] Implement reusable Langevin and SVGD samplers for two-dimensional
      targets. *(4–6 hours)*
- [ ] Validate both samplers against the known target distribution. *(3 hours)*

## Energy-based model

- [ ] Implement the neural energy function and contrastive training objective.
      *(3–4 hours)*
- [ ] Train the EBM using Langevin negative samples. *(4–6 hours)*
- [ ] Train the same EBM using SVGD negative samples. *(4–6 hours)*
- [ ] Add persistent particles, controlled resets, and shared regularisation.
      *(2–3 hours)*

## Evaluation

- [ ] Run both methods with at least five random seeds. *(3–5 hours)*
- [ ] Measure mode coverage and mixture-weight error. *(2 hours)*
- [ ] Estimate test negative log-likelihood using numerical normalisation in
      two dimensions. *(2–3 hours)*
- [ ] Measure sample quality, convergence, and runtime. *(2–3 hours)*
- [ ] Study the effect of the number of particles. *(3–4 hours)*
- [ ] Summarise results with uncertainty and discuss limitations. *(3 hours)*

## Final deliverables

- [ ] Produce one clean and reproducible Python notebook. *(3–4 hours)*
- [ ] Finalise figures, captions, environment, and reproduction instructions.
      *(2–3 hours)*
- [ ] Prepare the ten-minute LaTeX Beamer presentation. *(4–6 hours)*
- [ ] Prepare answers to likely technical questions and rehearse the
      presentation. *(2–3 hours)*

## Completion criteria

The project is complete when the notebook reproduces the main results from a
clean environment, the comparison changes only the negative sampler, reported
results include uncertainty across seeds, and every team member can explain the
method and conclusions.
