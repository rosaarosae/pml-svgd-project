"""Supplementary Langevin baseline on the paper's one-dimensional target.

Langevin is not part of Figure 1 in Liu and Wang (2016). This script is kept as
an explicitly labelled baseline, but it now uses exactly the same target and
initial distribution as the paper reproduction.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gmm_1d import (
    INITIAL_MEAN,
    INITIAL_STANDARD_DEVIATION,
    SEED,
    sample_initial_particles,
    target_density,
    target_score,
)

NPARTICLES = 100
NSTEPS = 500
STEPSIZE = 0.03

# Backwards-compatible names used by the existing comparison scripts.
density = target_density
score = target_score


# Apply repeated drift and diffusion updates to the complete particle set.
def langevin(
    particles: np.ndarray,
    rng,
    stepsize: float = STEPSIZE,
) -> np.ndarray:
    for _ in range(NSTEPS):
        # The drift follows the score towards regions of higher probability.
        #this is the drift term, and it moves the particles in the direction of the score function
        particles += stepsize * score(particles)
        # The diffusion noise lets the particles explore the target distribution.
        #this is the diffusion term, and it adds noise to the particles
        particles += np.sqrt(2.0 * stepsize) * rng.normal(size=particles.shape)
    return particles

def main() -> None:
    # Use q0(x) = N(-10, 1), the difficult initialization from the paper.
    rng = np.random.default_rng(SEED)
    particles = sample_initial_particles(NPARTICLES, rng)
    # Preserve the starting positions for the before-and-after comparison.
    initial_particles = particles.copy()

    # Move the particles using Langevin dynamics.
    particles = langevin(particles, rng)

    # Plot the initial and final particles against the same target density.
    xgrid = np.linspace(-15.0, 8.0, 1000)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True, sharey=True)
    for axis, values, title in (
        (axes[0], initial_particles, "Before Langevin"),
        (axes[1], particles, "After Langevin"),
    ):
        axis.hist(values, bins=22, density=True, alpha=0.55, color="#4C78A8")
        axis.plot(xgrid, density(xgrid), color="#E45756", linewidth=2.2)
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Density")
    fig.suptitle("Supplementary Langevin baseline on the paper's 1D target")
    fig.tight_layout()

    # Save the figure so the result remains available after the window closes.
    outputDir = Path(__file__).resolve().parent / "results"
    outputDir.mkdir(parents=True, exist_ok=True)
    figurePath = outputDir / "langevin_gmm_1d.png"
    fig.savefig(figurePath, dpi=180, bbox_inches="tight")

    # Report the same diagnostics used by the SVGD validation.
    initialLogDensity = np.log(density(initial_particles) + 1e-12).mean()
    finalLogDensity = np.log(density(particles) + 1e-12).mean()
    leftFraction = np.mean(particles < 0.0)

    print(f"Initial mean log target density: {initialLogDensity:.3f}")
    print(f"Final mean log target density:   {finalLogDensity:.3f}")
    print(f"Final fraction in left mode:     {leftFraction:.2f}")
    print(f"Final fraction in right mode:    {1.0 - leftFraction:.2f}")
    print(
        "Initial distribution: "
        f"N({INITIAL_MEAN:.0f}, {INITIAL_STANDARD_DEVIATION**2:.0f})"
    )
    print(f"Figure saved to: {figurePath}")

    plt.show()


if __name__ == "__main__":
    main()
