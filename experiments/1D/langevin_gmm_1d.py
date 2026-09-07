"""Langevin dynamics in a one-dimensional Gaussian mixture."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SEED = 7
NPARTICLES = 100
NSTEPS = 1200
STEPSIZE = 0.03
MEANS = np.array([-2.0, 2.0])
STD = 0.55


# Evaluate each Gaussian component separately before mixing them.
def components(x: np.ndarray) -> np.ndarray:
    z = (x[..., None] - MEANS) / STD
    return np.exp(-0.5 * z**2) / (np.sqrt(2.0 * np.pi) * STD)


# Both components have the same mixture weight of 0.5.
def density(x: np.ndarray) -> np.ndarray:
    return 0.5 * components(x).sum(axis=-1)


# The score points towards locations with higher target probability.
def score(x: np.ndarray) -> np.ndarray:
    densities = components(x)
    # Responsibilities measure how much each Gaussian explains every point.
    responsibilities = densities / densities.sum(axis=-1, keepdims=True)
    # Combine the component scores according to those responsibilities.
    componentScores = (MEANS - x[..., None]) / STD**2
    return (responsibilities * componentScores).sum(axis=-1)


# Apply repeated drift and diffusion updates to the complete particle set.
def langevin(
    particles: np.ndarray,
    rng,
    stepsize: float = STEPSIZE,
) -> np.ndarray:
    for step in range(NSTEPS):
        # The drift follows the score towards regions of higher probability.
        #this is the drift term, and it moves the particles in the direction of the score function
        particles += stepsize * score(particles)
        # The diffusion noise lets the particles explore the target distribution.
        #this is the diffusion term, and it adds noise to the particles
        particles += np.sqrt(2.0 * stepsize) * rng.normal(size=particles.shape)
    return particles

def main() -> None:
    # Start from a broad Gaussian so particles must discover both target modes.
    rng = np.random.default_rng(SEED)
    particles = rng.normal(loc=0.0, scale=3.0, size=NPARTICLES)
    # Preserve the starting positions for the before-and-after comparison.
    initial_particles = particles.copy()

    # Move the particles using Langevin dynamics.
    particles = langevin(particles, rng)

    # Plot the initial and final particles against the same target density.
    xgrid = np.linspace(-5.0, 5.0, 1000)
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
    fig.suptitle("Langevin approximation of a two-component 1D Gaussian mixture")
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
    print(f"Figure saved to: {figurePath}")

    plt.show()


if __name__ == "__main__":
    main()
