"""Small, reproducible SVGD test on a one-dimensional Gaussian mixture."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SEED = 7
N_PARTICLES = 100
N_STEPS = 1_200
STEP_SIZE = 0.03
MEANS = np.array([-2.0, 2.0])
STD = 0.55


def component_densities(x: np.ndarray) -> np.ndarray:
    """Return the two component densities at each x."""
    z = (x[..., None] - MEANS) / STD
    return np.exp(-0.5 * z**2) / (np.sqrt(2.0 * np.pi) * STD)


def target_density(x: np.ndarray) -> np.ndarray:
    return 0.5 * component_densities(x).sum(axis=-1)


def score(x: np.ndarray) -> np.ndarray:
    """Compute d/dx log p(x) for the Gaussian mixture."""
    densities = component_densities(x)
    responsibilities = densities / densities.sum(axis=-1, keepdims=True)
    component_scores = (MEANS - x[..., None]) / STD**2
    return (responsibilities * component_scores).sum(axis=-1)


def svgd_direction(particles: np.ndarray) -> np.ndarray:
    """Compute the SVGD direction using an RBF kernel."""
    differences = particles[:, None] - particles[None, :]
    squared_distances = differences**2
    nonzero = squared_distances[squared_distances > 0]
    median_squared_distance = np.median(nonzero) if nonzero.size else 1.0
    bandwidth = max(median_squared_distance / np.log(N_PARTICLES + 1.0), 1e-3)

    kernel = np.exp(-squared_distances / bandwidth)
    attraction = kernel.T @ score(particles)
    repulsion = (-2.0 / bandwidth * differences * kernel).sum(axis=0)
    return (attraction + repulsion) / N_PARTICLES


def main() -> None:
    rng = np.random.default_rng(SEED)
    particles = rng.normal(loc=0.0, scale=3.0, size=N_PARTICLES)
    initial_particles = particles.copy()

    # AdaGrad scaling makes the small experiment less sensitive to step size.
    accumulated_squared_gradient = np.zeros_like(particles)
    for _ in range(N_STEPS):
        direction = svgd_direction(particles)
        accumulated_squared_gradient += direction**2
        particles += STEP_SIZE * direction / (
            1e-6 + np.sqrt(accumulated_squared_gradient)
        )

    x_grid = np.linspace(-5.0, 5.0, 1_000)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True, sharey=True)
    for axis, values, title in (
        (axes[0], initial_particles, "Before SVGD"),
        (axes[1], particles, "After SVGD"),
    ):
        axis.hist(values, bins=22, density=True, alpha=0.55, color="#4C78A8")
        axis.plot(x_grid, target_density(x_grid), color="#E45756", linewidth=2.2)
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Density")
    fig.suptitle("SVGD approximation of a two-component 1D Gaussian mixture")
    fig.tight_layout()

    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "svgd_gmm_1d.png"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")

    initial_mean_log_density = np.log(target_density(initial_particles) + 1e-12).mean()
    final_mean_log_density = np.log(target_density(particles) + 1e-12).mean()
    left_fraction = np.mean(particles < 0.0)

    print(f"Initial mean log target density: {initial_mean_log_density:.3f}")
    print(f"Final mean log target density:   {final_mean_log_density:.3f}")
    print(f"Final fraction in left mode:     {left_fraction:.2f}")
    print(f"Final fraction in right mode:    {1.0 - left_fraction:.2f}")
    print(f"Figure saved to: {figure_path}")


if __name__ == "__main__":
    main()
