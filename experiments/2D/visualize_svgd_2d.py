"""Visualize SVGD on the two-dimensional Gaussian mixture."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gmm_2d import (
    DIMENSION,
    MEANS,
    SEED,
    mixture_density,
    target_energy,
    target_score,
)
from svgd_2d import N_PARTICLES, N_STEPS, STEP_SIZE, svgd_dynamics


def compute_mode_fractions(particles: np.ndarray) -> np.ndarray:
    """Assign particles to their nearest mode and return mode proportions."""
    differences = particles[:, np.newaxis, :] - MEANS[np.newaxis, :, :]
    squared_distances = np.sum(differences**2, axis=2)
    nearest_modes = np.argmin(squared_distances, axis=1)
    mode_counts = np.bincount(nearest_modes, minlength=len(MEANS))
    return mode_counts / len(particles)


def main() -> None:
    # Use the same broad initialization as the Langevin experiment.
    initial_rng = np.random.default_rng(SEED)
    initial_particles = initial_rng.normal(
        loc=0.0,
        scale=3.0,
        size=(N_PARTICLES, DIMENSION),
    )

    final_particles = svgd_dynamics(
        initial_particles=initial_particles,
        n_steps=N_STEPS,
        step_size=STEP_SIZE,
        score_function=target_score,
    )

    # Evaluate the known target density on a regular grid.
    x_values = np.linspace(-4.0, 4.0, 150)
    y_values = np.linspace(-4.0, 4.0, 150)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    grid_points = np.column_stack([x_grid.ravel(), y_grid.ravel()])
    density_grid = mixture_density(grid_points).reshape(x_grid.shape)

    initial_mode_fractions = compute_mode_fractions(initial_particles)
    final_mode_fractions = compute_mode_fractions(final_particles)
    initial_mean_energy = np.mean(target_energy(initial_particles))
    final_mean_energy = np.mean(target_energy(final_particles))

    print("Initial mode fractions:")
    print(initial_mode_fractions)
    print("\nFinal mode fractions:")
    print(final_mode_fractions)
    print(f"\nInitial mean target energy: {initial_mean_energy:.3f}")
    print(f"Final mean target energy:   {final_mean_energy:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for axis, particles, title in (
        (axes[0], initial_particles, "Before SVGD"),
        (axes[1], final_particles, "After SVGD"),
    ):
        axis.contourf(
            x_grid, y_grid, density_grid, levels=40, cmap="viridis"
        )
        axis.scatter(
            particles[:, 0],
            particles[:, 1],
            s=8,
            color="red",
            alpha=0.4,
        )
        axis.scatter(
            MEANS[:, 0],
            MEANS[:, 1],
            marker="x",
            s=80,
            linewidths=2,
            color="white",
            label="Target modes",
        )
        axis.set_title(title)
        axis.set_xlabel("x₁")
        axis.set_ylabel("x₂")
        axis.set_xlim(-4.0, 4.0)
        axis.set_ylim(-4.0, 4.0)
        axis.set_aspect("equal")
        axis.legend()

    fig.suptitle(f"SVGD: step size = {STEP_SIZE}, steps = {N_STEPS}")
    fig.tight_layout()

    output_directory = Path(__file__).resolve().parent / "results"
    output_directory.mkdir(parents=True, exist_ok=True)
    figure_path = output_directory / "svgd_gmm_2d.png"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")

    print(f"\nFigure saved to: {figure_path}")
    plt.show()


if __name__ == "__main__":
    main()
