"""Visualize the two-dimensional Gaussian mixture model."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gmm_2d import (
    SEED,
    TARGET_X_LIMITS,
    Y_LIMITS,
    mixture_density,
    sample_target,
)


def main() -> None:
    # Evaluate the paper-inspired 2D target on a regular grid.
    x = np.linspace(*TARGET_X_LIMITS, 150)
    y = np.linspace(*Y_LIMITS, 150)
    x_grid, y_grid = np.meshgrid(x, y)

    # Convert the grid into a list of points with shape (n_points, 2).
    points = np.column_stack([
        x_grid.ravel(),
        y_grid.ravel(),
    ])

    # Compute the density of the mixture at each point.
    densities = mixture_density(points)

    # Restore the original grid shape for the contour plot.
    density_grid = densities.reshape(x_grid.shape)

    # Generate 2,000 samples from the target distribution.
    rng = np.random.default_rng(SEED)
    samples = sample_target(2_000, rng)

    # Draw the density contours and the target samples.
    fig, axis = plt.subplots(figsize=(8, 6))

    contour = axis.contourf(
        x_grid,
        y_grid,
        density_grid,
        levels=50,
        cmap="viridis",
    )

    axis.scatter(
        samples[:, 0],
        samples[:, 1],
        s=5,
        color="red",
        alpha=0.3,
        label="Target samples",
    )

    fig.colorbar(
        contour,
        ax=axis,
        label="Target density",
    )

    axis.set_title("Paper-inspired two-dimensional Gaussian mixture")
    axis.set_xlabel("x₁")
    axis.set_ylabel("x₂")
    axis.set_aspect("equal")
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
    )

    # Save the figure inside the results directory.
    output_directory = Path(__file__).resolve().parent / "results"
    output_directory.mkdir(parents=True, exist_ok=True)

    figure_path = output_directory / "gmm_2d.png"

    fig.tight_layout()
    fig.savefig(
        figure_path,
        dpi=180,
        bbox_inches="tight",
    )

    print(f"Figure saved to: {figure_path}")

    plt.show()


if __name__ == "__main__":
    main()
