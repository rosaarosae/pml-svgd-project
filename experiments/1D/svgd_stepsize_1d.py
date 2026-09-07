"""Compare different SVGD step sizes in 1D."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from svgd_gmm_1d import (
    SEED,
    N_PARTICLES,
    N_STEPS,
    target_density,
    svgd_direction,
)


STEP_SIZES = [0.03, 0.05, 0.07, 0.1, 0.15, 0.3]


def run_svgd(initial_particles: np.ndarray, step_size: float) -> np.ndarray:
    particles = initial_particles.copy()

    accumulated_squared_gradient = np.zeros_like(particles)

    for _ in range(N_STEPS):
        direction = svgd_direction(particles)

        accumulated_squared_gradient += direction**2

        particles += step_size * direction / (
            1e-6 + np.sqrt(accumulated_squared_gradient)
        )

    return particles


def main() -> None:
    rng = np.random.default_rng(SEED)

    initial_particles = rng.normal(
        loc=0.0,
        scale=3.0,
        size=N_PARTICLES,
    )

    x = np.linspace(-5, 5, 1000)

    fig, axes = plt.subplots(
        1,
        len(STEP_SIZES),
        figsize=(18, 4),
        sharex=True,
        sharey=True,
    )

    for axis, step_size in zip(axes, STEP_SIZES):
        particles = run_svgd(
            initial_particles,
            step_size,
        )

        axis.plot(
            x,
            target_density(x),
            color="red",
            label="Target density",
        )

        axis.hist(
            particles,
            bins=30,
            density=True,
            alpha=0.5,
            label="SVGD particles",
        )

        axis.set_title(f"Step size = {step_size}")
        axis.tick_params(labelleft=True)

        mean_log_density = np.log(
            target_density(particles) + 1e-12
        ).mean()

        left_fraction = np.mean(particles < 0.0)

        print(
            f"Step size {step_size}: "
            f"log density = {mean_log_density:.3f}, "
            f"left fraction = {left_fraction:.2f}"
        )

    axes[0].legend()

    fig.suptitle("SVGD step size comparison in 1D")
    fig.tight_layout()

    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    figure_path = output_dir / "svgd_stepsize_1d.png"

    fig.savefig(
        figure_path,
        dpi=180,
        bbox_inches="tight",
    )

    print(f"Figure saved to: {figure_path}")

    plt.show()


if __name__ == "__main__":
    main()