"""Reproduce the paper's toy SVGD experiment on a 1D Gaussian mixture.

The target, initialization, particle count, RBF kernel and adaptive bandwidth
come from Section 5 and Equation (8) of Liu and Wang (NeurIPS 2016). The
AdaGrad-with-momentum update follows the implementation released by the
authors: https://github.com/DartML/Stein-Variational-Gradient-Descent.
"""

from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gmm_1d import (
    EXPECTED_MEAN,
    EXPECTED_SECOND_MOMENT,
    SEED,
    WEIGHTS,
    sample_initial_particles,
    target_density,
    target_score,
)


# Figure 1 of the paper uses 100 particles. Its last displayed panel is the
# 500th iteration, so those are the primary reproduction settings here.
N_PARTICLES = 100
N_STEPS = 500

# The paper states that AdaGrad is used. These values follow the defaults in
# the authors' released MATLAB implementation and reproduce its trajectory.
STEP_SIZE = 0.1
ADAGRAD_DECAY = 0.9
ADAGRAD_EPSILON = 1e-6

# These are the six iterations shown in Figure 1 of the published paper.
SNAPSHOT_STEPS = (0, 50, 75, 100, 150, 500)


def rbf_kernel(
    particles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute the paper's RBF kernel and its repulsive gradient.

    The paper uses k(x, x') = exp(-||x-x'||^2 / h) with the median heuristic
    h = med^2 / log(n). The released code uses log(n + 1), which is used here
    for implementation-level consistency and numerical safety.

    Args:
        particles: One-dimensional particle positions with shape (n,).

    Returns:
        The kernel matrix, the summed kernel-gradient term from Equation (8),
        and the adaptive bandwidth h.
    """
    particles = np.asarray(particles, dtype=float)
    if particles.ndim != 1 or len(particles) == 0:
        raise ValueError("particles must be a non-empty one-dimensional array")

    pairwise_differences = (
        particles[:, np.newaxis] - particles[np.newaxis, :]
    )
    squared_distances = pairwise_differences**2

    median_squared_distance = float(np.median(squared_distances))
    bandwidth = max(
        median_squared_distance / np.log(len(particles) + 1.0),
        1e-8,
    )

    kernel_matrix = np.exp(-squared_distances / bandwidth)

    # For particle i, this is sum_j grad_{x_j} k(x_j, x_i). It is the
    # repulsive term in Equation (8), preventing collapse to a single mode.
    kernel_gradient = (
        2.0
        * (
            particles * np.sum(kernel_matrix, axis=1)
            - kernel_matrix @ particles
        )
        / bandwidth
    )

    return kernel_matrix, kernel_gradient, bandwidth


def svgd_direction(particles: np.ndarray) -> np.ndarray:
    """Evaluate the empirical SVGD direction in Equation (8) of the paper."""
    kernel_matrix, kernel_gradient, _ = rbf_kernel(particles)

    # Attraction is the kernel-smoothed target score. Repulsion is supplied
    # by kernel_gradient. Their mean is the particle approximation of phi*.
    attraction = kernel_matrix @ target_score(particles)
    return (attraction + kernel_gradient) / len(particles)


def run_svgd(
    initial_particles: np.ndarray,
    n_steps: int = N_STEPS,
    step_size: float = STEP_SIZE,
    snapshot_steps: Iterable[int] = (),
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Transport particles with the authors' AdaGrad-with-momentum update.

    Args:
        initial_particles: Starting positions with shape (n_particles,).
        n_steps: Number of SVGD iterations.
        step_size: Master step size multiplied by the AdaGrad direction.
        snapshot_steps: Iterations whose particles should be copied and saved.

    Returns:
        Final particles and a dictionary of requested intermediate states.
    """
    particles = np.asarray(initial_particles, dtype=float).copy()
    if particles.ndim != 1 or len(particles) == 0:
        raise ValueError(
            "initial_particles must be a non-empty one-dimensional array"
        )
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative")
    if step_size <= 0.0:
        raise ValueError("step_size must be positive")

    requested_steps = set(snapshot_steps)
    invalid_steps = requested_steps.difference(range(n_steps + 1))
    if invalid_steps:
        raise ValueError(
            f"snapshot steps outside [0, {n_steps}]: {sorted(invalid_steps)}"
        )

    snapshots: dict[int, np.ndarray] = {}
    historical_gradient = np.zeros_like(particles)

    for step in range(n_steps + 1):
        if step in requested_steps:
            snapshots[step] = particles.copy()

        if step == n_steps:
            break

        direction = svgd_direction(particles)

        # This is the update used in the authors' released implementation. At
        # the first step it initializes the accumulator; afterwards it uses an
        # exponential moving average controlled by ADAGRAD_DECAY.
        if step == 0:
            historical_gradient = direction**2
        else:
            historical_gradient = (
                ADAGRAD_DECAY * historical_gradient
                + (1.0 - ADAGRAD_DECAY) * direction**2
            )

        adjusted_direction = direction / (
            ADAGRAD_EPSILON + np.sqrt(historical_gradient)
        )
        particles += step_size * adjusted_direction

    return particles, snapshots


def kernel_density_estimate(
    grid: np.ndarray,
    particles: np.ndarray,
    bandwidth: float | None = None,
) -> np.ndarray:
    """Estimate the particle density for the Figure 1 visualization.

    The paper states that it uses a kernel density estimator but does not
    publish its plotting bandwidth. We therefore use the standard Silverman
    rule; this affects only the displayed green curve, not the SVGD updates.
    """
    grid = np.asarray(grid, dtype=float)
    particles = np.asarray(particles, dtype=float)

    if bandwidth is None:
        sample_scale = float(np.std(particles, ddof=1))
        bandwidth = max(
            1.06 * sample_scale * len(particles) ** (-1.0 / 5.0),
            1e-3,
        )
    if bandwidth <= 0.0:
        raise ValueError("bandwidth must be positive")

    standardized = (
        grid[:, np.newaxis] - particles[np.newaxis, :]
    ) / bandwidth

    kernels = (
        np.exp(-0.5 * standardized**2)
        / (np.sqrt(2.0 * np.pi) * bandwidth)
    )
    return np.mean(kernels, axis=1)


def save_figure_one(
    snapshots: dict[int, np.ndarray],
) -> Path:
    """Create a modern reproduction of Figure 1 from the paper."""
    grid = np.linspace(-15.0, 15.0, 1_500)
    target_values = target_density(grid)

    # Use one fixed plotting bandwidth in every panel, calculated from q0.
    # This makes the six KDE curves directly comparable and avoids artificially
    # smoothing the final bimodal distribution because its global variance is
    # larger than the within-mode variance.
    initial_particles = snapshots[SNAPSHOT_STEPS[0]]
    plotting_bandwidth = max(
        1.06
        * float(np.std(initial_particles, ddof=1))
        * len(initial_particles) ** (-1.0 / 5.0),
        1e-3,
    )

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(13.5, 7.0),
        sharex=True,
        sharey=True,
    )

    for axis, step in zip(axes.flat, SNAPSHOT_STEPS):
        estimated_density = kernel_density_estimate(
            grid,
            snapshots[step],
            bandwidth=plotting_bandwidth,
        )
        axis.plot(
            grid,
            target_values,
            color="#D62728",
            linestyle="--",
            linewidth=2.2,
            label="Target density",
        )
        axis.plot(
            grid,
            estimated_density,
            color="#168A2E",
            linewidth=2.2,
            label="Particle KDE",
        )
        axis.set_title(f"Iteration {step}")
        axis.set_xlim(-15.0, 15.0)
        axis.set_ylim(0.0, 0.42)
        axis.grid(alpha=0.18)

    for axis in axes[-1, :]:
        axis.set_xlabel("x")
    for axis in axes[:, 0]:
        axis.set_ylabel("Density")

    axes[0, 0].legend(loc="upper right")
    figure.suptitle(
        "Reproduction of Liu & Wang (2016), Figure 1: 1D Gaussian mixture"
    )
    figure.tight_layout()

    output_directory = Path(__file__).resolve().parent / "results"
    output_directory.mkdir(parents=True, exist_ok=True)
    figure_path = output_directory / "svgd_gmm_1d.png"
    figure.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    return figure_path


def main() -> None:
    """Run and validate the published Figure 1 configuration."""
    rng = np.random.default_rng(SEED)
    initial_particles = sample_initial_particles(N_PARTICLES, rng)

    final_particles, snapshots = run_svgd(
        initial_particles,
        snapshot_steps=SNAPSHOT_STEPS,
    )
    figure_path = save_figure_one(snapshots)

    estimated_mean = float(np.mean(final_particles))
    estimated_second_moment = float(np.mean(final_particles**2))
    left_fraction = float(np.mean(final_particles < 0.0))

    assert final_particles.shape == (N_PARTICLES,)
    assert np.all(np.isfinite(final_particles))
    assert set(snapshots) == set(SNAPSHOT_STEPS)
    assert np.mean(initial_particles) < -9.0
    assert np.any(final_particles < 0.0)
    assert np.any(final_particles > 0.0)
    assert abs(estimated_mean - EXPECTED_MEAN) < 0.35
    assert abs(left_fraction - WEIGHTS[0]) < 0.08

    print("Paper configuration:")
    print("  target = 1/3 N(-2, 1) + 2/3 N(2, 1)")
    print("  initial distribution = N(-10, 1)")
    print(f"  particles = {N_PARTICLES}, iterations = {N_STEPS}")
    print("\nMoment estimates after SVGD:")
    print(
        f"  mean:          {estimated_mean:.4f} "
        f"(exact {EXPECTED_MEAN:.4f})"
    )
    print(
        f"  second moment: {estimated_second_moment:.4f} "
        f"(exact {EXPECTED_SECOND_MOMENT:.4f})"
    )
    print("\nMode mass after SVGD:")
    print(f"  left:  {left_fraction:.3f} (target {WEIGHTS[0]:.3f})")
    print(f"  right: {1.0 - left_fraction:.3f} (target {WEIGHTS[1]:.3f})")
    print(f"\nFigure saved to: {figure_path}")
    print("Paper Figure 1 behavior reproduced successfully.")


if __name__ == "__main__":
    main()
