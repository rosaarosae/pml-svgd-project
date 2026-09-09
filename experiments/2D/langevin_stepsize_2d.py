"""Evaluate Langevin step sizes visually and across multiple random seeds."""

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from gmm_2d import (
    MEANS,
    SEED,
    TRANSPORT_X_LIMITS,
    WEIGHTS,
    Y_LIMITS,
    mixture_density,
    sample_initial_particles,
    target_energy,
    target_score,
)
from langevin_2d import N_PARTICLES, N_STEPS, langevin_dynamics


STEP_SIZES = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
SEEDS = [SEED, 17, 27, 37, 47]


def compute_mode_fractions(particles: np.ndarray) -> np.ndarray:
    """Assign particles to their nearest mode and return mode proportions."""
    differences = particles[:, np.newaxis, :] - MEANS[np.newaxis, :, :]
    squared_distances = np.sum(differences**2, axis=2)
    nearest_modes = np.argmin(squared_distances, axis=1)
    mode_counts = np.bincount(nearest_modes, minlength=len(MEANS))
    return mode_counts / len(particles)


def main() -> None:
    n_seeds = len(SEEDS)
    n_step_sizes = len(STEP_SIZES)
    mean_energies = np.zeros((n_seeds, n_step_sizes))
    mode_errors = np.zeros((n_seeds, n_step_sizes))
    runtimes = np.zeros((n_seeds, n_step_sizes))
    representative_particles = []

    # Repeat every step size with several seeds. Within each seed, every step
    # size receives the same initial particles and random-noise stream.
    for seed_index, seed in enumerate(SEEDS):
        initial_rng = np.random.default_rng(seed)
        initial_particles = sample_initial_particles(N_PARTICLES, initial_rng)

        for step_index, step_size in enumerate(STEP_SIZES):
            noise_rng = np.random.default_rng(seed + 1)
            start_time = perf_counter()
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                final_particles = langevin_dynamics(
                    initial_particles=initial_particles,
                    n_steps=N_STEPS,
                    step_size=step_size,
                    score_function=target_score,
                    rng=noise_rng,
                )
            runtimes[seed_index, step_index] = perf_counter() - start_time

            if not np.all(np.isfinite(final_particles)):
                mean_energies[seed_index, step_index] = np.nan
                mode_errors[seed_index, step_index] = np.nan
                continue

            mode_fractions = compute_mode_fractions(final_particles)
            mean_energies[seed_index, step_index] = np.mean(
                target_energy(final_particles)
            )
            mode_errors[seed_index, step_index] = np.abs(
                mode_fractions - WEIGHTS
            ).sum()

            # Seed 7 is retained for the visual particle comparison.
            if seed_index == 0:
                representative_particles.append(final_particles.copy())

    energy_means = np.nanmean(mean_energies, axis=0)
    energy_stds = np.nanstd(mean_energies, axis=0, ddof=1)
    error_means = np.nanmean(mode_errors, axis=0)
    error_stds = np.nanstd(mode_errors, axis=0, ddof=1)
    runtime_means = np.mean(runtimes, axis=0)

    print(f"Results across {n_seeds} seeds:")
    print("step size | mean energy       | mode error        | mean time")
    print("----------+-------------------+-------------------+----------")
    for index, step_size in enumerate(STEP_SIZES):
        print(
            f"{step_size:>9g} | "
            f"{energy_means[index]:.3f} ± {energy_stds[index]:.3f} | "
            f"{error_means[index]:.3f} ± {error_stds[index]:.3f} | "
            f"{runtime_means[index]:.3f} s"
        )

    output_directory = Path(__file__).resolve().parent / "results"
    output_directory.mkdir(parents=True, exist_ok=True)

    # Figure 1: final particles for the representative seed 7.
    x_values = np.linspace(*TRANSPORT_X_LIMITS, 220)
    y_values = np.linspace(*Y_LIMITS, 150)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    grid_points = np.column_stack([x_grid.ravel(), y_grid.ravel()])
    density_grid = mixture_density(grid_points).reshape(x_grid.shape)

    particle_figure, particle_axes = plt.subplots(
        2, 3, figsize=(13, 9), sharex=True, sharey=True
    )
    for axis, step_size, particles in zip(
        particle_axes.ravel(), STEP_SIZES, representative_particles
    ):
        axis.contourf(
            x_grid, y_grid, density_grid, levels=40, cmap="viridis"
        )
        axis.scatter(
            particles[:, 0], particles[:, 1], s=7, color="red", alpha=0.35
        )
        axis.scatter(
            MEANS[:, 0],
            MEANS[:, 1],
            marker="x",
            s=60,
            linewidths=2,
            color="white",
        )
        axis.set_title(f"Step size = {step_size}")
        axis.set_xlabel("x₁")
        axis.set_ylabel("x₂")
        axis.set_xlim(*TRANSPORT_X_LIMITS)
        axis.set_ylim(*Y_LIMITS)

    particle_figure.suptitle(
        f"Langevin step-size comparison (representative seed {SEEDS[0]})"
    )
    particle_figure.tight_layout()
    particle_path = output_directory / "langevin_stepsize_2d.png"
    particle_figure.savefig(particle_path, dpi=180, bbox_inches="tight")

    # Figure 2: mean and standard deviation across all five seeds.
    summary_figure, summary_axes = plt.subplots(1, 2, figsize=(11, 4.5))
    summary_axes[0].errorbar(
        STEP_SIZES, energy_means, yerr=energy_stds, marker="o", capsize=4
    )
    summary_axes[0].set_title("Mean target energy")
    summary_axes[0].set_xlabel("Step size")
    summary_axes[0].set_ylabel("Energy")
    summary_axes[0].set_xscale("log")
    summary_axes[0].grid(alpha=0.25)

    summary_axes[1].errorbar(
        STEP_SIZES,
        error_means,
        yerr=error_stds,
        marker="o",
        capsize=4,
        color="tab:orange",
    )
    summary_axes[1].set_title("Mixture-weight error")
    summary_axes[1].set_xlabel("Step size")
    summary_axes[1].set_ylabel("Total absolute error")
    summary_axes[1].set_xscale("log")
    summary_axes[1].grid(alpha=0.25)

    summary_figure.suptitle(
        f"Langevin step-size evaluation across {n_seeds} seeds"
    )
    summary_figure.tight_layout()
    summary_path = output_directory / "langevin_stepsize_summary_2d.png"
    summary_figure.savefig(summary_path, dpi=180, bbox_inches="tight")

    print(f"\nParticle figure saved to: {particle_path}")
    print(f"Summary figure saved to:  {summary_path}")
    plt.show()


if __name__ == "__main__":
    main()
