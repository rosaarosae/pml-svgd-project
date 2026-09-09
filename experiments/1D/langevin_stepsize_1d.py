"""Supplementary Langevin step-size check on the paper's 1D target."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from langevin_gmm_1d import NPARTICLES, SEED, density, langevin
from gmm_1d import sample_initial_particles

#we put different values of the step size in a list
step_sizes = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]

def main() -> None:
    # Use the paper's difficult q0(x) = N(-10, 1) initialization for every
    # step size so that only the Langevin step size changes.
    initial_rng = np.random.default_rng(SEED)
    initial_particles = sample_initial_particles(NPARTICLES, initial_rng)

    #we store the final particles for each step size
    results = {}
    #we run the Langevin dynamics for each step size
    for step_size in step_sizes:
        noise_rng = np.random.default_rng(SEED + 1)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            final_particles = langevin(
                initial_particles.copy(),
                noise_rng,
                step_size,
            )
        results[step_size] = final_particles
        # Measure the result only when the sampler remains stable.
        if np.all(np.isfinite(final_particles)):
            mean_log_density = np.log(
                density(final_particles) + 1e-12
            ).mean()
            left_fraction = np.mean(final_particles < 0.0)

            print(
                f"Step size {step_size}: "
                f"mean log density = {mean_log_density:.3f}, "
                f"left = {left_fraction:.2f}, "
                f"right = {1.0 - left_fraction:.2f}"
            )
        else:
            print(f"Step size {step_size}: unstable")

    #we plot the results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    x = np.linspace(-15, 8, 1000)
    for i, step_size in enumerate(step_sizes):
        ax = axes[i]
        final_particles = results[step_size]
        ax.plot(x, density(x), label='Target density', color='red')

        if np.all(np.isfinite(final_particles)):
            ax.hist(final_particles, bins=50, density=True, alpha=0.5, label='Final particles')
        else:
            ax.text(0.5, 0.5, 'Unstable', ha='center', va='center', transform=ax.transAxes)

        ax.set_title(f'Step size: {step_size}')
        ax.set_xlabel('x')
        ax.set_ylabel('Density')
        ax.legend()

    plt.tight_layout()
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "langevin_stepsize_1d.png")
    plt.show()


if __name__ == "__main__":
    main()
