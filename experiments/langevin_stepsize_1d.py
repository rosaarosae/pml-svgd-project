"""We compare the effect of different step sizes on the Langevin dynamics in 1D."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from experiments.langevin_gmm_1d import NPARTICLES, SEED, density, langevin

#we put different values of the step size in a list
step_sizes = [0.003, 0.01, 0.03, 0.1,0.3, 1.0]

def main() -> None:
    #we create one initial particle set for all the step sizes
    initial_rng = np.random.default_rng(SEED)
    # A scale of 3.0 spreads the initial particles widely around zero,
    # allowing Langevin to demonstrate whether it can find both target modes.
    initial_particles = initial_rng.normal(
        loc=0.0,
        scale=3.0,
        size=NPARTICLES,
    )

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
    x = np.linspace(-5, 5, 1000)
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
