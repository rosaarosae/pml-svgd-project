"""We check which method works better in 1D."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from svgd_gmm_1d import (
    SEED,
    N_PARTICLES,
    N_STEPS,
    STEP_SIZE as SVGD_STEP_SIZE,
    target_density,
    svgd_direction,
)

from langevin_gmm_1d import (
    langevin,
    STEPSIZE as LANGEVIN_STEP_SIZE,
)

def main() -> None:
    # we create one initial particle set for both methods
    initial_rng = np.random.default_rng(SEED)
    initial_particles = initial_rng.normal(
        loc=0.0,
        scale=3.0,
        size=N_PARTICLES,
    )

    # we run the Langevin dynamics
    noise_rng = np.random.default_rng(SEED + 1)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        final_particles_langevin = langevin(
            initial_particles.copy(),
            noise_rng,
            LANGEVIN_STEP_SIZE,
        )

    # we run the SVGD dynamics
    svgd_particles = initial_particles.copy()
    accumulated_squared_gradient = np.zeros_like(svgd_particles)

    for _ in range(N_STEPS):
        direction = svgd_direction(svgd_particles)

        accumulated_squared_gradient += direction**2

        svgd_particles += SVGD_STEP_SIZE  * direction / (
            1e-6 + np.sqrt(accumulated_squared_gradient)
        )

    # we plot the results
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15, 5),
        sharex=True,
        sharey=True,
    )

    axes[1].tick_params(labelleft=True)

    x = np.linspace(-5, 5, 1000)

    axes[0].plot(
        x,
        target_density(x),
        label="Target density",
        color="red",
    )
    axes[0].hist(
        final_particles_langevin,
        bins=30,
        density=True,
        alpha=0.5,
        label="Langevin particles",
    )
    axes[0].set_title("Langevin Dynamics")
    axes[0].legend()

    axes[1].plot(
        x,
        target_density(x),
        label="Target density",
        color="red",
    )
    axes[1].hist(
        svgd_particles,
        bins=30,
        density=True,
        alpha=0.5,
        label="SVGD particles",
    )
    axes[1].set_title("SVGD Dynamics")
    axes[1].legend()

    # we compute simple diagnostics
    langevin_log_density = np.log(
        target_density(final_particles_langevin) + 1e-12
    ).mean()

    svgd_log_density = np.log(
        target_density(svgd_particles) + 1e-12
    ).mean()

    langevin_left_fraction = np.mean(
        final_particles_langevin < 0.0
    )

    svgd_left_fraction = np.mean(
        svgd_particles < 0.0
    )

    print(
        f"Langevin mean log density: {langevin_log_density:.3f}"
    )
    print(
        f"SVGD mean log density:     {svgd_log_density:.3f}"
    )
    print(
        f"Langevin left fraction:    {langevin_left_fraction:.2f}"
    )
    print(
        f"SVGD left fraction:        {svgd_left_fraction:.2f}"
    )
    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    figure_path = output_dir / "compare_svgd_langevin_1d.png"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    
    plt.show()


if __name__ == "__main__":
    main()