"""Quick Langevin step-size comparison during one-dimensional EBM training."""

from time import perf_counter
from typing import TypedDict

import numpy as np
import torch
from torch.optim import Adam

from energy_model import DIMENSION, NeuralEnergy
from gmm_1d import SEED, sample_target, target_density
from langevin_ebm_1d import langevin_run


STEP_SIZES = [0.1, 0.3]
N_EPOCHS = 500
BATCH_SIZE = 200
N_PARTICLES = 200
LEARNING_RATE = 1e-3
LANGEVIN_STEPS = 20

# The target mixture has mean 2/3 and variance 41/9.
TARGET_MEAN = 2.0 / 3.0
TARGET_STD = np.sqrt(41.0 / 9.0)


class TrainingResult(TypedDict):
    """Diagnostics returned by one matched EBM training run."""

    step_size: float
    loss: float
    particle_mean: float
    particle_std: float
    particle_min: float
    particle_max: float
    density_error: float
    left_mass: float
    right_mass: float
    finite: bool
    seconds: float


def run_training(step_size: float) -> TrainingResult:
    """Train one EBM and return lightweight stability diagnostics."""

    # Reset the generators so every step size uses matched random inputs.
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    model = NeuralEnergy()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    initial_samples = sample_target(N_PARTICLES, rng)
    negative_particles = torch.tensor(
        initial_samples,
        dtype=torch.float32,
    ).reshape(-1, DIMENSION)

    start_time = perf_counter()
    final_loss = float("nan")

    for _ in range(N_EPOCHS):
        positive_samples = torch.tensor(
            sample_target(BATCH_SIZE, rng),
            dtype=torch.float32,
        ).reshape(-1, DIMENSION)

        negative_particles = langevin_run(
            initial_particles=negative_particles,
            n_steps=LANGEVIN_STEPS,
            step_size=step_size,
            score_function=model.model_score,
        )

        positive_energy = model(positive_samples).mean()
        negative_energy = model(negative_particles).mean()
        loss = positive_energy - negative_energy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        final_loss = loss.item()

        if not torch.isfinite(loss) or not torch.all(
            torch.isfinite(negative_particles)
        ):
            break

    elapsed_seconds = perf_counter() - start_time
    particles_are_finite = bool(
        torch.all(torch.isfinite(negative_particles)).item()
    )
    run_is_finite = particles_are_finite and bool(np.isfinite(final_loss))

    density_error = float("inf")
    left_mass = float("nan")
    right_mass = float("nan")

    if run_is_finite:
        grid_values = torch.linspace(-8.0, 8.0, 1000)
        grid_points = grid_values.reshape(-1, DIMENSION)

        with torch.no_grad():
            grid_energies = model(grid_points)

        unnormalized_density = torch.exp(
            -(grid_energies - grid_energies.min())
        )
        learned_density = unnormalized_density / torch.trapezoid(
            y=unnormalized_density,
            x=grid_values,
        )
        exact_density = torch.as_tensor(
            target_density(grid_values.numpy()),
            dtype=learned_density.dtype,
        )
        density_error = torch.trapezoid(
            y=(learned_density - exact_density).square(),
            x=grid_values,
        ).item()

        left_indicator = (grid_values < 0.0).to(learned_density.dtype)
        left_mass = torch.trapezoid(
            y=learned_density * left_indicator,
            x=grid_values,
        ).item()
        right_mass = 1.0 - left_mass

    return {
        "step_size": step_size,
        "loss": final_loss,
        "particle_mean": negative_particles.mean().item(),
        "particle_std": negative_particles.std().item(),
        "particle_min": negative_particles.min().item(),
        "particle_max": negative_particles.max().item(),
        "density_error": density_error,
        "left_mass": left_mass,
        "right_mass": right_mass,
        "finite": run_is_finite,
        "seconds": elapsed_seconds,
    }


def main() -> None:
    """Run and print the matched Langevin step-size comparison."""

    print(
        f"Target particle scale: mean={TARGET_MEAN:.3f}, "
        f"std={TARGET_STD:.3f}\n"
    )

    results: list[TrainingResult] = []

    for step_size in STEP_SIZES:
        result = run_training(step_size)
        results.append(result)

        print(
            f"step_size={result['step_size']:.4f} | "
            f"loss={result['loss']:.6f} | "
            f"mean={result['particle_mean']:.3f} | "
            f"std={result['particle_std']:.3f} | "
            f"density_error={result['density_error']:.6f} | "
            f"mass={result['left_mass']:.3f}/{result['right_mass']:.3f} | "
            f"range=[{result['particle_min']:.3f}, "
            f"{result['particle_max']:.3f}] | "
            f"finite={result['finite']} | "
            f"time={result['seconds']:.2f}s"
        )

    stable_results = [result for result in results if result["finite"]]

    if stable_results:
        best = min(
            stable_results,
            key=lambda result: result["density_error"],
        )
        print(
            "\nLowest learned-density error: "
            f"step_size={best['step_size']:.4f}."
        )
    else:
        print("\nNo step size completed with finite values.")


if __name__ == "__main__":
    main()
