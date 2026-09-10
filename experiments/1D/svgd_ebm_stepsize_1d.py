"""Quick SVGD step-size comparison during one-dimensional EBM training."""

from time import perf_counter
from typing import TypedDict

import numpy as np
import torch
from torch.optim import Adam

from energy_model import DIMENSION, NeuralEnergy
from gmm_1d import SEED, sample_target
from svgd_ebm_1d import svgd_run


STEP_SIZES = [0.005, 0.01, 0.02, 0.05]
N_EPOCHS = 100
BATCH_SIZE = 100
N_PARTICLES = 100
LEARNING_RATE = 1e-3
SVGD_STEPS = 20

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
    finite: bool
    seconds: float


def run_training(step_size: float) -> TrainingResult:
    """Train one EBM and return lightweight stability diagnostics."""

    # Reset both random generators so every step size uses the same model
    # initialization, initial particles, and sequence of positive batches.
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

        negative_particles = svgd_run(
            initial_particles=negative_particles,
            n_steps=SVGD_STEPS,
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

    return {
        "step_size": step_size,
        "loss": final_loss,
        "particle_mean": negative_particles.mean().item(),
        "particle_std": negative_particles.std().item(),
        "particle_min": negative_particles.min().item(),
        "particle_max": negative_particles.max().item(),
        "finite": particles_are_finite and bool(np.isfinite(final_loss)),
        "seconds": elapsed_seconds,
    }


def main() -> None:
    """Run and print the matched step-size comparison."""

    print(
        f"Target particle scale: mean={TARGET_MEAN:.3f}, "
        f"std={TARGET_STD:.3f}\n"
    )

    results: list[TrainingResult] = []

    for step_size in STEP_SIZES:
        result = run_training(step_size)
        results.append(result)

        print(
            f"step_size={result['step_size']:.3f} | "
            f"loss={result['loss']:.6f} | "
            f"mean={result['particle_mean']:.3f} | "
            f"std={result['particle_std']:.3f} | "
            f"range=[{result['particle_min']:.3f}, "
            f"{result['particle_max']:.3f}] | "
            f"finite={result['finite']} | "
            f"time={result['seconds']:.2f}s"
        )

    stable_results = [result for result in results if result["finite"]]

    if stable_results:
        closest = min(
            stable_results,
            key=lambda result: abs(result["particle_std"] - TARGET_STD),
        )
        print(
            "\nClosest final particle standard deviation to the target: "
            f"step_size={closest['step_size']:.3f}."
        )
    else:
        print("\nNo step size completed with finite values.")


if __name__ == "__main__":
    main()
