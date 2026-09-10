"""Langevin sampler for the one-dimensional neural energy model."""

from collections.abc import Callable

import torch

from energy_model import DIMENSION, NeuralEnergy
from gmm_1d import SEED


def langevin_step(
    particles: torch.Tensor,
    step_size: float,
    score_function: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Apply one Langevin update."""

    if particles.ndim != 2 or particles.shape[1] != DIMENSION:
        raise ValueError(
            f"particles must have shape "
            f"(n_particles, {DIMENSION})"
        )

    if len(particles) == 0:
        raise ValueError("particles cannot be empty")

    if step_size <= 0.0:
        raise ValueError("step_size must be positive")

    score_values = score_function(particles)

    if score_values.shape != particles.shape:
        raise ValueError(
            "score_function must return a tensor "
            "with the same shape as particles"
        )

    noise_scale = torch.sqrt(
        particles.new_tensor(2.0 * step_size)
    )

    updated_particles = (
        particles
        + step_size * score_values
        + noise_scale * torch.randn_like(particles)
    )

    return updated_particles.detach()


def langevin_run(
    initial_particles: torch.Tensor,
    n_steps: int,
    step_size: float,
    score_function: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Apply several Langevin updates."""

    if n_steps < 0:
        raise ValueError("n_steps must be non-negative")

    particles = initial_particles.clone().detach()

    for _ in range(n_steps):
        particles = langevin_step(
            particles=particles,
            step_size=step_size,
            score_function=score_function,
        )

    return particles.detach()

def main() -> None:
    """Check the complete neural-EBM Langevin sampler."""

    torch.manual_seed(SEED)

    model = NeuralEnergy()

    initial_particles = torch.linspace(
        -5.0,
        5.0,
        50,
    ).reshape(-1, DIMENSION)

    particles_before = initial_particles.clone()

    # Reset the seed so the Langevin noise is reproducible.
    torch.manual_seed(SEED)

    final_particles = langevin_run(
        initial_particles=initial_particles,
        n_steps=100,
        step_size=0.01,
        score_function=model.model_score,
    )

    mean_movement = torch.mean(
        torch.abs(
            final_particles - particles_before
        )
    ).item()

    print("Initial particles shape:", initial_particles.shape)
    print("Final particles shape:", final_particles.shape)
    print(f"Mean particle movement: {mean_movement:.6f}")
    print(
        "Final particle range:",
        final_particles.min().item(),
        "to",
        final_particles.max().item(),
    )

    assert initial_particles.shape == (50, DIMENSION)
    assert final_particles.shape == initial_particles.shape

    assert torch.all(torch.isfinite(final_particles))
    assert mean_movement > 0.0

    # langevin_run must not modify the original tensor.
    assert torch.allclose(
        initial_particles,
        particles_before,
    )

    # Check reproducibility with the same random seed.
    torch.manual_seed(SEED)

    repeated_particles = langevin_run(
        initial_particles=initial_particles,
        n_steps=100,
        step_size=0.01,
        score_function=model.model_score,
    )

    assert torch.allclose(
        final_particles,
        repeated_particles,
    )

    print("\nAll neural-EBM Langevin checks passed.")


if __name__ == "__main__":
    main()
