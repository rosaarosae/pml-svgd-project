"""Train a one-dimensional neural EBM using SVGD negative samples."""

import numpy as np
import torch
from torch.optim import Adam

from energy_model import DIMENSION, NeuralEnergy
from gmm_1d import SEED, sample_target
from svgd_ebm_1d import svgd_run

N_EPOCHS = 500
BATCH_SIZE = 100
N_PARTICLES = 100

LEARNING_RATE = 1e-3

#we still have to check these SVGD parameters
SVGD_STEPS = 20
SVGD_STEP_SIZE = 0.05


def main() -> None:
    torch.manual_seed(SEED)
    model = NeuralEnergy()
    learning_rate = 1e-3

    # Adam updates the neural network parameters to reduce the training loss.
    optimizer = Adam(
        model.parameters(),
        lr=learning_rate,
    )
    # Generator used to draw reproducible positive samples from the target data.
    rng = np.random.default_rng(SEED)
    batch_size = 100

    # Draw real examples from the target mixture.
    positive_samples_numpy = sample_target(
        batch_size,
        rng,
    )

    # Convert the NumPy samples into a PyTorch tensor for the neural network.
    positive_samples = torch.tensor(
        positive_samples_numpy,
        dtype=torch.float32,
    ).reshape(-1, DIMENSION)
    print("Positive samples shape:", positive_samples.shape)
    print("First five positive samples:")
    print(positive_samples[:5])

    assert positive_samples.shape == (batch_size, DIMENSION)
    assert torch.all(torch.isfinite(positive_samples))

    print("\nPositive-sample check passed.")

    # Initialize the persistent negative particles from real data. SVGD will
    # move this independent copy while the positive samples remain unchanged.
    negative_particles = positive_samples.detach().clone()
    particles_before_svgd = negative_particles.clone()

    # Move the negative particles towards the distribution defined by the
    # current neural energy. The model is still randomly initialized here.
    negative_particles = svgd_run(
        initial_particles=negative_particles,
        n_steps=SVGD_STEPS,
        step_size=SVGD_STEP_SIZE,
        score_function=model.model_score,
    )

    mean_movement = torch.mean(
        torch.abs(negative_particles - particles_before_svgd)
    ).item()

    assert negative_particles.shape == (N_PARTICLES, DIMENSION)
    assert torch.all(torch.isfinite(negative_particles))
    assert mean_movement > 0.0

    print("Negative particles shape:", negative_particles.shape)
    print(f"Mean particle movement: {mean_movement:.6f}")


if __name__ == "__main__":
    main()
