"""Train a one-dimensional neural EBM using Langevin negative samples.

This experiment trains the same neural energy model used in the SVGD
experiment. The model architecture, training data, optimizer, number of
epochs, batch size, and number of particles remain fixed. Only the method
used to update the persistent negative particles changes from SVGD to
Langevin dynamics.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adam

from energy_model import DIMENSION, NeuralEnergy
from gmm_1d import SEED, sample_target, target_density
from langevin_ebm_1d import langevin_run


N_EPOCHS = 500
BATCH_SIZE = 200
N_PARTICLES = 200
LEARNING_RATE = 1e-3

LANGEVIN_STEPS = 20
LANGEVIN_STEP_SIZE = 0.1


def main() -> None:
    torch.manual_seed(SEED)

    model = NeuralEnergy()

    # Adam updates the neural network parameters to reduce the training loss.
    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # Generator used to draw reproducible positive samples from the target data.
    rng = np.random.default_rng(SEED)

    # Draw real examples from the target mixture.
    positive_samples_numpy = sample_target(
        BATCH_SIZE,
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

    assert positive_samples.shape == (BATCH_SIZE, DIMENSION)
    assert torch.all(torch.isfinite(positive_samples))

    print("\nPositive-sample check passed.")

    # Initialize the persistent negative particles from real data.
    negative_particles = positive_samples.detach().clone()

    # Alternate between updating the negative particles and
    # updating the neural energy model.
    for epoch in range(N_EPOCHS):

        # Draw a new batch of real samples from the target distribution.
        positive_samples_numpy = sample_target(
            BATCH_SIZE,
            rng,
        )

        # Convert the real samples from NumPy to a PyTorch tensor.
        positive_samples = torch.tensor(
            positive_samples_numpy,
            dtype=torch.float32,
        ).reshape(-1, DIMENSION)

        # Move the persistent negative particles using Langevin dynamics.
        #
        # The particles from the previous epoch are reused, so the
        # Langevin chains remain persistent throughout training.
        negative_particles = langevin_run(
            initial_particles=negative_particles,
            n_steps=LANGEVIN_STEPS,
            step_size=LANGEVIN_STEP_SIZE,
            score_function=model.model_score,
        )

        # Calculate the average energy assigned to real data.
        positive_energy = model(
            positive_samples
        ).mean()

        # Calculate the average energy assigned to negative particles.
        negative_energy = model(
            negative_particles
        ).mean()

        # Contrastive EBM loss:
        #
        # Minimizing this encourages the model to assign lower energy
        # to real data and higher energy to negative samples.
        loss = positive_energy - negative_energy

        # Stop immediately if the training becomes numerically unstable.
        assert torch.isfinite(loss)
        assert torch.all(
            torch.isfinite(negative_particles)
        )

        # Remove gradients left from the previous epoch.
        optimizer.zero_grad()

        # Calculate the gradient of the loss with respect to the
        # neural-network parameters.
        loss.backward()

        # Update the parameters of the neural energy model.
        optimizer.step()

        # Print diagnostics every 50 epochs.
        if epoch % 50 == 0:
            particle_mean = (
                negative_particles.mean().item()
            )
            particle_std = (
                negative_particles.std().item()
            )
            particle_min = (
                negative_particles.min().item()
            )
            particle_max = (
                negative_particles.max().item()
            )

            print(
                f"Epoch {epoch}: "
                f"loss={loss.item():.6f}, "
                f"positive_energy={positive_energy.item():.6f}, "
                f"negative_energy={negative_energy.item():.6f}, "
                f"particle_mean={particle_mean:.3f}, "
                f"particle_std={particle_std:.3f}, "
                f"range=[{particle_min:.3f}, "
                f"{particle_max:.3f}]"
            )

    # Evaluate the learned energy on a fixed grid after training.
    grid_points = torch.linspace(
        -8.0,
        8.0,
        1000,
    ).reshape(-1, DIMENSION)

    with torch.no_grad():
        grid_energies = model(grid_points)

    assert grid_energies.shape == (len(grid_points),)
    assert torch.all(torch.isfinite(grid_energies))

    print("\nGrid-energy check passed.")
    print("Grid points shape:", grid_points.shape)
    print("Grid energies shape:", grid_energies.shape)
    print(
        "Energy range:",
        grid_energies.min().item(),
        "to",
        grid_energies.max().item(),
    )

    grid_values = grid_points.squeeze(-1)

    # Shift the energies to prevent numerical overflow in the exponential.
    minimum_energy = grid_energies.min()
    shifted_energies = grid_energies - minimum_energy
    unnormalized_density = torch.exp(-shifted_energies)

    normalization_constant = torch.trapezoid(
        y=unnormalized_density,
        x=grid_values,
    )
    learned_density = (
        unnormalized_density
        / normalization_constant
    )
    density_integral = torch.trapezoid(
        y=learned_density,
        x=grid_values,
    )

    assert torch.all(torch.isfinite(learned_density))
    assert torch.isclose(
        density_integral,
        torch.ones_like(density_integral),
        atol=1e-4,
    )

    print(
        "Normalization constant:",
        normalization_constant.item(),
    )
    print(
        "Learned-density integral:",
        density_integral.item(),
    )

    grid_numpy = grid_values.numpy()
    learned_density_numpy = learned_density.numpy()
    negative_particles_numpy = (
        negative_particles.squeeze(-1).numpy()
    )

    exact_density = target_density(grid_numpy)
    exact_density_tensor = torch.as_tensor(
        exact_density,
        dtype=learned_density.dtype,
    )

    density_squared_error = torch.trapezoid(
        y=(learned_density - exact_density_tensor).square(),
        x=grid_values,
    )

    left_indicator = (grid_values < 0.0).to(learned_density.dtype)
    right_indicator = 1.0 - left_indicator

    learned_left_mass = torch.trapezoid(
        y=learned_density * left_indicator,
        x=grid_values,
    )
    learned_right_mass = torch.trapezoid(
        y=learned_density * right_indicator,
        x=grid_values,
    )
    exact_left_mass = torch.trapezoid(
        y=exact_density_tensor * left_indicator,
        x=grid_values,
    )
    exact_right_mass = torch.trapezoid(
        y=exact_density_tensor * right_indicator,
        x=grid_values,
    )

    print(
        "Integrated squared density error:",
        density_squared_error.item(),
    )
    print(
        "Learned left/right mass:",
        learned_left_mass.item(),
        learned_right_mass.item(),
    )
    print(
        "Exact left/right mass:",
        exact_left_mass.item(),
        exact_right_mass.item(),
    )

    figure, axis = plt.subplots(figsize=(8, 5))

    axis.plot(
        grid_numpy,
        exact_density,
        color="black",
        linewidth=2,
        label="Exact target density",
    )
    axis.plot(
        grid_numpy,
        learned_density_numpy,
        color="blue",
        linewidth=2,
        label="Learned EBM density",
    )
    axis.hist(
        negative_particles_numpy,
        bins=30,
        density=True,
        alpha=0.3,
        color="orange",
        label="Langevin negative particles",
    )

    axis.set_title("One-dimensional EBM trained with Langevin")
    axis.set_xlabel("x")
    axis.set_ylabel("Density")
    axis.legend()

    figure.tight_layout()

    output_directory = (
        Path(__file__).resolve().parent / "results"
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_path = output_directory / "ebm_langevin_1d.png"
    figure.savefig(
        figure_path,
        dpi=180,
        bbox_inches="tight",
    )

    print("Figure saved to:", figure_path)

    plt.show()


if __name__ == "__main__":
    main()
