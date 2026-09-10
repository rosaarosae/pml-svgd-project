"""Train a one-dimensional neural EBM using SVGD negative samples."""

import numpy as np
import torch
from torch.optim import Adam

from energy_model import DIMENSION, NeuralEnergy
from gmm_1d import SEED, sample_target, target_density
from svgd_ebm_1d import svgd_run
from pathlib import Path
import matplotlib.pyplot as plt

N_EPOCHS = 500
BATCH_SIZE = 200
N_PARTICLES = 200
LEARNING_RATE = 1e-3


SVGD_STEPS = 20
SVGD_STEP_SIZE = 0.005


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

    # Initialize the negative particles from real data.
    # SVGD will move this independent copy while the positive samples
    # remain unchanged.
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
        torch.abs(
            negative_particles - particles_before_svgd
        )
    ).item()

    assert negative_particles.shape == (N_PARTICLES, DIMENSION)
    assert torch.all(torch.isfinite(negative_particles))
    assert mean_movement > 0.0

    print("Negative particles shape:", negative_particles.shape)
    print(f"Mean particle movement: {mean_movement:.6f}")

    # Calculate the mean energy assigned to real data.
    positive_energy = model(
        positive_samples
    ).mean()

    # Calculate the mean energy assigned to SVGD negative particles.
    negative_energy = model(
        negative_particles
    ).mean()

    print("Mean positive energy:", positive_energy.item())
    print("Mean negative energy:", negative_energy.item())

    # Alternate persistent SVGD updates with neural-energy updates.
    for epoch in range(N_EPOCHS):
        # Draw a fresh batch of real examples from the target mixture.
        positive_samples_numpy = sample_target(
            BATCH_SIZE,
            rng,
        )

        positive_samples = torch.tensor(
            positive_samples_numpy,
            dtype=torch.float32,
        ).reshape(-1, DIMENSION)

        # Update the persistent negative particles using the current EBM.
        negative_particles = svgd_run(
            initial_particles=negative_particles,
            n_steps=SVGD_STEPS,
            step_size=SVGD_STEP_SIZE,
            score_function=model.model_score,
        )

        positive_energy = model(
            positive_samples
        ).mean()

        negative_energy = model(
            negative_particles
        ).mean()

        loss = positive_energy - negative_energy

        assert torch.isfinite(loss)
        assert torch.all(torch.isfinite(negative_particles))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            particle_mean = negative_particles.mean().item()
            particle_std = negative_particles.std().item()
            particle_min = negative_particles.min().item()
            particle_max = negative_particles.max().item()

            print(
                f"Epoch {epoch}: "
                f"loss={loss.item():.6f}, "
                f"positive_energy={positive_energy.item():.6f}, "
                f"negative_energy={negative_energy.item():.6f}, "
                f"particle_mean={particle_mean:.3f}, "
                f"particle_std={particle_std:.3f}, "
                f"range=[{particle_min:.3f}, {particle_max:.3f}]"
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
    # Convert the grid from shape (1000, 1) to shape (1000,).
    grid_values = grid_points.squeeze(-1)

    # Subtracting a constant does not change the final normalized
    # density. It only prevents exponentials from becoming too large.
    minimum_energy = grid_energies.min()
    shifted_energies = grid_energies - minimum_energy

    # Convert energy into an unnormalized density: exp(-energy(x)).
    unnormalized_density = torch.exp(
        -shifted_energies
    )

    # Approximate the normalization constant over the grid.
    normalization_constant = torch.trapezoid(
        y=unnormalized_density,
        x=grid_values,
    )

    # Normalize the learned density.
    learned_density = (
        unnormalized_density
        / normalization_constant
    )

    # Check that the normalized density integrates to one.
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
    # Convert the results to NumPy for plotting.
    grid_numpy = grid_values.numpy()
    learned_density_numpy = learned_density.numpy()
    negative_particles_numpy = (
        negative_particles.squeeze(-1).numpy()
    )

    # Evaluate the exact target density on the same grid.
    exact_density = target_density(grid_numpy)
    exact_density_tensor = torch.as_tensor(
        exact_density,
        dtype=learned_density.dtype,
    )

    # Measure the integrated squared error between both densities.
    density_squared_error = torch.trapezoid(
        y=(learned_density - exact_density_tensor).square(),
        x=grid_values,
    )

    # Measure the probability mass on either side of zero.
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

    # Create the comparison figure.
    figure, axis = plt.subplots(
        figsize=(8, 5)
    )

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
        label="SVGD negative particles",
    )

    axis.set_title("One-dimensional EBM trained with SVGD")
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

    figure_path = output_directory / "ebm_svgd_1d.png"

    figure.savefig(
        figure_path,
        dpi=180,
        bbox_inches="tight",
    )

    print("Figure saved to:", figure_path)

    plt.show()


if __name__ == "__main__":
    main()
