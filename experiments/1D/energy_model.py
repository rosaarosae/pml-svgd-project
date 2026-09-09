"""Learnable neural energy function for the one-dimensional EBM experiment."""

import torch
from torch import nn

DIMENSION = 1
HIDDEN_UNITS = 32 #Number of hidden units in the neural network
CONFINING_SCALE = 4.0 #We have chosen a confining scale of 4.0 to ensure that the energy function grows sufficiently fast outside the region of interest

class NeuralEnergy(nn.Module):
    """Neural energy function for the one-dimensional EBM experiment."""

    # Initialize the neural network architecture.
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(DIMENSION, HIDDEN_UNITS),
            nn.Tanh(),
            nn.Linear(HIDDEN_UNITS, HIDDEN_UNITS),
            nn.Tanh(),
            nn.Linear(HIDDEN_UNITS, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the energy of input x.

        Args:
            x: Input tensor of shape (batch_size, DIMENSION).

        Returns:
            Energy tensor of shape (batch_size,).
        """
        if x.ndim != 2 or x.shape[1] != DIMENSION:
            raise ValueError(
                f"x must have shape (batch_size, {DIMENSION}), "
                f"got {tuple(x.shape)}"
            )

        learned_energy = self.network(x)
        confining_energy = 0.5 * (x / CONFINING_SCALE).square()
        total_energy = learned_energy + confining_energy

        return total_energy.squeeze(-1)

    def model_score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the score required by SVGD.

        For an energy-based model, the score satisfies
        grad_x log p_theta(x) = -grad_x E_theta(x).

        Args:
            x: Input tensor with shape (batch_size, DIMENSION).

        Returns:
            Score tensor with shape (batch_size, DIMENSION).
        """
        points = x.detach().clone().requires_grad_(True)
        energy = self(points)

        energy_gradient = torch.autograd.grad(
            outputs=energy.sum(),
            inputs=points,
            create_graph=False,
        )[0]

        return -energy_gradient.detach()

def main() -> None:
    """Run basic checks of the neural energy and its score."""

    # Make the random initialization reproducible.
    torch.manual_seed(7)

    # Create the neural energy model.
    model = NeuralEnergy()

    # Five points at which we evaluate the initial random model.
    test_points = torch.tensor(
        [[-5.0], [-2.0], [0.0], [2.0], [5.0]],
        dtype=torch.float32,
    )

    # Compute one energy and one score for every point.
    energy_values = model(test_points)
    score_values = model.model_score(test_points)

    print("Test points:")
    print(test_points)

    print("\nInitial energy:")
    print(energy_values)

    print("\nInitial score:")
    print(score_values)

    # The model must return one scalar energy for every point.
    assert energy_values.shape == (len(test_points),)

    # SVGD requires one score vector with the same shape as every point.
    assert score_values.shape == test_points.shape

    # Training cannot continue if the model produces NaN or infinite values.
    assert torch.all(torch.isfinite(energy_values))
    assert torch.all(torch.isfinite(score_values))

    print("\nAll neural-energy and score checks passed.")


if __name__ == "__main__":
    main()
