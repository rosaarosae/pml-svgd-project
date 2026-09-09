"""SVGD sampler for the one-dimensional neural energy model."""

from collections.abc import Callable

import torch

from energy_model import DIMENSION, NeuralEnergy

#we reuse the kernel  from svgd_gmm_1d, but we adapt it to pytorch

def rbf_kernel(
    particles: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the RBF kernel matrix and its adaptive bandwidth."""

    if particles.ndim != 2 or particles.shape[1] != DIMENSION:
        raise ValueError(
            f"particles must have shape (n_particles, {DIMENSION})"
        )

    if len(particles) == 0:
        raise ValueError("particles cannot be empty")

    # differences[i, j] contains x_i - x_j.
    differences = (
        particles[:, None, :]
        - particles[None, :, :]
    )

    # One squared distance for every pair of particles.
    squared_distances = differences.square().sum(dim=-1)

    # Same median heuristic used in the paper reproduction.
    median_squared_distance = torch.median(squared_distances)

    denominator = torch.log(
        particles.new_tensor(float(len(particles) + 1))
    )

    bandwidth = torch.clamp(
        median_squared_distance / denominator,
        min=1e-8,
    )

    kernel_matrix = torch.exp(
        -squared_distances / bandwidth
    )

    return kernel_matrix, bandwidth

def svgd_direction(
    particles: torch.Tensor,
    score_function: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Compute the SVGD direction for a batch of particles.

    Args:
        particles: Tensor of shape (n_particles, DIMENSION).
        score_function: Function that computes the score of the target
            distribution. It should accept a tensor of shape
            (n_particles, DIMENSION) and return a tensor of the same shape.

    Returns:
        Tensor of shape (n_particles, DIMENSION) representing the SVGD direction.
    """ 
        
    if particles.ndim != 2 or particles.shape[1] != DIMENSION:
        raise ValueError(
            f"particles must have shape (n_particles, {DIMENSION})"
        )

    if len(particles) == 0:
        raise ValueError("particles cannot be empty")

    # Compute the kernel matrix and its bandwidth.
    kernel_matrix, bandwidth = rbf_kernel(particles)

    # Compute the score of the current EBM at the particle locations.
    score_values = score_function(particles)

    if score_values.shape != particles.shape:
        raise ValueError(
            "score_function must return a tensor with the same "
            "shape as particles"
        )

    n_particles = particles.shape[0]

    # The attraction term transports particles towards lower-energy regions.
    attraction = kernel_matrix @ score_values

    # For the paper's RBF kernel, this is the summed kernel gradient
    # sum_j grad_{x_j} k(x_j, x_i), which repels nearby particles.
    kernel_row_sums = kernel_matrix.sum(dim=1, keepdim=True)
    repulsion = (
        2.0
        * (
            particles * kernel_row_sums
            - kernel_matrix @ particles
        )
        / bandwidth
    )

    direction = (attraction + repulsion) / n_particles

    return direction.detach()

#we create a function that moves the particles in the direction of the svgd direction
def svgd_run(
    initial_particles: torch.Tensor,
    n_steps: int,
    step_size: float,
    score_function: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Run SVGD to sample from a target distribution.

    Args:
        initial_particles: Initial particle positions with shape
            (n_particles, DIMENSION).
        n_steps: Number of SVGD updates.
        step_size: Step size used in each update.
        score_function: Function that computes the score of the target
            distribution. It should accept a tensor of shape
            (n_particles, DIMENSION) and return a tensor of the same shape. 

    Returns:
        Final particle positions with the same shape as initial_particles.          
    """ 
    
    particles = initial_particles.clone().detach()

    if particles.ndim != 2 or particles.shape[1] != DIMENSION:
        raise ValueError(
            f"initial_particles must have shape (n_particles, {DIMENSION})"
        )

    if len(particles) == 0:
        raise ValueError("initial_particles cannot be empty")

    if n_steps < 0:
        raise ValueError("n_steps must be non-negative")

    if step_size <= 0.0:
        raise ValueError("step_size must be positive")

    for _ in range(n_steps):
        direction = svgd_direction(particles, score_function)
        particles = particles + step_size * direction

    return particles.detach()


def main() -> None:
    """Check the complete neural-EBM SVGD sampler."""

    torch.manual_seed(7)

    model = NeuralEnergy()

    initial_particles = torch.linspace(
        -5.0,
        5.0,
        50,
    ).reshape(-1, DIMENSION)

    kernel_matrix, bandwidth = rbf_kernel(
        initial_particles
    )

    initial_direction = svgd_direction(
        initial_particles,
        model.model_score,
    )

    final_particles = svgd_run(
        initial_particles=initial_particles,
        n_steps=100,
        step_size=0.05,
        score_function=model.model_score,
    )

    print("Initial particles shape:", initial_particles.shape)
    print("Kernel matrix shape:", kernel_matrix.shape)
    print("Bandwidth:", bandwidth.item())
    print("SVGD direction shape:", initial_direction.shape)
    print("Final particles shape:", final_particles.shape)

    assert kernel_matrix.shape == (50, 50)
    assert initial_direction.shape == initial_particles.shape
    assert final_particles.shape == initial_particles.shape

    assert torch.all(torch.isfinite(kernel_matrix))
    assert torch.isfinite(bandwidth)
    assert torch.all(torch.isfinite(initial_direction))
    assert torch.all(torch.isfinite(final_particles))

    # An RBF kernel must be symmetric.
    assert torch.allclose(
        kernel_matrix,
        kernel_matrix.T,
        atol=1e-6,
    )

    # Every particle has kernel similarity one with itself.
    assert torch.allclose(
        torch.diag(kernel_matrix),
        torch.ones(50),
        atol=1e-6,
    )

    # At least one particle must have moved.
    assert not torch.allclose(
        initial_particles,
        final_particles,
    )

    print("\nAll neural-EBM SVGD checks passed.")


if __name__ == "__main__":
    main()

