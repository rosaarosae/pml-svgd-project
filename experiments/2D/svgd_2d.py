"""Stein Variational Gradient Descent in two dimensions."""

from collections.abc import Callable
import numpy as np

from gmm_2d import DIMENSION, MEANS, SEED, target_energy, target_score


N_PARTICLES = 500
N_STEPS = 1_000

# The five-seed experiment selects 1.0: its mean target energy is almost equal
# to the true target value, with low variability and stable particle updates.
STEP_SIZE = 1.0

def svgd_dynamics(
    initial_particles: np.ndarray,
    n_steps: int,
    step_size: float,
    score_function: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Run Stein Variational Gradient Descent to sample from a target distribution.

    Args:
        initial_particles: Initial particle positions with shape
            (n_particles, DIMENSION).
        n_steps: Number of SVGD updates.
        step_size: Step size used in each update.
        score_function: Function that accepts particle positions and returns
            their target scores with the same shape.    

    Returns:
        Final particle positions with the same shape as initial_particles.
    """ 
    particles= np.asarray(initial_particles, dtype=float).copy()
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
    
    #now we calculate the distance matrix between the particles
    n_particles = particles.shape[0]
    for _ in range(n_steps):    
        score_values = score_function(particles)
        # Compute all squared distances through matrix multiplication. This is
        # equivalent to ||x_i - x_j||^2 but avoids a large 3D array.
        squared_norms = np.sum(particles**2, axis=1)
        squared_distances = (
            squared_norms[:, np.newaxis]
            + squared_norms[np.newaxis, :]
            - 2.0 * particles @ particles.T
        )
        squared_distances = np.maximum(squared_distances, 0.0)

        #now we calculate the kernel matrix using the RBF kernel
        nonzero_squared_distances = squared_distances[
            squared_distances > 1e-12
        ]

        if nonzero_squared_distances.size == 0:
            median_squared_distance = 1.0
        else:
            median_squared_distance = np.median(nonzero_squared_distances)

        bandwidth = max(
            median_squared_distance / np.log(n_particles + 1.0),
            1e-8,
        )
        kernel_matrix = np.exp(-squared_distances / bandwidth)

        #now we calculate the atraction through the kernel matrix and the score values
        attraction = kernel_matrix @ score_values / n_particles

        # Compute the RBF repulsion without constructing its (N, N, 2)
        # gradient tensor. This is algebraically the same pairwise sum.
        kernel_row_sums = kernel_matrix.sum(axis=1, keepdims=True)
        repulsion = (
            2.0
            * (particles * kernel_row_sums - kernel_matrix @ particles)
            / bandwidth
            / n_particles
        )
        #now we update the particles using the attraction and repulsion
        particles += step_size * (attraction + repulsion)

    return particles


def main() -> None:
    # Create the initial particles from a broad Gaussian distribution.
    initial_rng = np.random.default_rng(SEED)
    initial_particles = initial_rng.normal(
        loc=0.0,
        scale=3.0,
        size=(N_PARTICLES, DIMENSION),
    )

    # Run SVGD using the exact score of the Gaussian mixture.
    final_particles = svgd_dynamics(
        initial_particles=initial_particles,
        n_steps=N_STEPS,
        step_size=STEP_SIZE,
        score_function=target_score,
    )

    # Compare the target energy before and after moving the particles.
    initial_mean_energy = np.mean(target_energy(initial_particles))
    final_mean_energy = np.mean(target_energy(final_particles))

    # Assign every final particle to its nearest Gaussian mode.
    differences = (
        final_particles[:, np.newaxis, :]
        - MEANS[np.newaxis, :, :]
    )
    squared_distances = np.sum(differences**2, axis=2)
    nearest_modes = np.argmin(squared_distances, axis=1)
    mode_counts = np.bincount(nearest_modes, minlength=len(MEANS))
    mode_fractions = mode_counts / N_PARTICLES

    print("Initial particles shape:", initial_particles.shape)
    print("Final particles shape:", final_particles.shape)
    print(f"\nInitial mean target energy: {initial_mean_energy:.3f}")
    print(f"Final mean target energy:   {final_mean_energy:.3f}")
    print("\nFinal mode fractions:")
    print(mode_fractions)

    # Check shapes and numerical stability.
    assert initial_particles.shape == (N_PARTICLES, DIMENSION)
    assert final_particles.shape == (N_PARTICLES, DIMENSION)
    assert np.all(np.isfinite(final_particles))
    assert np.isfinite(initial_mean_energy)
    assert np.isfinite(final_mean_energy)

    # SVGD should move particles towards lower target energy.
    assert final_mean_energy < initial_mean_energy

    # Fractions must sum to one, and every target mode must be covered.
    assert np.isclose(mode_fractions.sum(), 1.0)
    assert np.all(mode_counts > 0)

    print("\nBasic SVGD checks passed.")


if __name__ == "__main__":
    main()
