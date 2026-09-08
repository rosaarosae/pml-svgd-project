"""Langevin dynamics in two dimensions."""

from collections.abc import Callable
import numpy as np

from gmm_2d import DIMENSION, SEED, target_score

#now we add the initial constants
N_PARTICLES = 500
N_STEPS = 1_000
# The multi-seed experiment selects 0.01 as a stable compromise: it gives a
# near-target mean energy, low mode-weight error, and avoids excessive spread.
STEP_SIZE = 0.01

#we define the langevin dynamics function in 2D
def langevin_dynamics(
    initial_particles: np.ndarray,
    n_steps: int,
    step_size: float,
    score_function: Callable[[np.ndarray], np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """Run Langevin dynamics to sample from a target distribution.

    Args:
        initial_particles: Initial particle positions with shape
            (n_particles, DIMENSION).
        n_steps: Number of Langevin updates.
        step_size: Step size used in each update.
        score_function: Function that accepts particle positions and returns
            their target scores with the same shape.
        rng: Random number generator used for the Langevin noise.

    Returns:
        Final particle positions with the same shape as initial_particles.
    """
    particles = np.asarray(
        initial_particles,
        dtype=float,
    ).copy()

    if particles.ndim != 2 or particles.shape[1] != DIMENSION:
        raise ValueError(
            f"initial_particles must have shape (n_particles, {DIMENSION})"
        )

    if n_steps < 0:
        raise ValueError("n_steps must be non-negative")

    if step_size <= 0.0:
        raise ValueError("step_size must be positive")

    for _ in range(n_steps):
        score_values = score_function(particles)

        drift = step_size * score_values

        noise = np.sqrt(2.0 * step_size) * rng.normal(
            size=particles.shape,
        )

        particles += drift + noise

    return particles

def main() -> None:
    # Generator used to create the initial particles.
    initial_rng = np.random.default_rng(SEED)

    # Start from a broad Gaussian distribution.
    initial_particles = initial_rng.normal(
        loc=0.0,
        scale=3.0,
        size=(N_PARTICLES, DIMENSION),
    )

    # Use a separate generator for the Langevin noise.
    noise_rng = np.random.default_rng(SEED + 1)

    # Run Langevin using the exact score of the target GMM.
    final_particles = langevin_dynamics(
        initial_particles=initial_particles,
        n_steps=N_STEPS,
        step_size=STEP_SIZE,
        score_function=target_score,
        rng=noise_rng,
    )

    print("Initial particles shape:", initial_particles.shape)
    print("Final particles shape:", final_particles.shape)

    assert initial_particles.shape == (N_PARTICLES, DIMENSION)
    assert final_particles.shape == (N_PARTICLES, DIMENSION)
    assert np.all(np.isfinite(final_particles))

    print("\nBasic Langevin checks passed.")


if __name__ == "__main__":
    main()
