"""One-dimensional Gaussian mixture from Liu and Wang (2016).

This module reproduces the target distribution used in Section 5, "Toy
Example on 1D Gaussian Mixture", of:

    Q. Liu and D. Wang, "Stein Variational Gradient Descent: A General
    Purpose Bayesian Inference Algorithm", NeurIPS 2016.
    https://arxiv.org/abs/1608.04471

The paper defines

    p(x) = 1/3 N(x; -2, 1) + 2/3 N(x; 2, 1)

and initializes the particles from q0(x) = N(x; -10, 1). Keeping these
quantities in one module ensures that every one-dimensional experiment uses
the same published target.
"""

import numpy as np


PAPER_URL = "https://arxiv.org/abs/1608.04471"

SEED = 7
MEANS = np.array([-2.0, 2.0])
WEIGHTS = np.array([1.0 / 3.0, 2.0 / 3.0])
STANDARD_DEVIATION = 1.0

# Initial distribution q0 from the paper's toy experiment. It has almost no
# overlap with the target, deliberately making the transport problem hard.
INITIAL_MEAN = -10.0
INITIAL_STANDARD_DEVIATION = 1.0

# Exact moments of the published mixture, used to validate the reproduction.
EXPECTED_MEAN = float(np.dot(WEIGHTS, MEANS))
EXPECTED_SECOND_MOMENT = float(
    np.dot(WEIGHTS, STANDARD_DEVIATION**2 + MEANS**2)
)


def component_densities(x: np.ndarray) -> np.ndarray:
    """Evaluate both Gaussian component densities.

    Args:
        x: Scalar or array of positions.

    Returns:
        An array with one final dimension of length two, one value for each
        Gaussian component.
    """
    x = np.asarray(x, dtype=float)
    standardized = (x[..., np.newaxis] - MEANS) / STANDARD_DEVIATION

    return (
        np.exp(-0.5 * standardized**2)
        / (np.sqrt(2.0 * np.pi) * STANDARD_DEVIATION)
    )


def target_density(x: np.ndarray) -> np.ndarray:
    """Evaluate the exact mixture density p(x) from the paper."""
    return component_densities(x) @ WEIGHTS


def target_energy(x: np.ndarray) -> np.ndarray:
    """Evaluate the exact energy -log p(x) of the published target."""
    return -np.log(target_density(x) + 1e-300)


def target_score(x: np.ndarray) -> np.ndarray:
    """Evaluate the score d log p(x) / dx required by SVGD.

    SVGD only needs the target score, not its normalizing constant. For a
    mixture, the score is the responsibility-weighted sum of the two Gaussian
    component scores.
    """
    x = np.asarray(x, dtype=float)
    weighted_densities = component_densities(x) * WEIGHTS
    mixture_density = np.sum(weighted_densities, axis=-1, keepdims=True)
    responsibilities = weighted_densities / mixture_density
    component_scores = (
        MEANS - x[..., np.newaxis]
    ) / STANDARD_DEVIATION**2

    return np.sum(responsibilities * component_scores, axis=-1)


def sample_target(
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw independent Monte Carlo samples from the exact mixture."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    components = rng.choice(len(MEANS), size=n_samples, p=WEIGHTS)
    return rng.normal(
        loc=MEANS[components],
        scale=STANDARD_DEVIATION,
    )


def sample_initial_particles(
    n_particles: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw particles from q0(x) = N(-10, 1), as specified in the paper."""
    if n_particles <= 0:
        raise ValueError("n_particles must be positive")

    return rng.normal(
        loc=INITIAL_MEAN,
        scale=INITIAL_STANDARD_DEVIATION,
        size=n_particles,
    )


def main() -> None:
    """Run deterministic checks of the published target distribution."""
    grid = np.linspace(-12.0, 8.0, 200_001)
    density_values = target_density(grid)

    integrated_mass = np.trapezoid(density_values, grid)
    numerical_mean = np.trapezoid(grid * density_values, grid)
    numerical_second_moment = np.trapezoid(
        grid**2 * density_values,
        grid,
    )

    test_points = np.array([-5.0, -2.0, 0.0, 2.0, 5.0])
    epsilon = 1e-5
    numerical_negative_energy_gradient = -(
        target_energy(test_points + epsilon)
        - target_energy(test_points - epsilon)
    ) / (2.0 * epsilon)

    assert np.isclose(integrated_mass, 1.0, atol=1e-8)
    assert np.isclose(numerical_mean, EXPECTED_MEAN, atol=1e-8)
    assert np.isclose(
        numerical_second_moment,
        EXPECTED_SECOND_MOMENT,
        atol=1e-7,
    )
    assert np.allclose(
        target_score(test_points),
        numerical_negative_energy_gradient,
        atol=1e-7,
    )

    print(f"Integrated target mass: {integrated_mass:.6f}")
    print(f"Exact target mean:       {EXPECTED_MEAN:.6f}")
    print(f"Exact second moment:     {EXPECTED_SECOND_MOMENT:.6f}")
    print("All paper-target checks passed.")


if __name__ == "__main__":
    main()
