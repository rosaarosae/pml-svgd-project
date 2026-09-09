"""Paper-inspired two-dimensional Gaussian mixture.

Liu and Wang (2016) use the one-dimensional target

    p(x1) = 1/3 N(-2, 1) + 2/3 N(2, 1)

in their published SVGD toy experiment. The paper does not define a 2D version,
so this module makes the smallest possible extension: it preserves that exact
mixture along x1 and adds an independent x2 ~ N(0, 1). Consequently, the two
component means are (-2, 0) and (2, 0), with identity covariance.

Paper: https://arxiv.org/abs/1608.04471
"""

import numpy as np

SEED = 7
DIMENSION = 2

# Each row contains the mean of a Gaussian component. The first coordinate is
# exactly the target from the paper; the second is the added standard normal.
MEANS = np.array([[-2.0, 0.0], [2.0, 0.0]])

N_COMPONENTS = len(MEANS)

# Use the unequal component probabilities from the published 1D experiment.
WEIGHTS = np.array([1.0 / 3.0, 2.0 / 3.0])

# A general Gaussian can have any valid covariance matrix.
# Here we choose an isotropic covariance: sigma² times the identity matrix.
# This creates circular modes with equal dispersion in both directions.

STD = 1.0
COVARIANCE = STD**2 * np.eye(DIMENSION)
INVERSE_COVARIANCE = np.linalg.inv(COVARIANCE)
DETERMINANT_COVARIANCE = np.linalg.det(COVARIANCE)
NORMALIZATION_CONSTANT = 1.0 / (2 * np.pi * np.sqrt(DETERMINANT_COVARIANCE))

# Direct 2D extension of q0(x1) = N(-10, 1): x2 already follows N(0, 1).
INITIAL_MEAN = np.array([-10.0, 0.0])
INITIAL_STD = 1.0

# Shared limits keep the 2D figures consistent. Transport plots include q0;
# target-only plots can focus on the two modes.
TARGET_X_LIMITS = (-6.0, 6.0)
TRANSPORT_X_LIMITS = (-13.0, 6.0)
Y_LIMITS = (-4.0, 4.0)

#Once we have the parameters, we write the function

def component_densities(x: np.ndarray) -> np.ndarray:
    """Compute the density of each Gaussian component.

    Args:
        x: Array of shape (n_points, 2).

    Returns:
        Array of shape (n_points, n_components).
    """
    # Allow a single point with shape (2,) or several points with shape (N, 2).
    x = np.atleast_2d(x)

    densities = np.zeros((x.shape[0], N_COMPONENTS))

    for i in range(N_COMPONENTS):
        # Difference between every point and the current Gaussian mean.
        differences = x - MEANS[i]

        # Compute (x - mean)^T covariance^-1 (x - mean)
        # separately for every input point.
        squared_mahalanobis_distance = np.sum(
            (differences @ INVERSE_COVARIANCE) * differences,
            axis=1,
        )

        exponent = -0.5 * squared_mahalanobis_distance

        densities[:, i] = NORMALIZATION_CONSTANT * np.exp(exponent)

    return densities

def mixture_density(x: np.ndarray) -> np.ndarray:
    """Compute the density of the Gaussian mixture.

    Args:
        x: Array of shape (n_points, 2).

    Returns:
        Array of shape (n_points,).
    """
    # Allow a single point with shape (2,) or several points with shape (N, 2).
    x = np.atleast_2d(x)

    # Compute the density of each component.
    component_densities_array = component_densities(x)

    # Compute the weighted sum of the component densities.
    mixture_density_array = np.dot(component_densities_array, WEIGHTS)

    return mixture_density_array

def target_energy(x: np.ndarray) -> np.ndarray:
    """Compute the energy of the Gaussian mixture.

    Args:
        x: Array of shape (n_points, 2).

    Returns:
        Array of shape (n_points,).
    """
    # Allow a single point with shape (2,) or several points with shape (N, 2).
    x = np.atleast_2d(x)

    # Compute the density of the mixture.
    mixture_density_array = mixture_density(x)

    # Compute the energy as the negative log density.
    # The tiny constant prevents log(0) without changing ordinary tail values.
    energy_array = -np.log(mixture_density_array + 1e-300)

    return energy_array

def target_score(x: np.ndarray) -> np.ndarray:
    """Compute the score of the Gaussian mixture.

    Args:
        x: Array of shape (2,) or (n_points, 2).

    Returns:
        Array of shape (n_points, 2).
    """
    x = np.atleast_2d(x).astype(float)

    component_density_values = component_densities(x)
    mixture_density_values = mixture_density(x)

    score_values = np.zeros_like(x, dtype=float)

    for i in range(N_COMPONENTS):
        differences = x - MEANS[i]

        component_score = (
            -differences @ INVERSE_COVARIANCE
        )

        weighted_component_score = (
            WEIGHTS[i]
            * component_density_values[:, i, np.newaxis]
            * component_score
        )

        score_values += weighted_component_score

    score_values /= mixture_density_values[:, np.newaxis]

    return score_values

def numerical_gradient(func, x, epsilon=1e-5):
    """Compute the numerical gradient of a function at a given point.

    Args:
        x: Array of shape (2,) or (n_points, 2).
        step: Small displacement used for the numerical approximation.

    Returns:
        Array of shape (n_points, 2).
    """
    x = np.atleast_2d(x).astype(float)
    gradient = np.zeros_like(x)

    for dimension in range(DIMENSION):
        displacement = np.zeros_like(x)
        displacement[:, dimension] = epsilon

        function_plus = func(x + displacement)
        function_minus = func(x - displacement)

        gradient[:, dimension] = (
            function_plus - function_minus
        ) / (2.0 * epsilon)

    return gradient

def sample_target(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Draw independent samples from the target Gaussian mixture.

    Args:
        n_samples: Number of samples to generate.
        rng: NumPy random number generator.

    Returns:
        Array of shape (n_samples, 2).
    """
    component_indices = rng.choice(
        N_COMPONENTS,
        size=n_samples,
        p=WEIGHTS,
    )

    samples = rng.normal(
        loc=MEANS[component_indices],
        scale=STD,
        size=(n_samples, DIMENSION),
    )

    return samples


def sample_initial_particles(
    n_particles: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw from the paper-inspired initial distribution in two dimensions."""
    if n_particles <= 0:
        raise ValueError("n_particles must be positive")

    return rng.normal(
        loc=INITIAL_MEAN,
        scale=INITIAL_STD,
        size=(n_particles, DIMENSION),
    )

def main() -> None:
    test_points = np.array([
        [0.0, 0.0],
        [-2.0, 0.0],
        [2.0, 0.0],
        [1.5, 0.0],
        [2.0, 1.0],
    ])

    component_values = component_densities(test_points)
    mixture_values = mixture_density(test_points)
    energy_values = target_energy(test_points)
    score_values = target_score(test_points)

    energy_gradient_values = numerical_gradient(target_energy, test_points)
    negative_energy_gradient = -energy_gradient_values

    rng = np.random.default_rng(SEED)
    target_samples = sample_target(10_000, rng)

    distances_to_means = np.sum(
        (target_samples[:, np.newaxis, :] - MEANS[np.newaxis, :, :])**2,
        axis=2,
    )
    nearest_components = np.argmin(distances_to_means, axis=1)
    mode_fractions = np.bincount(
        nearest_components,
        minlength=N_COMPONENTS,
    ) / len(target_samples)

    print("Test points:")
    print(test_points)

    print("\nComponent densities:")
    print(component_values)

    print("\nMixture density:")
    print(mixture_values)

    print("\nTarget energy:")
    print(energy_values)

    print("\nTarget score:")
    print(score_values)

    print("\nNumerical negative energy gradient:")
    print(negative_energy_gradient)

    print("\nSample fraction assigned to each mode:")
    print(mode_fractions)

    # Check the returned shapes.
    assert component_values.shape == (5, N_COMPONENTS)
    assert mixture_values.shape == (5,)
    assert energy_values.shape == (5,)
    assert score_values.shape == (5, DIMENSION)
    assert energy_gradient_values.shape == (5, DIMENSION)
    assert target_samples.shape == (10_000, DIMENSION)

    # All computed values must be finite.
    assert np.all(np.isfinite(component_values))
    assert np.all(np.isfinite(mixture_values))
    assert np.all(np.isfinite(energy_values))
    assert np.all(np.isfinite(score_values))
    assert np.all(np.isfinite(energy_gradient_values))
    assert np.all(np.isfinite(target_samples))

    # Densities cannot be negative.
    assert np.all(component_values >= 0.0)
    assert np.all(mixture_values >= 0.0)

    # The selected Gaussian centres belong to the expected components.
    assert np.argmax(component_values[1]) == 0
    assert np.argmax(component_values[2]) == 1

    # A Gaussian reaches its maximum density at its centre.
    assert np.isclose(
        component_values[1, 0],
        NORMALIZATION_CONSTANT,
    )

    # The right mode has greater density because its paper weight is 2/3.
    assert mixture_values[2] > mixture_values[1]
    assert energy_values[2] < energy_values[1]

    # The origin has lower density and higher energy than a mode centre.
    assert mixture_values[0] < mixture_values[1]
    assert energy_values[0] > energy_values[1]

    # At (1.5, 0), the score points mainly towards the right-hand mode.
    assert score_values[3, 0] > 0.0
    assert np.isclose(score_values[3, 1], 0.0)

    # The added coordinate is standard normal, so its score at x2=1 is -1.
    assert np.isclose(score_values[4, 1], -1.0)

    # The analytic score must equal the negative energy gradient.
    assert np.allclose(
        score_values,
        negative_energy_gradient,
        atol=1e-5,
    )

    # Direct target samples should be approximately balanced across the modes.
    assert np.allclose(
        mode_fractions,
        WEIGHTS,
        atol=0.02,
    )

    print(
        "\nAll density, target-energy, score, "
        "score-energy relation, and sampling checks passed."
    )

if __name__ == "__main__":
    main()
