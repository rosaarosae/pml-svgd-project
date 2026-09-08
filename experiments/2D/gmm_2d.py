"""Multimodal two-dimensional isotropic Gaussian mixture.

This module implements a two-dimensional isotropic Gaussian mixture model (GMM) and
is used to validate Langevin dynamics and SVGD before training the neural energy-based
 model.The mixture has four equally weighted Gaussian components.
"""

import numpy as np

SEED = 7
DIMENSION = 2

#Each row contains the mean of a Gaussian component
MEANS = np.array([[-2, -2], [-2, 2], [2, -2], [2, 2]])

N_COMPONENTS = len(MEANS)

#all components have the same probability 
WEIGHTS = np.full(N_COMPONENTS, 1.0 / N_COMPONENTS)

# A general Gaussian can have any valid covariance matrix.
# Here we choose an isotropic covariance: sigma² times the identity matrix.
# This creates circular modes with equal dispersion in both directions.

STD = 0.5
COVARIANCE = STD**2 * np.eye(DIMENSION)
INVERSE_COVARIANCE = np.linalg.inv(COVARIANCE)
DETERMINANT_COVARIANCE = np.linalg.det(COVARIANCE)
NORMALIZATION_CONSTANT = 1.0 / (2 * np.pi * np.sqrt(DETERMINANT_COVARIANCE))

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
    energy_array = -np.log(mixture_density_array + 1e-12)  # Add a small constant to avoid log(0).

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

def main() -> None:
    test_points = np.array([
        [0.0, 0.0],
        [-2.0, -2.0],
        [2.0, 2.0],
        [1.5, 2.0],
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
    assert component_values.shape == (4, N_COMPONENTS)
    assert mixture_values.shape == (4,)
    assert energy_values.shape == (4,)
    assert score_values.shape == (4, DIMENSION)
    assert energy_gradient_values.shape == (4, DIMENSION)
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
    assert np.argmax(component_values[2]) == 3

    # A Gaussian reaches its maximum density at its centre.
    assert np.isclose(
        component_values[1, 0],
        NORMALIZATION_CONSTANT,
    )

    # Symmetric mode centres have equal density and energy.
    assert np.isclose(
        mixture_values[1],
        mixture_values[2],
    )

    assert np.isclose(
        energy_values[1],
        energy_values[2],
    )

    # The origin has lower density and higher energy than a mode centre.
    assert mixture_values[0] < mixture_values[1]
    assert energy_values[0] > energy_values[1]

    # The score is zero at the origin because the four forces cancel.
    assert np.allclose(
        score_values[0],
        [0.0, 0.0],
        atol=1e-10,
    )

    # The score is approximately zero at the Gaussian centres.
    assert np.allclose(
        score_values[1],
        [0.0, 0.0],
        atol=1e-10,
    )

    assert np.allclose(
        score_values[2],
        [0.0, 0.0],
        atol=1e-10,
    )

    # At (1.5, 2.0), the score points towards the mode at (2.0, 2.0).
    assert np.allclose(
        score_values[3],
        [2.0, 0.0],
        atol=1e-8,
    )

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
