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

def main() -> None:
    test_points = np.array([
        [0.0, 0.0],
        [-2.0, -2.0],
        [2.0, 2.0],
    ])

    densities = component_densities(test_points)

    print("Test points:")
    print(test_points)

    print("\nComponent densities:")
    print(densities)

    print("\nShape:", densities.shape)

    # There are three test points and four Gaussian components.
    assert densities.shape == (3, 4)

    # All densities must be finite and non-negative.
    assert np.all(np.isfinite(densities))
    assert np.all(densities >= 0.0)

    # The point (-2, -2) is the centre of component 0.
    assert np.argmax(densities[1]) == 0

    # The point (2, 2) is the centre of component 3.
    assert np.argmax(densities[2]) == 3

    # Density at the exact centre must equal the normalization constant.
    assert np.isclose(
        densities[1, 0],
        NORMALIZATION_CONSTANT,
    )

    print("\nAll component-density checks passed.")


if __name__ == "__main__":
    main()