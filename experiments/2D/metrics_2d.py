"""Shared metrics for evaluating two-dimensional particle approximations."""

from dataclasses import dataclass

import numpy as np

from gmm_2d import (
    DIMENSION,
    MEANS,
    STD,
    WEIGHTS,
    mixture_density,
    target_energy,
)


@dataclass(frozen=True)
class SampleMetrics:
    """Metrics describing how well particles approximate the target GMM."""

    mean_target_energy: float
    energy_error: float
    mode_coverage: int
    mode_weight_error: float
    mode_center_error: float
    within_mode_variance_error: float
    sliced_wasserstein: float


def assign_to_modes(samples: np.ndarray) -> np.ndarray:
    """Assign every sample to the nearest target mean."""
    samples = _validate_samples(samples, "samples")
    differences = samples[:, np.newaxis, :] - MEANS[np.newaxis, :, :]
    squared_distances = np.sum(differences**2, axis=2)
    return np.argmin(squared_distances, axis=1)


def mode_fractions(samples: np.ndarray) -> np.ndarray:
    """Return the empirical fraction assigned to every target mode."""
    assignments = assign_to_modes(samples)
    counts = np.bincount(assignments, minlength=len(MEANS))
    return counts / len(assignments)


def mode_weight_error(samples: np.ndarray) -> float:
    """Return the total absolute error from the true mixture weights."""
    return float(np.abs(mode_fractions(samples) - WEIGHTS).sum())


def mode_coverage(samples: np.ndarray, minimum_fraction: float = 0.01) -> int:
    """Count modes containing at least a minimum fraction of all samples."""
    if not 0.0 <= minimum_fraction <= 1.0:
        raise ValueError("minimum_fraction must be between 0 and 1")
    return int(np.sum(mode_fractions(samples) >= minimum_fraction))


def mode_center_error(samples: np.ndarray) -> float:
    """Measure the average distance between empirical and true mode centres."""
    samples = _validate_samples(samples, "samples")
    assignments = assign_to_modes(samples)
    errors = []

    for mode_index, target_mean in enumerate(MEANS):
        mode_samples = samples[assignments == mode_index]
        if len(mode_samples) == 0:
            return float("inf")
        empirical_mean = np.mean(mode_samples, axis=0)
        errors.append(np.linalg.norm(empirical_mean - target_mean))

    return float(np.mean(errors))


def within_mode_variance_error(samples: np.ndarray) -> float:
    """Measure error in average per-coordinate variance inside target modes."""
    samples = _validate_samples(samples, "samples")
    assignments = assign_to_modes(samples)
    errors = []

    for mode_index, target_mean in enumerate(MEANS):
        mode_samples = samples[assignments == mode_index]
        if len(mode_samples) == 0:
            return float("inf")

        centered_samples = mode_samples - target_mean
        empirical_variance = np.mean(centered_samples**2)
        errors.append(abs(empirical_variance - STD**2))

    return float(np.mean(errors))


def sliced_wasserstein_distance(
    samples: np.ndarray,
    reference_samples: np.ndarray,
    n_projections: int = 128,
) -> float:
    """Approximate 2-Wasserstein distance using evenly spaced projections."""
    samples = _validate_samples(samples, "samples")
    reference_samples = _validate_samples(reference_samples, "reference_samples")

    if len(samples) != len(reference_samples):
        raise ValueError("samples and reference_samples must have equal length")
    if n_projections <= 0:
        raise ValueError("n_projections must be positive")

    # Directions in [0, pi) cover every distinct line through two-dimensional
    # space. Sorting projected values gives the one-dimensional optimal matching.
    angles = np.linspace(0.0, np.pi, n_projections, endpoint=False)
    directions = np.column_stack([np.cos(angles), np.sin(angles)])
    sample_projections = np.sort(samples @ directions.T, axis=0)
    reference_projections = np.sort(reference_samples @ directions.T, axis=0)

    mean_squared_difference = np.mean(
        (sample_projections - reference_projections) ** 2
    )
    return float(np.sqrt(mean_squared_difference))


def evaluate_samples(
    samples: np.ndarray,
    reference_samples: np.ndarray,
    expected_target_energy: float,
) -> SampleMetrics:
    """Compute all shared quality metrics for one particle approximation."""
    samples = _validate_samples(samples, "samples")
    reference_samples = _validate_samples(reference_samples, "reference_samples")
    mean_energy = float(np.mean(target_energy(samples)))

    return SampleMetrics(
        mean_target_energy=mean_energy,
        energy_error=abs(mean_energy - expected_target_energy),
        mode_coverage=mode_coverage(samples),
        mode_weight_error=mode_weight_error(samples),
        mode_center_error=mode_center_error(samples),
        within_mode_variance_error=within_mode_variance_error(samples),
        sliced_wasserstein=sliced_wasserstein_distance(
            samples,
            reference_samples,
        ),
    )


def numerical_expected_target_energy(
    grid_limit: float = 6.0,
    points_per_axis: int = 601,
) -> float:
    """Numerically integrate E_p[E_target(X)] on a regular 2D grid."""
    if grid_limit <= 0.0:
        raise ValueError("grid_limit must be positive")
    if points_per_axis < 2:
        raise ValueError("points_per_axis must be at least 2")

    coordinates = np.linspace(-grid_limit, grid_limit, points_per_axis)
    grid_x, grid_y = np.meshgrid(coordinates, coordinates)
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    density_values = mixture_density(grid_points)
    energy_values = target_energy(grid_points)
    grid_spacing = coordinates[1] - coordinates[0]

    # Dividing by the numerical integral makes the result robust to the tiny
    # amount of target mass outside the finite grid.
    numerical_mass = np.sum(density_values) * grid_spacing**2
    expected_energy = (
        np.sum(density_values * energy_values)
        * grid_spacing**2
        / numerical_mass
    )
    return float(expected_energy)


def _validate_samples(samples: np.ndarray, name: str) -> np.ndarray:
    """Return a validated floating-point sample matrix."""
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != DIMENSION:
        raise ValueError(f"{name} must have shape (n_samples, {DIMENSION})")
    if len(samples) == 0:
        raise ValueError(f"{name} cannot be empty")
    if not np.all(np.isfinite(samples)):
        raise ValueError(f"{name} must contain only finite values")
    return samples


def main() -> None:
    """Run small deterministic checks for the shared metrics."""
    repeated_means = np.repeat(MEANS, repeats=10, axis=0)
    fractions = mode_fractions(repeated_means)

    assert np.allclose(fractions, WEIGHTS)
    assert mode_coverage(repeated_means) == len(MEANS)
    assert np.isclose(mode_weight_error(repeated_means), 0.0)
    assert np.isclose(mode_center_error(repeated_means), 0.0)
    assert np.isclose(
        sliced_wasserstein_distance(repeated_means, repeated_means),
        0.0,
    )
    expected_energy = numerical_expected_target_energy()
    assert 2.8 < expected_energy < 2.9

    print(f"Numerical expected target energy: {expected_energy:.6f}")
    print("All shared-metric checks passed.")


if __name__ == "__main__":
    main()
