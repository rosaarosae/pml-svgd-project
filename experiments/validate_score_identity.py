"""Validate the score identity used by Langevin dynamics and SVGD."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


WEIGHTS = np.array([1.0 / 3.0, 2.0 / 3.0])
MEANS = np.array([-2.0, 2.0])
STD = 1.0


def component_densities(x: np.ndarray) -> np.ndarray:
    """Return the weighted Gaussian component densities at every input."""
    z = (x[..., None] - MEANS) / STD
    normal = np.exp(-0.5 * z**2) / (np.sqrt(2.0 * np.pi) * STD)
    return normal * WEIGHTS


def target_density(x: np.ndarray) -> np.ndarray:
    return component_densities(x).sum(axis=-1)


def energy(x: np.ndarray) -> np.ndarray:
    return -np.log(target_density(x))


def analytic_score(x: np.ndarray) -> np.ndarray:
    components = component_densities(x)
    responsibilities = components / components.sum(axis=-1, keepdims=True)
    component_scores = (MEANS - x[..., None]) / STD**2
    return (responsibilities * component_scores).sum(axis=-1)


def numerical_score(x: np.ndarray, delta: float = 1e-5) -> np.ndarray:
    """Approximate -dE/dx with a centred finite difference."""
    return -(energy(x + delta) - energy(x - delta)) / (2.0 * delta)


def main() -> None:
    x = np.linspace(-6.0, 6.0, 1_200)
    score = analytic_score(x)
    score_from_energy = numerical_score(x)
    maximum_error = np.max(np.abs(score - score_from_energy))

    if maximum_error >= 1e-7:
        raise AssertionError(f"Score identity error is too large: {maximum_error}")

    figure, axes = plt.subplots(3, 1, figsize=(8.0, 7.0), sharex=True)
    axes[0].plot(x, target_density(x), color="#E45756")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Gaussian mixture: density, energy, and score")

    axes[1].plot(x, energy(x), color="#4C78A8")
    axes[1].set_ylabel("Energy")

    axes[2].plot(x, score, color="#54A24B", label="Analytic score")
    axes[2].plot(
        x,
        score_from_energy,
        color="#B279A2",
        linestyle="--",
        label=r"Numerical $-dE/dx$",
    )
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("Score")
    axes[2].legend()

    for axis in axes:
        axis.grid(alpha=0.2)

    figure.tight_layout()
    output_path = Path(__file__).resolve().parent / "results" / "score_identity.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")

    print(f"Maximum |score + dE/dx| error: {maximum_error:.3e}")
    print(f"Figure saved to: {output_path}")


if __name__ == "__main__":
    main()
