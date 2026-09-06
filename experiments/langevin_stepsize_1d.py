"""Study the effect of the Langevin step size on a known 1D target."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from langevin_gmm_1d import NPARTICLES, SEED, density, langevin


# These values represent a small, medium, and large Langevin step.
STEPSIZES = (0.003, 0.03, 0.3)


def draw(axis, particles: np.ndarray, title: str, xgrid: np.ndarray) -> None:
    """Draw one particle histogram against the known target density."""
    axis.hist(particles, bins=22, density=True, alpha=0.55, color="#4C78A8")
    axis.plot(xgrid, density(xgrid), color="#E45756", linewidth=2.2)
    axis.set_title(title)
    axis.set_xlabel("x")
    axis.grid(alpha=0.2)


def main() -> None:
    # Use one fixed initial particle set in every condition.
    initialRng = np.random.default_rng(SEED)
    initial = initialRng.normal(loc=0.0, scale=3.0, size=NPARTICLES)

    # The first panel provides the common starting point for all three runs.
    xgrid = np.linspace(-8.0, 8.0, 1000)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    draw(axes.flat[0], initial, "Initial particles", xgrid)

    for index, stepsize in enumerate(STEPSIZES):
        # Restart the same noise sequence so only the step size changes.
        rng = np.random.default_rng(SEED + 1)
        final = langevin(initial.copy(), rng, stepsize)

        # Record both concentration in likely regions and mode balance.
        meanLogDensity = np.log(density(final) + 1e-12).mean()
        leftFraction = np.mean(final < 0.0)
        print(
            f"Step size {stepsize:g}: mean log density={meanLogDensity:.3f}, "
            f"left={leftFraction:.2f}, right={1.0 - leftFraction:.2f}"
        )

        draw(axes.flat[index + 1], final, f"Step size = {stepsize:g}", xgrid)

    axes[0, 0].set_ylabel("Density")
    axes[1, 0].set_ylabel("Density")
    fig.suptitle("Langevin step-size sensitivity on a 1D Gaussian mixture")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    # Save the complete comparison instead of relying on a temporary window.
    outputDir = Path(__file__).resolve().parent / "results"
    outputDir.mkdir(parents=True, exist_ok=True)
    figurePath = outputDir / "langevin_stepsize_1d.png"
    fig.savefig(figurePath, dpi=180, bbox_inches="tight")
    print(f"Figure saved to: {figurePath}")


if __name__ == "__main__":
    main()
