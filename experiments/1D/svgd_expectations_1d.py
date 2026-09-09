"""Reproduce the expectation-estimation study in Figure 2 of the SVGD paper.

Liu and Wang compare estimates obtained from transported SVGD particles with
estimates from the same number of independent Monte Carlo samples. They study
h(x) = x, h(x) = x^2 and h(x) = cos(omega*x + b). This script implements that
comparison on the exact mixture and initialization used in their Figure 1.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gmm_1d import (
    EXPECTED_MEAN,
    EXPECTED_SECOND_MOMENT,
    MEANS,
    SEED,
    STANDARD_DEVIATION,
    WEIGHTS,
    sample_initial_particles,
    sample_target,
)
from svgd_gmm_1d import N_STEPS, STEP_SIZE, run_svgd


# These sizes cover the labelled range displayed in Figure 2 while keeping the
# reproduction quick enough to run during development and presentation.
SAMPLE_SIZES = np.array([10, 20, 50, 100, 250])

# The paper averages the random cosine experiment over 20 draws of omega and b.
# We also use 20 independent particle/sample draws to estimate every MSE.
N_TRIALS = 20

METRICS = ("x", "x_squared", "cosine")


def exact_cosine_expectation(omega: float, phase: float) -> float:
    """Compute E[cos(omega*x + phase)] under the Gaussian mixture."""
    gaussian_attenuation = np.exp(
        -0.5 * STANDARD_DEVIATION**2 * omega**2
    )
    component_expectations = gaussian_attenuation * np.cos(
        omega * MEANS + phase
    )
    return float(np.dot(WEIGHTS, component_expectations))


def squared_errors(
    samples: np.ndarray,
    omega: float,
    phase: float,
) -> dict[str, float]:
    """Return squared expectation errors for the paper's test functions."""
    exact_cosine = exact_cosine_expectation(omega, phase)

    return {
        "x": float((np.mean(samples) - EXPECTED_MEAN) ** 2),
        "x_squared": float(
            (np.mean(samples**2) - EXPECTED_SECOND_MOMENT) ** 2
        ),
        "cosine": float(
            (
                np.mean(np.cos(omega * samples + phase))
                - exact_cosine
            )
            ** 2
        ),
    }


def run_experiment() -> dict[str, dict[str, np.ndarray]]:
    """Estimate MSE curves for SVGD and exact Monte Carlo sampling."""
    errors = {
        method: {
            metric: np.zeros((len(SAMPLE_SIZES), N_TRIALS))
            for metric in METRICS
        }
        for method in ("Monte Carlo", "SVGD")
    }

    for size_index, sample_size in enumerate(SAMPLE_SIZES):
        for trial in range(N_TRIALS):
            # Independent and reproducible generators avoid accidentally giving
            # either method the same samples or correlated randomness.
            base_seed = SEED + 10_000 * size_index + 10 * trial
            initial_rng = np.random.default_rng(base_seed)
            monte_carlo_rng = np.random.default_rng(base_seed + 1)
            function_rng = np.random.default_rng(base_seed + 2)

            initial_particles = sample_initial_particles(
                int(sample_size),
                initial_rng,
            )
            svgd_particles, _ = run_svgd(
                initial_particles,
                n_steps=N_STEPS,
                step_size=STEP_SIZE,
            )
            monte_carlo_samples = sample_target(
                int(sample_size),
                monte_carlo_rng,
            )

            # This matches Figure 2(c): omega ~ N(0,1) and
            # b ~ Uniform([0, 2*pi]).
            omega = float(function_rng.normal())
            phase = float(function_rng.uniform(0.0, 2.0 * np.pi))

            method_samples = {
                "Monte Carlo": monte_carlo_samples,
                "SVGD": svgd_particles,
            }
            for method, samples in method_samples.items():
                trial_errors = squared_errors(samples, omega, phase)
                for metric in METRICS:
                    errors[method][metric][size_index, trial] = (
                        trial_errors[metric]
                    )

    return errors


def summarize_errors(
    errors: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, np.ndarray]]:
    """Average squared errors over the repeated trials."""
    return {
        method: {
            metric: np.mean(values, axis=1)
            for metric, values in method_errors.items()
        }
        for method, method_errors in errors.items()
    }


def save_results(
    summaries: dict[str, dict[str, np.ndarray]],
) -> tuple[Path, Path]:
    """Save the Figure 2 reproduction and its numerical values."""
    output_directory = Path(__file__).resolve().parent / "results"
    output_directory.mkdir(parents=True, exist_ok=True)

    csv_path = output_directory / "svgd_expectations_1d.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file, lineterminator="\n")
        writer.writerow(
            ["sample_size", "method", "x_mse", "x_squared_mse", "cosine_mse"]
        )
        for size_index, sample_size in enumerate(SAMPLE_SIZES):
            for method in ("Monte Carlo", "SVGD"):
                writer.writerow(
                    [
                        int(sample_size),
                        method,
                        summaries[method]["x"][size_index],
                        summaries[method]["x_squared"][size_index],
                        summaries[method]["cosine"][size_index],
                    ]
                )

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    plot_labels = {
        "x": r"Estimating $\mathbb{E}[x]$",
        "x_squared": r"Estimating $\mathbb{E}[x^2]$",
        "cosine": r"Estimating $\mathbb{E}[\cos(\omega x+b)]$",
    }
    styles = {
        "Monte Carlo": {"color": "#D62728", "marker": "s"},
        "SVGD": {"color": "#168A2E", "marker": "o"},
    }

    for axis, metric in zip(axes, METRICS):
        for method in ("Monte Carlo", "SVGD"):
            axis.plot(
                SAMPLE_SIZES,
                np.log10(summaries[method][metric]),
                linewidth=2.0,
                markersize=6,
                label=method,
                **styles[method],
            )
        axis.set_xscale("log")
        axis.set_xlabel("Sample size n")
        axis.set_ylabel(r"$\log_{10}$ MSE")
        axis.set_title(plot_labels[metric])
        axis.grid(alpha=0.2)

    axes[-1].legend()
    figure.suptitle(
        "Reproduction of Liu & Wang (2016), Figure 2: expectation estimates"
    )
    figure.tight_layout()

    figure_path = output_directory / "svgd_expectations_1d.png"
    figure.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    return figure_path, csv_path


def main() -> None:
    """Run, validate and report the Figure 2 reproduction."""
    errors = run_experiment()
    summaries = summarize_errors(errors)
    figure_path, csv_path = save_results(summaries)

    for method in ("Monte Carlo", "SVGD"):
        for metric in METRICS:
            assert np.all(np.isfinite(summaries[method][metric]))
            assert np.all(summaries[method][metric] >= 0.0)

    print("Mean squared error at the largest sample size:")
    for metric in METRICS:
        monte_carlo_mse = summaries["Monte Carlo"][metric][-1]
        svgd_mse = summaries["SVGD"][metric][-1]
        ratio = svgd_mse / monte_carlo_mse
        print(
            f"  {metric:10s}: MC={monte_carlo_mse:.6g}, "
            f"SVGD={svgd_mse:.6g}, ratio={ratio:.3f}"
        )

    print(f"\nFigure saved to: {figure_path}")
    print(f"Data saved to:   {csv_path}")
    print("Paper Figure 2 expectation study completed successfully.")


if __name__ == "__main__":
    main()
