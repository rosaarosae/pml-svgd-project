"""Compare SVGD and Langevin for training a one-dimensional neural EBM.

This experiment trains the same neural energy model with SVGD and Langevin
negative particles under matched conditions. Both methods use the same model
architecture, data distribution, optimizer, training schedule, number of
particles, and random seeds. Only the negative-particle sampler and its
selected step size differ.

"""
import argparse
import csv
from pathlib import Path
from time import perf_counter
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from energy_model import DIMENSION, NeuralEnergy
from gmm_1d import (
    EXPECTED_MEAN,
    EXPECTED_SECOND_MOMENT,
    sample_target,
    target_density,
    target_energy,
)
from langevin_ebm_1d import langevin_run
from svgd_ebm_1d import svgd_run

# SEEDS = [7]
SEEDS = [7, 17, 27, 37, 47, 57, 67, 77, 87, 97]

N_EPOCHS = 1000
BATCH_SIZE = 200
N_PARTICLES = 500
LEARNING_RATE = 1e-3
MIN_LEARNING_RATE = 1e-4
SAMPLER_STEPS = 20
INITIAL_PARTICLE_MEAN = 0.0
INITIAL_PARTICLE_STD = 3.0

# Selected using the separate validation seeds in tune_ebm_samplers_1d.py.
SVGD_STEP_SIZE = 0.02
LANGEVIN_STEP_SIZE = 0.05

GRID_MIN = -8.0
GRID_MAX = 8.0
N_GRID_POINTS = 1000
N_TEST_SAMPLES = 10_000


class RunResult(TypedDict):
    """Metrics produced by one training run."""

    method: str
    seed: int
    density_error: float
    left_mass: float
    right_mass: float
    learned_mean: float
    learned_second_moment: float
    test_nll: float
    runtime_seconds: float
    particle_mean: float
    particle_std: float
    stable: bool


class CurveResult(TypedDict):
    """Normalized density and energy curves from one trained model."""

    method: str
    seed: int
    density: np.ndarray
    energy: np.ndarray


def train_one_run(
    method: str,
    seed: int,
    step_size: float,
    curve_results: list[CurveResult] | None = None,
) -> RunResult:
    """Train one neural EBM and return its evaluation metrics."""

    # Use the same seed for model initialization and random samples.
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # Create the shared neural energy model and optimizer.
    model = NeuralEnergy()
    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )
    learning_rate_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=N_EPOCHS,
        eta_min=MIN_LEARNING_RATE,
    )

    # Select the negative-particle sampler.
    if method == "SVGD":
        sampler_function = svgd_run
    elif method == "Langevin":
        sampler_function = langevin_run
    else:
        raise ValueError(
            f"Unknown method: {method}"
        )

    # Start both samplers from the same broad, neutral distribution.
    initial_samples = rng.normal(
        loc=INITIAL_PARTICLE_MEAN,
        scale=INITIAL_PARTICLE_STD,
        size=(N_PARTICLES, DIMENSION),
    )
    negative_particles = torch.tensor(
        initial_samples,
        dtype=torch.float32,
    ).reshape(-1, DIMENSION)

    stable = True
    start_time = perf_counter()

    for _ in range(N_EPOCHS):
        # Draw a fresh positive batch in every epoch.
        positive_samples_numpy = sample_target(
            BATCH_SIZE,
            rng,
        )
        positive_samples = torch.tensor(
            positive_samples_numpy,
            dtype=torch.float32,
        ).reshape(-1, DIMENSION)

        # Update the persistent negative particles.
        negative_particles = sampler_function(
            initial_particles=negative_particles,
            n_steps=SAMPLER_STEPS,
            step_size=step_size,
            score_function=model.model_score,
        )

        # Calculate the shared contrastive EBM loss.
        positive_energy = model(
            positive_samples
        ).mean()
        negative_energy = model(
            negative_particles
        ).mean()
        loss = positive_energy - negative_energy

        # Record instability without stopping the complete comparison.
        if (
            not torch.isfinite(loss)
            or not torch.all(torch.isfinite(negative_particles))
        ):
            stable = False
            break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        learning_rate_scheduler.step()

    runtime_seconds = perf_counter() - start_time

    # Evaluate and normalize the learned density on a fixed grid.
    grid_values = torch.linspace(
        GRID_MIN,
        GRID_MAX,
        N_GRID_POINTS,
    )
    grid_points = grid_values.reshape(-1, DIMENSION)

    with torch.no_grad():
        grid_energies = model(grid_points)

    shifted_energies = grid_energies - grid_energies.min()
    unnormalized_density = torch.exp(-shifted_energies)
    normalization_constant = torch.trapezoid(
        y=unnormalized_density,
        x=grid_values,
    )
    learned_density = (
        unnormalized_density
        / normalization_constant
    )

    # Evaluate the exact target density on the same grid.
    exact_density_numpy = target_density(
        grid_values.numpy()
    )
    exact_density = torch.as_tensor(
        exact_density_numpy,
        dtype=learned_density.dtype,
    )

    # Check both densities before calculating their difference.
    assert learned_density.shape == exact_density.shape
    assert torch.all(torch.isfinite(learned_density))
    assert torch.all(torch.isfinite(exact_density))
    assert torch.isfinite(normalization_constant)
    assert normalization_constant > 0.0

    density_integral = torch.trapezoid(
        y=learned_density,
        x=grid_values,
    )
    assert torch.isclose(
        density_integral,
        torch.ones_like(density_integral),
        atol=1e-4,
    )

    # Calculate the squared error between both densities.
    squared_difference = (
        learned_density - exact_density
    ).square()

    density_error = torch.trapezoid(
        y=squared_difference,
        x=grid_values,
    ).item()

    assert np.isfinite(density_error)

    # Identify the grid points on each side of zero.
    left_indicator = (grid_values < 0.0).to(
        learned_density.dtype
    )

    right_indicator = 1.0 - left_indicator

    # Calculate the learned density mass on each side of zero.
    left_mass = torch.trapezoid(
        y=learned_density * left_indicator,
        x=grid_values,
    ).item()
    right_mass = torch.trapezoid(
        y=learned_density * right_indicator,
        x=grid_values,
    ).item()

    # Check that both masses are finite and sum to one.
    assert np.isfinite(left_mass)
    assert np.isfinite(right_mass)
    assert 0.0 <= left_mass <= 1.0
    assert 0.0 <= right_mass <= 1.0
    assert np.isclose(
        left_mass + right_mass,
        1.0,
        atol=1e-4,
    )

    # Calculate the learned mean and second moment.
    learned_mean = torch.trapezoid(
        y=grid_values * learned_density,
        x=grid_values,
    ).item()
    learned_second_moment = torch.trapezoid(
        y=grid_values.square() * learned_density,
        x=grid_values,
    ).item()

    assert np.isfinite(learned_mean)
    assert np.isfinite(learned_second_moment)

    # Draw an independent test sample for evaluating the trained model.
    test_rng = np.random.default_rng(seed + 10_000)
    test_samples_numpy = sample_target(
        N_TEST_SAMPLES,
        test_rng,
    )
    test_samples = torch.tensor(
        test_samples_numpy,
        dtype=torch.float32,
    ).reshape(-1, DIMENSION)

    # Calculate the negative log-likelihood on the test samples.
    with torch.no_grad():
        test_energies = model(test_samples)

    energy_shift = grid_energies.min()
    shifted_test_energies = test_energies - energy_shift
    test_nll = (
        shifted_test_energies.mean()
        + torch.log(normalization_constant)
    ).item()

    assert np.isfinite(test_nll)

    # Store normalized curves when requested by the final comparison. Energy
    # functions are identifiable only up to an additive constant, so using
    # -log of the normalized density puts every run on the same scale.
    if curve_results is not None:
        learned_density_numpy = learned_density.detach().cpu().numpy()
        curve_results.append(
            {
                "method": method,
                "seed": seed,
                "density": learned_density_numpy,
                "energy": -np.log(
                    np.clip(learned_density_numpy, 1e-12, None)
                ),
            }
        )

    # Summarize the final negative-particle distribution.
    particle_mean = negative_particles.mean().item()
    particle_std = negative_particles.std().item()

    assert np.isfinite(particle_mean)
    assert np.isfinite(particle_std)

    # Return all the metrics produced by this training run.
    return {
        "method": method,
        "seed": seed,
        "density_error": density_error,
        "left_mass": left_mass,
        "right_mass": right_mass,
        "learned_mean": learned_mean,
        "learned_second_moment": learned_second_moment,
        "test_nll": test_nll,
        "runtime_seconds": runtime_seconds,
        "particle_mean": particle_mean,
        "particle_std": particle_std,
        "stable": stable,
    }

def run_comparison() -> tuple[list[RunResult], list[CurveResult]]:
    """Run SVGD and Langevin under the matched experimental conditions."""

    results: list[RunResult] = []
    curve_results: list[CurveResult] = []

    configurations = [
        ("SVGD", SVGD_STEP_SIZE),
        ("Langevin", LANGEVIN_STEP_SIZE),
    ]

    for seed in SEEDS:
        for method, step_size in configurations:
            print(
                f"Training {method} with seed {seed}..."
            )

            result = train_one_run(
                method=method,
                seed=seed,
                step_size=step_size,
                curve_results=curve_results,
            )
            results.append(result)

            print(
                f"density_error={result['density_error']:.6f}, "
                f"test_nll={result['test_nll']:.6f}, "
                f"time={result['runtime_seconds']:.2f}s"
            )

    return results, curve_results


def save_results(results: list[RunResult], output_path: Path) -> None:
    """Save one row for every method and random seed."""
    fieldnames = list(RunResult.__annotations__)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def load_results(input_path: Path) -> list[RunResult]:
    """Load previously computed results so plots need no retraining."""
    results: list[RunResult] = []

    with input_path.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            results.append(
                {
                    "method": row["method"],
                    "seed": int(row["seed"]),
                    "density_error": float(row["density_error"]),
                    "left_mass": float(row["left_mass"]),
                    "right_mass": float(row["right_mass"]),
                    "learned_mean": float(row["learned_mean"]),
                    "learned_second_moment": float(
                        row["learned_second_moment"]
                    ),
                    "test_nll": float(row["test_nll"]),
                    "runtime_seconds": float(row["runtime_seconds"]),
                    "particle_mean": float(row["particle_mean"]),
                    "particle_std": float(row["particle_std"]),
                    "stable": row["stable"].lower() == "true",
                }
            )

    return results


def save_curves(
    curve_results: list[CurveResult],
    output_path: Path,
) -> None:
    """Save per-seed curves separately from the scalar CSV metrics."""
    np.savez_compressed(
        output_path,
        methods=np.asarray([curve["method"] for curve in curve_results]),
        seeds=np.asarray([curve["seed"] for curve in curve_results]),
        densities=np.stack([curve["density"] for curve in curve_results]),
        energies=np.stack([curve["energy"] for curve in curve_results]),
    )


def load_curves(input_path: Path) -> list[CurveResult]:
    """Load curves saved by a previous final comparison."""
    with np.load(input_path) as stored:
        return [
            {
                "method": str(method),
                "seed": int(seed),
                "density": density,
                "energy": energy,
            }
            for method, seed, density, energy in zip(
                stored["methods"],
                stored["seeds"],
                stored["densities"],
                stored["energies"],
            )
        ]


def plot_comparison(
    results: list[RunResult],
    output_path: Path,
) -> None:
    """Plot paired accuracy and runtime results across random seeds."""
    methods = ("SVGD", "Langevin")
    colors = ("tab:blue", "tab:orange")
    plotted_metrics = (
        ("density_error", "Integrated squared density error"),
        ("test_nll", "Test negative log-likelihood"),
        ("runtime_seconds", "Training time (seconds)"),
    )

    figure, axes = plt.subplots(
        1,
        len(plotted_metrics),
        figsize=(13, 4),
    )

    for axis, (metric, label) in zip(axes, plotted_metrics):
        for seed in SEEDS:
            seed_results = {
                result["method"]: result
                for result in results
                if result["seed"] == seed and result["stable"]
            }
            if all(method in seed_results for method in methods):
                axis.plot(
                    (0, 1),
                    [seed_results[method][metric] for method in methods],
                    color="0.8",
                    linewidth=1.0,
                    zorder=1,
                )

        for position, (method, color) in enumerate(zip(methods, colors)):
            values = np.asarray(
                [
                    result[metric]
                    for result in results
                    if result["method"] == method and result["stable"]
                ],
                dtype=float,
            )
            axis.scatter(
                np.full(len(values), position),
                values,
                color=color,
                alpha=0.8,
                s=34,
                label=method,
                zorder=2,
            )
            axis.scatter(
                position,
                values.mean(),
                color="black",
                marker="D",
                s=42,
                label="Mean" if position == 0 else None,
                zorder=3,
            )

        axis.set_xticks((0, 1), methods)
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.25)

    axes[0].legend(frameon=False)
    figure.suptitle(
        "Neural EBM training: SVGD versus Langevin",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")

    print(f"Figure saved to: {output_path}")
    plt.show()
    plt.close(figure)


def plot_density_and_energy_curves(
    curve_results: list[CurveResult],
    output_path: Path,
) -> None:
    """Plot exact and learned curves with across-seed uncertainty bands."""
    grid = np.linspace(GRID_MIN, GRID_MAX, N_GRID_POINTS)
    exact_density = target_density(grid)
    exact_energy = target_energy(grid)

    figure, (density_axis, energy_axis) = plt.subplots(
        1,
        2,
        figsize=(11, 4),
    )

    density_axis.plot(
        grid,
        exact_density,
        color="black",
        linewidth=2.0,
        label="Exact target",
    )
    energy_axis.plot(
        grid,
        exact_energy,
        color="black",
        linewidth=2.0,
        label="Exact target",
    )

    for method, color in (("SVGD", "tab:blue"), ("Langevin", "tab:orange")):
        method_curves = [
            curve for curve in curve_results if curve["method"] == method
        ]
        densities = np.stack([curve["density"] for curve in method_curves])
        energies = np.stack([curve["energy"] for curve in method_curves])
        mean_density = densities.mean(axis=0)
        std_density = densities.std(axis=0, ddof=1)
        mean_energy = energies.mean(axis=0)
        std_energy = energies.std(axis=0, ddof=1)

        density_axis.plot(grid, mean_density, color=color, label=method)
        density_axis.fill_between(
            grid,
            np.maximum(0.0, mean_density - std_density),
            mean_density + std_density,
            color=color,
            alpha=0.18,
        )
        energy_axis.plot(grid, mean_energy, color=color, label=method)
        energy_axis.fill_between(
            grid,
            mean_energy - std_energy,
            mean_energy + std_energy,
            color=color,
            alpha=0.18,
        )

    density_axis.set_xlabel("x")
    density_axis.set_ylabel("Normalized density")
    density_axis.set_title("Learned density")
    density_axis.legend(frameon=False)

    energy_axis.set_xlabel("x")
    energy_axis.set_ylabel("Normalized energy: -log p(x)")
    energy_axis.set_title("Learned energy")
    energy_axis.legend(frameon=False)

    for axis in (density_axis, energy_axis):
        axis.set_xlim(-6.0, 6.0)
        axis.grid(alpha=0.25)

    figure.suptitle("Neural EBM curves across 10 seeds", fontsize=14)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Curve figure saved to: {output_path}")
    plt.show()
    plt.close(figure)


def main() -> None:
    """Run the comparison, print its summary, and save every result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="regenerate the figure from the saved CSV without training",
    )
    arguments = parser.parse_args()

    output_directory = Path(__file__).resolve().parent / "results"
    output_directory.mkdir(parents=True, exist_ok=True)
    results_path = (
        output_directory
        / "ebm_svgd_langevin_neutral_init_1d.csv"
    )
    figure_path = (
        output_directory
        / "ebm_svgd_langevin_comparison_1d.png"
    )
    curves_path = output_directory / "ebm_svgd_langevin_curves_1d.npz"
    curves_figure_path = (
        output_directory / "ebm_svgd_langevin_curves_1d.png"
    )

    if arguments.plot_only:
        results = load_results(results_path)
        curve_results = (
            load_curves(curves_path) if curves_path.exists() else None
        )
    else:
        results, curve_results = run_comparison()
        save_results(results, results_path)
        save_curves(curve_results, curves_path)

    summary_metrics = [
        "density_error",
        "test_nll",
        "runtime_seconds",
        "left_mass",
        "right_mass",
        "learned_mean",
        "learned_second_moment",
        "particle_mean",
        "particle_std",
    ]

    print("\nExact target values:")
    print(f"mean={EXPECTED_MEAN:.6f}")
    print(f"second_moment={EXPECTED_SECOND_MOMENT:.6f}")

    for method in ("SVGD", "Langevin"):
        method_results = [
            result
            for result in results
            if result["method"] == method and result["stable"]
        ]

        print(f"\n{method} summary ({len(method_results)} stable runs):")
        if not method_results:
            print("No stable runs were completed.")
            continue

        for metric in summary_metrics:
            values = np.asarray(
                [result[metric] for result in method_results],
                dtype=float,
            )
            mean = values.mean()
            standard_deviation = (
                values.std(ddof=1)
                if len(values) > 1
                else 0.0
            )
            print(
                f"{metric}: mean={mean:.6f}, "
                f"std={standard_deviation:.6f}"
            )

    if not arguments.plot_only:
        print(f"\nResults saved to: {results_path}")

    plot_comparison(results, figure_path)
    if curve_results is not None:
        plot_density_and_energy_curves(
            curve_results,
            curves_figure_path,
        )
    else:
        print(
            "Curve data are not available in the old CSV. Run the full "
            "comparison once to create the density-and-energy figure."
        )


if __name__ == "__main__":
    main()
