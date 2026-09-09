"""Compare samplers on the paper-inspired two-dimensional GMM extension."""

from csv import DictWriter
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from gmm_2d import (
    DIMENSION,
    MEANS,
    SEED,
    TRANSPORT_X_LIMITS,
    Y_LIMITS,
    mixture_density,
    sample_initial_particles,
    sample_target,
    target_score,
)
from langevin_2d import (
    N_PARTICLES as LANGEVIN_N_PARTICLES,
    N_STEPS as LANGEVIN_N_STEPS,
    STEP_SIZE as LANGEVIN_STEP_SIZE,
    langevin_dynamics,
)
from metrics_2d import (
    SampleMetrics,
    evaluate_samples,
    numerical_expected_target_energy,
)
from svgd_2d import (
    N_PARTICLES as SVGD_N_PARTICLES,
    N_STEPS as SVGD_N_STEPS,
    STEP_SIZE as SVGD_STEP_SIZE,
    svgd_dynamics,
)


SEEDS = [SEED, 17, 27, 37, 47]
METHOD_NAMES = ["Target", "Langevin", "SVGD"]
METRIC_NAMES = [
    "mean_target_energy",
    "energy_error",
    "mode_coverage",
    "mode_weight_error",
    "mode_center_error",
    "within_mode_variance_error",
    "sliced_wasserstein",
]


def main() -> None:
    if LANGEVIN_N_PARTICLES != SVGD_N_PARTICLES:
        raise ValueError("Both samplers must use the same number of particles")
    if LANGEVIN_N_STEPS != SVGD_N_STEPS:
        raise ValueError("Both samplers must use the same number of steps")

    n_particles = LANGEVIN_N_PARTICLES
    n_steps = LANGEVIN_N_STEPS
    results = _empty_results()
    per_seed_rows = []
    representative_samples = None
    expected_target_energy = numerical_expected_target_energy()

    print(f"Numerical expected target energy: {expected_target_energy:.6f}\n")

    for seed in SEEDS:
        # Both algorithms receive copies of exactly the same initial particles.
        initial_rng = np.random.default_rng(seed)
        initial_particles = sample_initial_particles(
            n_particles,
            initial_rng,
        )
        initial_snapshot = initial_particles.copy()

        # The reference is independent of the second direct target sample. Their
        # comparison gives the finite-sample error floor for 100 exact samples.
        reference_rng = np.random.default_rng(seed + 2_000)
        reference_samples = sample_target(n_particles, reference_rng)
        target_rng = np.random.default_rng(seed + 3_000)
        target_samples = sample_target(n_particles, target_rng)

        langevin_rng = np.random.default_rng(seed + 1_000)
        start_time = perf_counter()
        langevin_particles = langevin_dynamics(
            initial_particles=initial_particles,
            n_steps=n_steps,
            step_size=LANGEVIN_STEP_SIZE,
            score_function=target_score,
            rng=langevin_rng,
        )
        langevin_runtime = perf_counter() - start_time

        start_time = perf_counter()
        svgd_particles = svgd_dynamics(
            initial_particles=initial_particles,
            n_steps=n_steps,
            step_size=SVGD_STEP_SIZE,
            score_function=target_score,
        )
        svgd_runtime = perf_counter() - start_time

        # A regression guard for the fairness condition: neither sampler may
        # mutate the shared starting array.
        assert np.array_equal(initial_particles, initial_snapshot)

        samples_by_method = {
            "Target": target_samples,
            "Langevin": langevin_particles,
            "SVGD": svgd_particles,
        }
        runtimes = {
            "Target": float("nan"),
            "Langevin": langevin_runtime,
            "SVGD": svgd_runtime,
        }

        for method_name, samples in samples_by_method.items():
            metrics = evaluate_samples(
                samples,
                reference_samples,
                expected_target_energy,
            )
            _record_result(
                results,
                per_seed_rows,
                seed,
                method_name,
                metrics,
                runtimes[method_name],
            )

        if seed == SEEDS[0]:
            representative_samples = {
                "Initial": initial_particles.copy(),
                "Target": target_samples.copy(),
                "Langevin": langevin_particles.copy(),
                "SVGD": svgd_particles.copy(),
            }

        print(
            f"Seed {seed} completed: "
            f"Langevin {langevin_runtime:.3f} s, "
            f"SVGD {svgd_runtime:.3f} s"
        )

    summaries = _summarize_results(results)
    _print_summary(summaries)
    _print_paired_wins(results)

    output_directory = Path(__file__).resolve().parent / "results"
    output_directory.mkdir(parents=True, exist_ok=True)

    _save_per_seed_results(per_seed_rows, output_directory)
    _save_summary_results(summaries, output_directory)
    _plot_representative_particles(representative_samples, output_directory)
    _plot_metric_summary(summaries, output_directory)

    print(f"\nResults saved in: {output_directory}")
    plt.show()


def _empty_results() -> dict:
    """Create storage for every metric and runtime."""
    return {
        method: {
            **{metric: [] for metric in METRIC_NAMES},
            "runtime": [],
        }
        for method in METHOD_NAMES
    }


def _record_result(
    results: dict,
    rows: list[dict],
    seed: int,
    method_name: str,
    metrics: SampleMetrics,
    runtime: float,
) -> None:
    """Store one method result in array and row representations."""
    metric_values = asdict(metrics)
    for metric_name, value in metric_values.items():
        results[method_name][metric_name].append(value)
    results[method_name]["runtime"].append(runtime)

    rows.append(
        {
            "seed": seed,
            "method": method_name,
            **metric_values,
            "runtime_seconds": runtime,
        }
    )


def _summarize_results(results: dict) -> dict:
    """Calculate mean and sample standard deviation across seeds."""
    summaries = {}
    for method_name in METHOD_NAMES:
        summaries[method_name] = {}
        for metric_name, values in results[method_name].items():
            values = np.asarray(values, dtype=float)
            finite_values = values[np.isfinite(values)]
            if len(finite_values) == 0:
                mean = float("nan")
                std = float("nan")
            else:
                mean = float(np.mean(finite_values))
                std = float(np.std(finite_values, ddof=1))
            summaries[method_name][metric_name] = (mean, std)
    return summaries


def _print_summary(summaries: dict) -> None:
    """Print compact tables with all comparison results."""
    print("\nQuality metrics across five seeds (mean ± standard deviation):")
    print("method    | energy        | energy error | modes | weight error")
    print("----------+---------------+--------------+-------+-------------")
    for method_name in METHOD_NAMES:
        energy = summaries[method_name]["mean_target_energy"]
        energy_error = summaries[method_name]["energy_error"]
        coverage = summaries[method_name]["mode_coverage"]
        weight_error = summaries[method_name]["mode_weight_error"]
        print(
            f"{method_name:<9} | "
            f"{energy[0]:.3f} ± {energy[1]:.3f} | "
            f"{energy_error[0]:.3f} ± {energy_error[1]:.3f} | "
            f"{coverage[0]:.1f}   | "
            f"{weight_error[0]:.3f} ± {weight_error[1]:.3f}"
        )

    print("\nDistribution and efficiency metrics (lower is better):")
    print("method    | sliced-W2     | centre error | variance error | runtime")
    print("----------+---------------+--------------+----------------+---------")
    for method_name in METHOD_NAMES:
        sliced_wasserstein = summaries[method_name]["sliced_wasserstein"]
        center_error = summaries[method_name]["mode_center_error"]
        variance_error = summaries[method_name]["within_mode_variance_error"]
        runtime = summaries[method_name]["runtime"]
        runtime_text = "reference" if not np.isfinite(runtime[0]) else (
            f"{runtime[0]:.3f} ± {runtime[1]:.3f} s"
        )
        print(
            f"{method_name:<9} | "
            f"{sliced_wasserstein[0]:.3f} ± {sliced_wasserstein[1]:.3f} | "
            f"{center_error[0]:.3f} ± {center_error[1]:.3f} | "
            f"{variance_error[0]:.3f} ± {variance_error[1]:.3f} | "
            f"{runtime_text}"
        )


def _print_paired_wins(results: dict) -> None:
    """Show how often each sampler wins under paired initial conditions."""
    comparison_metrics = [
        ("energy_error", "Energy error"),
        ("mode_weight_error", "Mixture-weight error"),
        ("mode_center_error", "Mode-centre error"),
        ("within_mode_variance_error", "Within-mode variance error"),
        ("sliced_wasserstein", "Sliced Wasserstein"),
        ("runtime", "Runtime"),
    ]

    print("\nPaired wins across seeds (lower value wins):")
    print("metric                     | Langevin | SVGD | ties")
    print("---------------------------+----------+------+-----")
    for metric_name, label in comparison_metrics:
        langevin_values = np.asarray(results["Langevin"][metric_name])
        svgd_values = np.asarray(results["SVGD"][metric_name])
        langevin_wins = int(np.sum(langevin_values < svgd_values))
        svgd_wins = int(np.sum(svgd_values < langevin_values))
        ties = len(SEEDS) - langevin_wins - svgd_wins
        print(f"{label:<26} | {langevin_wins:^8} | {svgd_wins:^4} | {ties:^4}")


def _save_per_seed_results(rows: list[dict], output_directory: Path) -> None:
    """Save every individual run to CSV."""
    path = output_directory / "compare_svgd_langevin_2d_per_seed.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _save_summary_results(summaries: dict, output_directory: Path) -> None:
    """Save means and standard deviations to CSV."""
    path = output_directory / "compare_svgd_langevin_2d_summary.csv"
    rows = []
    for method_name in METHOD_NAMES:
        row = {"method": method_name}
        for metric_name, (mean, std) in summaries[method_name].items():
            row[f"{metric_name}_mean"] = mean
            row[f"{metric_name}_std"] = std
        rows.append(row)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _plot_representative_particles(
    representative_samples: dict,
    output_directory: Path,
) -> None:
    """Plot initial, exact, Langevin, and SVGD particles for seed 7."""
    x_values = np.linspace(*TRANSPORT_X_LIMITS, 220)
    y_values = np.linspace(*Y_LIMITS, 150)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    grid_points = np.column_stack([x_grid.ravel(), y_grid.ravel()])
    density_grid = mixture_density(grid_points).reshape(x_grid.shape)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)
    for axis, (name, samples) in zip(axes.ravel(), representative_samples.items()):
        axis.contourf(
            x_grid, y_grid, density_grid, levels=40, cmap="viridis"
        )
        axis.scatter(
            samples[:, 0], samples[:, 1], s=7, color="red", alpha=0.35
        )
        axis.scatter(
            MEANS[:, 0],
            MEANS[:, 1],
            marker="x",
            s=60,
            linewidths=2,
            color="white",
        )
        axis.set_title(name)
        axis.set_xlabel("x₁")
        axis.set_ylabel("x₂")
        axis.set_xlim(*TRANSPORT_X_LIMITS)
        axis.set_ylim(*Y_LIMITS)

    fig.suptitle(f"Direct sampler comparison (representative seed {SEEDS[0]})")
    fig.tight_layout()
    path = output_directory / "compare_svgd_langevin_2d_particles.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")


def _plot_metric_summary(summaries: dict, output_directory: Path) -> None:
    """Plot the principal quality metrics and sampler runtime."""
    quality_plots = [
        ("energy_error", "Mean-energy error"),
        ("sliced_wasserstein", "Sliced Wasserstein"),
        ("mode_weight_error", "Mixture-weight error"),
        ("within_mode_variance_error", "Within-mode variance error"),
        ("mode_center_error", "Mode-centre error"),
    ]
    colors = ["0.55", "#4C78A8", "#F58518"]
    x_positions = np.arange(len(METHOD_NAMES))
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))

    for axis, (metric_name, title) in zip(axes.ravel(), quality_plots):
        means = [summaries[name][metric_name][0] for name in METHOD_NAMES]
        stds = [summaries[name][metric_name][1] for name in METHOD_NAMES]
        axis.bar(x_positions, means, yerr=stds, capsize=4, color=colors)
        axis.set_xticks(x_positions, METHOD_NAMES)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)

    runtime_axis = axes.ravel()[-1]
    sampler_names = ["Langevin", "SVGD"]
    runtime_means = [summaries[name]["runtime"][0] for name in sampler_names]
    runtime_stds = [summaries[name]["runtime"][1] for name in sampler_names]
    runtime_axis.bar(
        np.arange(2),
        runtime_means,
        yerr=runtime_stds,
        capsize=4,
        color=colors[1:],
    )
    runtime_axis.set_xticks(np.arange(2), sampler_names)
    runtime_axis.set_title("Runtime (seconds)")
    runtime_axis.grid(axis="y", alpha=0.25)

    fig.suptitle("Langevin and SVGD comparison across five seeds")
    fig.tight_layout()
    path = output_directory / "compare_svgd_langevin_2d_metrics.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
