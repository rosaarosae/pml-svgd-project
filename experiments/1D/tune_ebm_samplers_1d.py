"""Tune SVGD and Langevin step sizes on separate validation seeds.

The final comparison seeds are deliberately excluded from this program.
Both samplers reuse the same model, training loop, metrics, particle count,
and neutral particle initialization from compare_ebm_svgd_langevin_1d.py.
"""

import csv
from pathlib import Path

import numpy as np

from compare_ebm_svgd_langevin_1d import RunResult, train_one_run


VALIDATION_SEEDS = [107, 117, 127, 137, 147]

STEP_SIZES = {
    "SVGD": [0.01, 0.015, 0.02, 0.025, 0.03],
    "Langevin": [0.02, 0.03, 0.04, 0.05, 0.075],
}

# Penalize configurations whose results change substantially across seeds.
# A value of one selects the smallest mean-plus-one-standard-deviation score.
VARIABILITY_PENALTY = 1.0


def main() -> None:
    """Evaluate each candidate and report the best stable configuration."""
    results: list[dict[str, object]] = []

    for method, candidate_step_sizes in STEP_SIZES.items():
        for step_size in candidate_step_sizes:
            for seed in VALIDATION_SEEDS:
                print(
                    f"Validating {method}: "
                    f"step_size={step_size}, seed={seed}..."
                )

                run_result: RunResult = train_one_run(
                    method=method,
                    seed=seed,
                    step_size=step_size,
                )
                results.append(
                    {
                        "step_size": step_size,
                        **run_result,
                    }
                )

                print(
                    f"density_error={run_result['density_error']:.6f}, "
                    f"test_nll={run_result['test_nll']:.6f}, "
                    f"stable={run_result['stable']}"
                )

    output_directory = Path(__file__).resolve().parent / "results"
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / "ebm_sampler_tuning_1d.csv"

    fieldnames = ["step_size", *RunResult.__annotations__]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    for method, candidate_step_sizes in STEP_SIZES.items():
        summaries: list[tuple[float, float, float, float, float]] = []

        print(f"\n{method} validation summary:")
        for step_size in candidate_step_sizes:
            matching_results = [
                result
                for result in results
                if result["method"] == method
                and result["step_size"] == step_size
                and result["stable"]
            ]

            if len(matching_results) != len(VALIDATION_SEEDS):
                print(
                    f"step_size={step_size}: excluded because only "
                    f"{len(matching_results)}/{len(VALIDATION_SEEDS)} "
                    "runs were stable"
                )
                continue

            density_errors = np.asarray(
                [result["density_error"] for result in matching_results],
                dtype=float,
            )
            test_nlls = np.asarray(
                [result["test_nll"] for result in matching_results],
                dtype=float,
            )
            mean_density_error = density_errors.mean()
            std_density_error = density_errors.std(ddof=1)
            mean_test_nll = test_nlls.mean()
            robust_score = (
                mean_density_error
                + VARIABILITY_PENALTY * std_density_error
            )
            summaries.append(
                (
                    step_size,
                    mean_density_error,
                    std_density_error,
                    mean_test_nll,
                    robust_score,
                )
            )

            print(
                f"step_size={step_size}: "
                f"density_error={mean_density_error:.6f} "
                f"+/- {std_density_error:.6f}, "
                f"robust_score={robust_score:.6f}, "
                f"test_nll={mean_test_nll:.6f} "
                f"+/- {test_nlls.std(ddof=1):.6f}"
            )

        if summaries:
            (
                best_step_size,
                best_error,
                best_error_std,
                best_nll,
                best_robust_score,
            ) = min(
                summaries,
                key=lambda summary: summary[4],
            )
            print(
                f"Best {method} step_size: {best_step_size} "
                f"(density_error={best_error:.6f} "
                f"+/- {best_error_std:.6f}, "
                f"robust_score={best_robust_score:.6f}, "
                f"test_nll={best_nll:.6f})"
            )

    print(f"\nValidation results saved to: {output_path}")


if __name__ == "__main__":
    main()
