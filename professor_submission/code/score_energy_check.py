"""Check the energy-score identity for the paper's 1D target."""

import numpy as np

from gmm_1d import target_energy, target_score

# Calculate a numerical approximation of the energy derivative.
def numerical_derivative(f, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """Compute the numerical derivative of f at x using central differences."""
    return (f(x + h) - f(x - h)) / (2 * h)

def check_score_energy_relation(x: np.ndarray) -> None:
    """Check that the score function is the negative gradient of the energy function."""
    # Compute the score and the numerical derivative of the energy.
    score_values = target_score(x)
    energy_derivative = numerical_derivative(target_energy, x)

    # Check if they are approximately equal.
    if np.allclose(score_values, -energy_derivative, atol=1e-5):
        print("The score function is consistent with the energy function.")
    else:
        print("Discrepancy found between the score and energy functions.")
        print("Score values:", score_values)
        print("Negative energy derivative:", -energy_derivative)

def main() -> None:
    # Generate a range of x values to test.
    x_values = np.linspace(-12, 8, 200)
    check_score_energy_relation(x_values)

if __name__ == "__main__":
    main()
