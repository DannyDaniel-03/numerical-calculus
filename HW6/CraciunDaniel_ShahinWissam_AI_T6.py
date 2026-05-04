from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import os

os.environ.setdefault("MPLCONFIGDIR", ".matplotlib")

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class FunctionExample:
    name: str
    a: float
    b: float
    x_bar: float
    f: Callable[[np.ndarray | float], np.ndarray | float]
    f_prime: Callable[[np.ndarray | float], np.ndarray | float]

    @property
    def da(self) -> float:
        return float(self.f_prime(self.a))

    @property
    def db(self) -> float:
        return float(self.f_prime(self.b))


@dataclass(frozen=True)
class LeastSquaresPolynomial:
    degree: int
    coefficients: np.ndarray  # ascending order: a0, a1, ..., am


@dataclass(frozen=True)
class ClampedCubicSpline:
    x: np.ndarray
    y: np.ndarray
    A: np.ndarray
    b: np.ndarray
    c: np.ndarray


FUNCTIONS: dict[int, FunctionExample] = {
    1: FunctionExample(
        name="f(x) = x^4 - 12x^3 + 30x^2 + 12",
        a=0.0,
        b=2.0,
        x_bar=1.5,
        f=lambda x: np.asarray(x) ** 4 - 12 * np.asarray(x) ** 3 + 30 * np.asarray(x) ** 2 + 12,
        f_prime=lambda x: 4 * np.asarray(x) ** 3 - 36 * np.asarray(x) ** 2 + 60 * np.asarray(x),
    ),
    2: FunctionExample(
        name="f(x) = x^3 + 3x^2 - 5x + 12",
        a=1.0,
        b=5.0,
        x_bar=1.5,
        f=lambda x: np.asarray(x) ** 3 + 3 * np.asarray(x) ** 2 - 5 * np.asarray(x) + 12,
        f_prime=lambda x: 3 * np.asarray(x) ** 2 + 6 * np.asarray(x) - 5,
    ),
    3: FunctionExample(
        name="f(x) = x^4 - 10x^3 + 6x + 50",
        a=0.0,
        b=5.0,
        x_bar=1.5,
        f=lambda x: np.asarray(x) ** 4 - 10 * np.asarray(x) ** 3 + 6 * np.asarray(x) + 50,
        f_prime=lambda x: 4 * np.asarray(x) ** 3 - 30 * np.asarray(x) ** 2 + 6,
    ),
    4: FunctionExample(
        name="f(x) = sin(2x) + 0.5cos(5x)",
        a=0.0,
        b=2 * np.pi,
        x_bar=1.5,
        f=lambda x: np.sin(2 * np.asarray(x)) + 0.5 * np.cos(5 * np.asarray(x)),
        f_prime=lambda x: 2 * np.cos(2 * np.asarray(x)) - 2.5 * np.sin(5 * np.asarray(x)),
    ),
    5: FunctionExample(
        name="f(x) = e^(-0.15x)sin(3x)",
        a=0.0,
        b=8.0,
        x_bar=2.3,
        f=lambda x: np.exp(-0.15 * np.asarray(x)) * np.sin(3 * np.asarray(x)),
        f_prime=lambda x: (
                -0.15 * np.exp(-0.15 * np.asarray(x)) * np.sin(3 * np.asarray(x))
                + 3 * np.exp(-0.15 * np.asarray(x)) * np.cos(3 * np.asarray(x))
        ),
    ),
    6: FunctionExample(
        name="f(x) = sin(x^2)",
        a=0.0,
        b=4.0,
        x_bar=2.1,
        f=lambda x: np.sin(np.asarray(x) ** 2),
        f_prime=lambda x: 2 * np.asarray(x) * np.cos(np.asarray(x) ** 2),
    ),
}


def solve_linear_system(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """
    This is the numerical-library part allowed by the statement.

    It can be replaced with your Tema 2 LDLT solver because the two systems
    built here are symmetric. For the homework itself, np.linalg.solve is enough.
    """
    return np.linalg.solve(matrix, rhs)


def generate_interpolation_nodes(a: float, b: float, intervals: int, seed: int | None = None) -> np.ndarray:
    """
    Generates x0=a xn=b and n-1 random internal nodes in sorted order.

    Intervals = n from the statement, so the number of nodes is n + 1.
    """
    if not a < b:
        raise ValueError("Expected a < b")
    if intervals < 1:
        raise ValueError("Expected at least one interval, so n >= 1")

    rng = np.random.default_rng(seed)
    internal = rng.uniform(a, b, size=intervals - 1)
    nodes = np.concatenate(([a], np.sort(internal), [b]))

    if np.any(np.diff(nodes) <= 0):
        raise ValueError("Generated duplicate nodes; rerun with another seed")

    return nodes


def build_least_squares_polynomial(x: np.ndarray, y: np.ndarray, degree: int) -> LeastSquaresPolynomial:
    """
    Builds the normal equation system Ba = f from the statement:
        sum_j (sum_k x_k^(i+j)) a_j = sum_k y_k x_k^i.

    Coefficients are returned in ascending order: a0, a1, ..., am.
    """
    if degree < 0:
        raise ValueError("Polynomial degree must be non-negative")
    if degree >= 6:
        raise ValueError("The statement asks for m < 6")
    if degree >= len(x):
        raise ValueError("Need m < number of nodes for a full-rank least-squares system")

    size = degree + 1
    matrix = np.zeros((size, size), dtype=float)
    rhs = np.zeros(size, dtype=float)

    for i in range(size):
        rhs[i] = np.sum(y * x ** i)
        for j in range(size):
            matrix[i, j] = np.sum(x ** (i + j))

    coefficients = solve_linear_system(matrix, rhs)
    return LeastSquaresPolynomial(degree=degree, coefficients=coefficients)


def evaluate_polynomial_horner(coefficients: np.ndarray, values: np.ndarray | float) -> np.ndarray | float:
    """
    Horner evaluation for coefficients stored as a0, a1, ..., am.
    """
    values_array = np.asarray(values, dtype=float)
    result = np.zeros_like(values_array, dtype=float) + coefficients[-1]

    for coefficient in coefficients[-2::-1]:
        result = result * values_array + coefficient

    if np.isscalar(values):
        return float(result)

    return result


def build_clamped_cubic_spline(x: np.ndarray, y: np.ndarray, da: float, db: float) -> ClampedCubicSpline:
    """
    Builds the clamped C^2 cubic spline from the statement.

    A[i] are the second-derivative-like constants from the homework formula.
    The returned b[i], c[i] are the coefficients used directly in:
        S_i(x) = ((x-x_i)^3 A[i+1])/(6h_i)
               + ((x_{i+1}-x)^3 A[i])/(6h_i)
               + b[i]x + c[i]
    """
    node_count = len(x)
    if node_count != len(y):
        raise ValueError("x and y must have the same length")
    if node_count < 2:
        raise ValueError("Need at least two nodes")
    if np.any(np.diff(x) <= 0):
        raise ValueError("Nodes must be strictly increasing")

    intervals = node_count - 1
    h = np.diff(x)

    matrix = np.zeros((node_count, node_count), dtype=float)
    rhs = np.zeros(node_count, dtype=float)

    matrix[0, 0] = 2 * h[0]
    matrix[0, 1] = h[0]
    rhs[0] = 6 * ((y[1] - y[0]) / h[0] - da)

    for i in range(1, intervals):
        matrix[i, i - 1] = h[i - 1]
        matrix[i, i] = 2 * (h[i - 1] + h[i])
        matrix[i, i + 1] = h[i]
        rhs[i] = 6 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

    matrix[intervals, intervals - 1] = h[intervals - 1]
    matrix[intervals, intervals] = 2 * h[intervals - 1]
    rhs[intervals] = 6 * (db - (y[intervals] - y[intervals - 1]) / h[intervals - 1])

    A = solve_linear_system(matrix, rhs)

    b_coeff = np.zeros(intervals, dtype=float)
    c_coeff = np.zeros(intervals, dtype=float)

    for i in range(intervals):
        b_coeff[i] = (y[i + 1] - y[i]) / h[i] - h[i] * (A[i + 1] - A[i]) / 6
        c_coeff[i] = (
                (x[i + 1] * y[i] - x[i] * y[i + 1]) / h[i]
                - h[i] * (x[i + 1] * A[i] - x[i] * A[i + 1]) / 6
        )

    return ClampedCubicSpline(x=x, y=y, A=A, b=b_coeff, c=c_coeff)


def evaluate_clamped_cubic_spline(spline: ClampedCubicSpline, values: np.ndarray | float) -> np.ndarray | float:
    values_array = np.asarray(values, dtype=float)

    if np.any(values_array < spline.x[0]) or np.any(values_array > spline.x[-1]):
        raise ValueError("Spline can only be evaluated on [a, b]")

    interval_index = np.searchsorted(spline.x, values_array, side="right") - 1
    interval_index = np.clip(interval_index, 0, len(spline.x) - 2)

    x_i = spline.x[interval_index]
    x_next = spline.x[interval_index + 1]
    h_i = x_next - x_i

    A_i = spline.A[interval_index]
    A_next = spline.A[interval_index + 1]
    b_i = spline.b[interval_index]
    c_i = spline.c[interval_index]

    result = (
            ((values_array - x_i) ** 3 * A_next) / (6 * h_i)
            + ((x_next - values_array) ** 3 * A_i) / (6 * h_i)
            + b_i * values_array
            + c_i
    )

    if np.isscalar(values):
        return float(result)

    return result


def read_int(prompt: str, default: int) -> int:
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default
    return int(raw)


def read_float(prompt: str, default: float) -> float:
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default
    return float(raw)


def choose_function() -> FunctionExample:
    print("Available examples:")
    for key, example in FUNCTIONS.items():
        print(f"  {key}. {example.name}, a={example.a}, b={example.b}, da={example.da}, db={example.db}")

    choice = read_int("Choose function", 3)
    if choice not in FUNCTIONS:
        raise ValueError(f"Unknown function choice: {choice}")

    return FUNCTIONS[choice]


def print_results(
        example: FunctionExample,
        x_nodes: np.ndarray,
        y_nodes: np.ndarray,
        polynomial: LeastSquaresPolynomial,
        spline: ClampedCubicSpline,
        x_bar: float,
) -> None:
    p_x_bar = evaluate_polynomial_horner(polynomial.coefficients, x_bar)
    s_x_bar = evaluate_clamped_cubic_spline(spline, x_bar)
    exact_x_bar = float(example.f(x_bar))

    p_at_nodes = evaluate_polynomial_horner(polynomial.coefficients, x_nodes)
    polynomial_node_error_sum = float(np.sum(np.abs(p_at_nodes - y_nodes) ** 2))

    np.set_printoptions(precision=10, suppress=True)

    print("\n=== Input data ===")
    print(f"Function: {example.name}")
    print(f"a = {x_nodes[0]}, b = {x_nodes[-1]}, x_bar = {x_bar}")
    print(f"da = {example.da}, db = {example.db}")
    print("x nodes =")
    print(x_nodes)
    print("y values =")
    print(y_nodes)

    print("\n=== Least-squares polynomial ===")
    print(f"m = {polynomial.degree}")
    print("coefficients a0..am =")
    print(polynomial.coefficients)
    print(f"Pm(x_bar) = {p_x_bar}")
    print(f"|Pm(x_bar) - f(x_bar)| = {abs(p_x_bar - exact_x_bar)}")
    print(f"sum_i |Pm(x_i) - y_i|^2 = {polynomial_node_error_sum}")

    print("\n=== Clamped cubic spline C^2 ===")
    print("A coefficients =")
    print(spline.A)
    print(f"Sf(x_bar) = {s_x_bar}")
    print(f"|Sf(x_bar) - f(x_bar)| = {abs(s_x_bar - exact_x_bar)}")


def plot_results(
        example: FunctionExample,
        x_nodes: np.ndarray,
        y_nodes: np.ndarray,
        polynomial: LeastSquaresPolynomial,
        spline: ClampedCubicSpline,
        output_path: str = "tema6_plot.png",
) -> None:
    grid = np.linspace(x_nodes[0], x_nodes[-1], 500)
    f_values = example.f(grid)
    p_values = evaluate_polynomial_horner(polynomial.coefficients, grid)
    s_values = evaluate_clamped_cubic_spline(spline, grid)

    plt.figure(figsize=(10, 6))
    plt.plot(grid, f_values, label="f(x)")
    plt.plot(grid, p_values, label=f"P{polynomial.degree}(x)")
    plt.plot(grid, s_values, label="Sf(x)")
    plt.scatter(x_nodes, y_nodes, label="interpolation nodes")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Tema 6: least-squares polynomial and clamped cubic spline")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    print(f"\nPlot saved to: {output_path}")
    plt.close()


def main() -> None:
    example = choose_function()

    intervals = read_int("n from the statement; number of intervals, nodes = n + 1", 10)
    degree = read_int("m for least-squares polynomial, with m < 6", 4)
    seed = read_int("random seed for internal nodes", 12345)

    a = read_float("x0 = a", example.a)
    b = read_float("xn = b", example.b)
    x_bar = read_float("x_bar", example.x_bar)

    if not a <= x_bar <= b:
        raise ValueError("x_bar must be inside [a, b]")

    x_nodes = generate_interpolation_nodes(a, b, intervals, seed=seed)

    if np.any(np.isclose(x_nodes, x_bar, rtol=0.0, atol=1e-12)):
        raise ValueError("x_bar must not be equal to any interpolation node")

    y_nodes = np.asarray(example.f(x_nodes), dtype=float)

    polynomial = build_least_squares_polynomial(x_nodes, y_nodes, degree)

    da = float(example.f_prime(a))
    db = float(example.f_prime(b))

    spline = build_clamped_cubic_spline(x_nodes, y_nodes, da, db)

    print_results(example, x_nodes, y_nodes, polynomial, spline, x_bar)
    plot_results(example, x_nodes, y_nodes, polynomial, spline)


if __name__ == "__main__":
    main()
