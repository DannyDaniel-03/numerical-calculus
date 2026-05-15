# Shahin, Wissam, 310910401ESL231073, Wissam.shahin05@gmail.com, shukakah
# Craciun, Daniel, 310910401ESL231020, danielcraciun72@gmail.com, donnavant
# Estimated AI-assisted portion: 35% (questions, print/output wording, and small refinements)
# Bibliography:
# GPT: used mainly for questions about how to approach the solution and for checking/refining explanations/prints
# Course PDF: main source for formulas, pseudo-code, algorithms, and homework requirements

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import os

os.environ.setdefault("MPLCONFIGDIR", ".matplotlib")

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class FunctionData:
    name: str
    a: float
    b: float
    x_query: float
    f: Callable[[np.ndarray | float], np.ndarray | float]
    f_prime: Callable[[np.ndarray | float], np.ndarray | float]

    @property
    def da(self) -> float:
        return float(self.f_prime(self.a))

    @property
    def db(self) -> float:
        return float(self.f_prime(self.b))


@dataclass(frozen=True)
class LsPolynomial:
    poly_degree: int
    coefs: np.ndarray  # ascending order: a0, a1, ..., am


@dataclass(frozen=True)
class CubicSplineData:
    x: np.ndarray
    y: np.ndarray
    A: np.ndarray
    b: np.ndarray
    c: np.ndarray


EXAMPLES: dict[int, FunctionData] = {
    1: FunctionData(
        name="f(x) = x^4 - 12x^3 + 30x^2 + 12",
        a=0.0,
        b=2.0,
        x_query=1.5,
        f=lambda x: np.asarray(x) ** 4 - 12 * np.asarray(x) ** 3 + 30 * np.asarray(x) ** 2 + 12,
        f_prime=lambda x: 4 * np.asarray(x) ** 3 - 36 * np.asarray(x) ** 2 + 60 * np.asarray(x),
    ),
    2: FunctionData(
        name="f(x) = x^3 + 3x^2 - 5x + 12",
        a=1.0,
        b=5.0,
        x_query=1.5,
        f=lambda x: np.asarray(x) ** 3 + 3 * np.asarray(x) ** 2 - 5 * np.asarray(x) + 12,
        f_prime=lambda x: 3 * np.asarray(x) ** 2 + 6 * np.asarray(x) - 5,
    ),
    3: FunctionData(
        name="f(x) = x^4 - 10x^3 + 6x + 50",
        a=0.0,
        b=5.0,
        x_query=1.5,
        f=lambda x: np.asarray(x) ** 4 - 10 * np.asarray(x) ** 3 + 6 * np.asarray(x) + 50,
        f_prime=lambda x: 4 * np.asarray(x) ** 3 - 30 * np.asarray(x) ** 2 + 6,
    ),
    4: FunctionData(
        name="f(x) = sin(2x) + 0.5cos(5x)",
        a=0.0,
        b=2 * np.pi,
        x_query=1.5,
        f=lambda x: np.sin(2 * np.asarray(x)) + 0.5 * np.cos(5 * np.asarray(x)),
        f_prime=lambda x: 2 * np.cos(2 * np.asarray(x)) - 2.5 * np.sin(5 * np.asarray(x)),
    ),
    5: FunctionData(
        name="f(x) = e^(-0.15x)sin(3x)",
        a=0.0,
        b=8.0,
        x_query=2.3,
        f=lambda x: np.exp(-0.15 * np.asarray(x)) * np.sin(3 * np.asarray(x)),
        f_prime=lambda x: (
                -0.15 * np.exp(-0.15 * np.asarray(x)) * np.sin(3 * np.asarray(x))
                + 3 * np.exp(-0.15 * np.asarray(x)) * np.cos(3 * np.asarray(x))
        ),
    ),
    6: FunctionData(
        name="f(x) = sin(x^2)",
        a=0.0,
        b=4.0,
        x_query=2.1,
        f=lambda x: np.sin(np.asarray(x) ** 2),
        f_prime=lambda x: 2 * np.asarray(x) * np.cos(np.asarray(x) ** 2),
    ),
}


def solve_system(system_mat: np.ndarray, right_side: np.ndarray) -> np.ndarray:
    """
    This is the numerical-library part allowed by the statement.
    It can be replaced with your Tema 2 LDLT solver because the two systems
    built here are symmetric. For the homework itself, np.linalg.solve is enough.
    """
    return np.linalg.solve(system_mat, right_side)


def make_nodes(a: float, b: float, parts: int, seed_no: int | None = None) -> np.ndarray:
    """
    Generates x0=a xn=b and n-1 random inside_nodes made_nodes in sorted order.
    Intervals = n from the statement, so the number of made_nodes is n + 1.
    """
    if not a < b:
        raise ValueError("Expected a < b")
    if parts < 1:
        raise ValueError("Expected at least one interval, so n >= 1")
    rng = np.random.default_rng(seed_no)
    inside_nodes = rng.uniform(a, b, size=parts - 1)
    made_nodes = np.concatenate(([a], np.sort(inside_nodes), [b]))
    if np.any(np.diff(made_nodes) <= 0):
        raise ValueError("Generated duplicate made_nodes; rerun with another seed_no")
    return made_nodes


def make_ls_poly(x: np.ndarray, y: np.ndarray, poly_degree: int) -> LsPolynomial:
    """
    Builds the normal equation system Ba = f from the statement:
        sum_j (sum_k x_k^(i+j)) a_j = sum_k y_k x_k^i.
    Coefficients are returned in ascending order: a0, a1, ..., am.
    """
    if poly_degree < 0:
        raise ValueError("Polynomial poly_degree must be non-negative")
    if poly_degree >= 6:
        raise ValueError("The statement asks for m < 6")
    if poly_degree >= len(x):
        raise ValueError("Need m < number of made_nodes for a full-rank least-squares system")
    size = poly_degree + 1
    system_mat = np.zeros((size, size), dtype=float)
    right_side = np.zeros(size, dtype=float)
    for i in range(size):
        right_side[i] = np.sum(y * x ** i)
        for j in range(size):
            system_mat[i, j] = np.sum(x ** (i + j))
    coefs = solve_system(system_mat, right_side)
    return LsPolynomial(poly_degree=poly_degree, coefs=coefs)


def horner_value(coefs: np.ndarray, values: np.ndarray | float) -> np.ndarray | float:
    """
    Horner evaluation for coefs stored as a0, a1, ..., am.
    """
    all_values = np.asarray(values, dtype=float)
    answer_now = np.zeros_like(all_values, dtype=float) + coefs[-1]
    for coef_now in coefs[-2::-1]:
        answer_now = answer_now * all_values + coef_now
    if np.isscalar(values):
        return float(answer_now)
    return answer_now


def make_clamped_spline(x: np.ndarray, y: np.ndarray, da: float, db: float) -> CubicSplineData:
    """
    Builds the clamped C^2 cubic sp_line from the statement.
    A[i] are the second-derivative-like constants from the homework formula.
    The returned b[i], c[i] are the coefs used directly in:
        S_i(x) = ((x-x_at)^3 A[i+1])/(6h_i)
               + ((x_{i+1}-x)^3 A[i])/(6h_i)
               + b[i]x + c[i]
    """
    points_count = len(x)
    if points_count != len(y):
        raise ValueError("x and y must have the same length")
    if points_count < 2:
        raise ValueError("Need at least two made_nodes")
    if np.any(np.diff(x) <= 0):
        raise ValueError("Nodes must be strictly increasing")
    parts = points_count - 1
    h = np.diff(x)
    system_mat = np.zeros((points_count, points_count), dtype=float)
    right_side = np.zeros(points_count, dtype=float)
    system_mat[0, 0] = 2 * h[0]
    system_mat[0, 1] = h[0]
    right_side[0] = 6 * ((y[1] - y[0]) / h[0] - da)
    for i in range(1, parts):
        system_mat[i, i - 1] = h[i - 1]
        system_mat[i, i] = 2 * (h[i - 1] + h[i])
        system_mat[i, i + 1] = h[i]
        right_side[i] = 6 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
    system_mat[parts, parts - 1] = h[parts - 1]
    system_mat[parts, parts] = 2 * h[parts - 1]
    right_side[parts] = 6 * (db - (y[parts] - y[parts - 1]) / h[parts - 1])
    A = solve_system(system_mat, right_side)
    b_vals = np.zeros(parts, dtype=float)
    c_vals = np.zeros(parts, dtype=float)
    for i in range(parts):
        b_vals[i] = (y[i + 1] - y[i]) / h[i] - h[i] * (A[i + 1] - A[i]) / 6
        c_vals[i] = (
                (x[i + 1] * y[i] - x[i] * y[i + 1]) / h[i]
                - h[i] * (x[i + 1] * A[i] - x[i] * A[i + 1]) / 6
        )
    return CubicSplineData(x=x, y=y, A=A, b=b_vals, c=c_vals)


def spline_value(sp_line: CubicSplineData, values: np.ndarray | float) -> np.ndarray | float:
    all_values = np.asarray(values, dtype=float)
    if np.any(all_values < sp_line.x[0]) or np.any(all_values > sp_line.x[-1]):
        raise ValueError("Spline can only be evaluated on [a, b]")
    part_index = np.searchsorted(sp_line.x, all_values, side="right") - 1
    part_index = np.clip(part_index, 0, len(sp_line.x) - 2)
    x_at = sp_line.x[part_index]
    x_after = sp_line.x[part_index + 1]
    h_now = x_after - x_at
    A_here = sp_line.A[part_index]
    A_after = sp_line.A[part_index + 1]
    b_here = sp_line.b[part_index]
    c_here = sp_line.c[part_index]
    answer_now = (
            ((all_values - x_at) ** 3 * A_after) / (6 * h_now)
            + ((x_after - all_values) ** 3 * A_here) / (6 * h_now)
            + b_here * all_values
            + c_here
    )
    if np.isscalar(values):
        return float(answer_now)
    return answer_now


def ask_int(prompt: str, default: int) -> int:
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default
    return int(raw)


def ask_float(prompt: str, default: float) -> float:
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default
    return float(raw)


def pick_function() -> FunctionData:
    print("Available examples:")
    for key, func_item in EXAMPLES.items():
        print(f"  {key}. {func_item.name}, a={func_item.a}, b={func_item.b}, da={func_item.da}, db={func_item.db}")
    picked_no = ask_int("Choose function", 3)
    if picked_no not in EXAMPLES:
        raise ValueError(f"Unknown function picked_no: {picked_no}")
    return EXAMPLES[picked_no]


# Generated by AI: full print block for the final homework output formatting.
def print_answers(
        func_item: FunctionData,
        x_list: np.ndarray,
        y_list: np.ndarray,
        poly: LsPolynomial,
        sp_line: CubicSplineData,
        x_query: float,
) -> None:
    poly_at_query = horner_value(poly.coefs, x_query)
    spline_at_query = spline_value(sp_line, x_query)
    real_at_query = float(func_item.f(x_query))
    poly_on_nodes = horner_value(poly.coefs, x_list)
    sum_nodes_error = float(np.sum(np.abs(poly_on_nodes - y_list) ** 2))
    np.set_printoptions(precision=10, suppress=True)
    print("\n=== Input data ===")
    print(f"Function: {func_item.name}")
    print(f"a = {x_list[0]}, b = {x_list[-1]}, x_query = {x_query}")
    print(f"da = {func_item.da}, db = {func_item.db}")
    print("x nodes =")
    print(x_list)
    print("y values =")
    print(y_list)
    print("\n=== Least-squares polynomial ===")
    print(f"m = {poly.poly_degree}")
    print("coefficients a0..am =")
    print(poly.coefs)
    print(f"Pm(x_query) = {poly_at_query}")
    print(f"|Pm(x_query) - f(x_query)| = {abs(poly_at_query - real_at_query)}")
    print(f"sum_i |Pm(x_i) - y_i|^2 = {sum_nodes_error}")
    print("\n=== Clamped cubic spline C^2 ===")
    print("A coefficients =")
    print(sp_line.A)
    print(f"Sf(x_query) = {spline_at_query}")
    print(f"|Sf(x_query) - f(x_query)| = {abs(spline_at_query - real_at_query)}")


def draw_plot(
        func_item: FunctionData,
        x_list: np.ndarray,
        y_list: np.ndarray,
        poly: LsPolynomial,
        sp_line: CubicSplineData,
        plot_file: str = "tema6_plot.png",
) -> None:
    grid = np.linspace(x_list[0], x_list[-1], 500)
    real_values = func_item.f(grid)
    poly_values = horner_value(poly.coefs, grid)
    spline_values = spline_value(sp_line, grid)
    plt.figure(figsize=(10, 6))
    plt.plot(grid, real_values, label="f(x)")
    plt.plot(grid, poly_values, label=f"P{poly.poly_degree}(x)")
    plt.plot(grid, spline_values, label="Sf(x)")
    plt.scatter(x_list, y_list, label="interpolation made_nodes")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Tema 6: least-squares poly and clamped cubic sp_line")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_file, dpi=160)
    print(f"\nPlot saved to: {plot_file}")
    plt.close()


def main() -> None:
    func_item = pick_function()
    parts = ask_int("n from the statement; number of intervals, nodes = n + 1", 10)
    poly_degree = ask_int("m for least-squares polynomial, with m < 6", 4)
    seed_no = ask_int("random seed for internal nodes", 12345)
    a = ask_float("x0 = a", func_item.a)
    b = ask_float("xn = b", func_item.b)
    x_query = ask_float("x_query", func_item.x_query)
    if not a <= x_query <= b:
        raise ValueError("x_query must be inside [a, b]")
    x_list = make_nodes(a, b, parts, seed_no=seed_no)
    if np.any(np.isclose(x_list, x_query, rtol=0.0, atol=1e-12)):
        raise ValueError("x_query must not be equal to any interpolation node")
    y_list = np.asarray(func_item.f(x_list), dtype=float)
    poly = make_ls_poly(x_list, y_list, poly_degree)
    da = float(func_item.f_prime(a))
    db = float(func_item.f_prime(b))
    sp_line = make_clamped_spline(x_list, y_list, da, db)
    print_answers(func_item, x_list, y_list, poly, sp_line, x_query)
    draw_plot(func_item, x_list, y_list, poly, sp_line)


if __name__ == "__main__":
    main()
