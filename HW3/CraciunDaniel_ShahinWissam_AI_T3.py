import argparse
import numpy as np


def build_b_from_A_and_s(A, s):
    # Builds b from A and s, as it was asked from the PDF
    A = np.asarray(A, dtype=float)
    s = np.asarray(s, dtype=float)

    n = A.shape[0]
    b = np.zeros(n, dtype=float)

    # Solves item 1 from the PDF
    for i in range(n):
        total = 0.0

        # Continues item 1 from the PDF
        for j in range(n):
            total += s[j] * A[i, j]

        b[i] = total

    return b


# Solves the final upper triangular system in the same order provided by the PDF
def solve_upper_triangular(R, b, eps=1e-12):
    R = np.asarray(R, dtype=float)
    b = np.asarray(b, dtype=float)

    n = R.shape[0]
    x = np.zeros(n, dtype=float)

    # Solves the back substitution part from the pages 2 and 6 of the PDF
    for i in range(n - 1, -1, -1):
        if abs(R[i, i]) < eps:
            raise np.linalg.LinAlgError(
                f"Singular matrix: |R[{i},{i}]| <= eps, so the system cannot be solved."
            )

        known_sum = 0.0

        # Uses the already found values (like in the PDF back substitution step)
        for j in range(i + 1, n):
            known_sum += R[i, j] * x[j]

        x[i] = (b[i] - known_sum) / R[i, i]

    return x


# Goes through the Householder method step by step
def householder_qr(A_init, b_init, eps=1e-12):
    A = np.array(A_init, dtype=float, copy=True)
    b = np.array(b_init, dtype=float, copy=True)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be square.")
    if b.ndim != 1 or b.shape[0] != A.shape[0]:
        raise ValueError("Vector b must have size n.")

    n = A.shape[0]
    Qt = np.eye(n, dtype=float)

    # Follows the Householder steps from pages 2-5 of the PDF
    for r in range(n - 1):
        sigma = 0.0

        # Builds the helper value that is used for this step from the PDF
        for i in range(r, n):
            sigma += A[i, r] * A[i, r]

        if sigma <= eps:
            break

        k = np.sqrt(sigma)
        if A[r,r] > 0:
            k = -k

        beta = sigma - k * A[r, r]

        u = np.zeros(n, dtype=float)
        u[r] = A[r, r] - k

        # Fills the rest of u for the current step provided for from the PDF
        for i in range(r + 1, n):
            u[i] = A[i, r]

        # Updates A for the current Householder step
        for j in range(r, n):
            gamma_num = 0.0

            # Builds the helper value that is used on this column
            for i in range(r, n):
                gamma_num += u[i] * A[i, j]

            gamma = gamma_num / beta

            # Applies the current step to A now
            for i in range(r, n):
                A[i, j] = A[i, j] - gamma * u[i]

        A[r, r] = k

        # Clears the values under the diagonal explanation provided by the PDF
        for i in range(r + 1, n):
            A[i, r] = 0.0

        gamma_num = 0.0

        # Updates b in the same order as the PDF
        for i in range(r, n):
            gamma_num += u[i] * b[i]

        gamma = gamma_num / beta

        # Applies the current step to b now
        for i in range(r, n):
            b[i] = b[i] - gamma * u[i]

        # Updates Q^T in the same order as the PDF
        for j in range(n):
            gamma_num = 0.0

            # Builds the helper value that is used on this column of Q^T
            for i in range(r, n):
                gamma_num += u[i] * Qt[i, j]

            gamma = gamma_num / beta

            # Applies the current step to Q^T
            for i in range(r, n):
                Qt[i, j] = Qt[i, j] - gamma * u[i]

    # Provided by chatGPT as an enchancement to print better, it cleans the small numbers
    for i in range(n):
        for j in range(n):
            if abs(A[i, j]) < eps:
                A[i, j] = 0.0
            if abs(Qt[i, j]) < eps:
                Qt[i, j] = 0.0

    R = np.triu(A)
    Q = Qt.T

    return Q, R, Qt, b


# Checking singularity fully according to the PDF
def is_singular_from_diagonal(R, eps=1e-12):
    R = np.asarray(R, dtype=float)
    n = R.shape[0]

    # Uses the diagonal test from pages 4 and 6 from the PDF
    for i in range(n):
        if abs(R[i, i]) < eps:
            return True

    return False


# Library thing
def library_qr(A):
    Q, R = np.linalg.qr(np.asarray(A, dtype=float))
    return Q, R


def inverse_from_big_H(R, Qt, eps=1e-12):
    # Builds the inverse one column at a time
    R = np.asarray(R, dtype=float)
    Qt = np.asarray(Qt, dtype=float)

    n = R.shape[0]

    if is_singular_from_diagonal(R, eps):
        raise np.linalg.LinAlgError(
            "Singular matrix: the PDF says to stop when some |r_ii| <= eps."
        )

    A_inv = np.zeros((n, n), dtype=float)

    # Solves the inverse part from page 6 of the PDF
    for j in range(n):
        b_column = np.zeros(n, dtype=float)

        # Takes the needed column from Q^T like provided in the PDF
        for i in range(n):
            b_column[i] = Qt[i, j]

        x_column = solve_upper_triangular(R, b_column, eps)

        # Places the result in column j of the inverse
        for i in range(n):
            A_inv[i, j] = x_column[i]

    return A_inv


def generate_random_problem(n, seed=None):
    # radnom data for homework
    rng = np.random.default_rng(seed)

    while True:
        A = rng.standard_normal((n, n))
        A += n * np.eye(n)

        if np.linalg.matrix_rank(A) == n:
            break

    s = rng.standard_normal(n)
    return A, s


def example_problem():
    # The example problem that was at the end of the PDF, purely for testing
    A = np.array([
        [0.0, 0.0, 4.0],
        [1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0],
    ])
    s = np.array([3.0, 2.0, 1.0])
    return A, s


def run_full(A, s, eps=1e-12):
    # Runs the homework in the same main order as the statement
    A_init = np.array(A, dtype=float, copy=True)
    s = np.array(s, dtype=float, copy=True)

    if A_init.ndim != 2 or A_init.shape[0] != A_init.shape[1]:
        raise ValueError("Matrix A must be square.")
    if s.ndim != 1 or s.shape[0] != A_init.shape[0]:
        raise ValueError("Vector s must have size n.")

    # Solves item 1
    b_init = build_b_from_A_and_s(A_init, s)

    # Solves item 2
    Q_house, R_house, Qt_house, b_house = householder_qr(A_init, b_init, eps)
    singular_house = is_singular_from_diagonal(R_house, eps)

    if singular_house:
        raise np.linalg.LinAlgError(
            "Singular matrix: the PDF says to stop when some |r_ii| <= eps after Householder."
        )

    # Solves the Householder answer for item 3
    x_householder = solve_upper_triangular(R_house, b_house, eps)

    # Solves the library answer for item 3
    Q_lib, R_lib = library_qr(A_init)
    b_lib = Q_lib.T @ b_init
    x_qr = solve_upper_triangular(R_lib, b_lib, eps)

    # Solves the comparison between the 2 for the item 3
    diff_x = np.linalg.norm(x_qr - x_householder, 2)

    # Solves the first error for item 4
    err_house_residual = np.linalg.norm(A_init @ x_householder - b_init, 2)

    # Solves the second error for item 4
    err_qr_residual = np.linalg.norm(A_init @ x_qr - b_init, 2)

    # Solves the third error for item 4
    err_house_relative = np.linalg.norm(x_householder - s, 2) / np.linalg.norm(s, 2)

    # Solves the fourth error for item 4
    err_qr_relative = np.linalg.norm(x_qr - s, 2) / np.linalg.norm(s, 2)

    # Solves item 5
    A_inv_house = inverse_from_big_H(R_house, Qt_house, eps)
    A_inv_lib = np.linalg.inv(A_init)

    return {
        "A": A_init,
        "s": s,
        "b": b_init,
        "Q_house": Q_house,
        "R_house": R_house,
        "Qt_house": Qt_house,
        "b_house": b_house,
        "Q_lib": Q_lib,
        "R_lib": R_lib,
        "b_lib": b_lib,
        "x_householder": x_householder,
        "x_qr": x_qr,
        "diff_x": diff_x,
        "err_house_residual": err_house_residual,
        "err_qr_residual": err_qr_residual,
        "err_house_relative": err_house_relative,
        "err_qr_relative": err_qr_relative,
        "A_inv_house": A_inv_house,
        "A_inv_lib": A_inv_lib,
        "singular_house": singular_house,
    }


def print_results(results, eps):
    # Prints everything clearly, so it is easier to compare with the homework items.
    np.set_printoptions(precision=10, suppress=True)

    print("=== Input data ===")
    print("A =\n", results["A"])
    print("s =\n", results["s"])
    print("eps =", eps)
    print()

    print("1. Vector b built from A and s")
    print("b =\n", results["b"])
    print()

    print("2. Householder QR decomposition")
    print("Q_house =\n", results["Q_house"])
    print("R_house =\n", results["R_house"])
    print("Q^T b from the Householder steps =\n", results["b_house"])
    print("Singular according to the diagonal test from the PDF =", results["singular_house"])
    print("Verification ||A - Q_house @ R_house||_2 =",
          np.linalg.norm(results["A"] - results["Q_house"] @ results["R_house"], 2))
    print()

    print("3. Solving Ax = b")
    print("x_qr =\n", results["x_qr"])
    print("x_householder =\n", results["x_householder"])
    print("||x_qr - x_householder||_2 =", results["diff_x"])
    print()

    print("4. Required errors")
    print("||A_init * x_householder - b_init||_2 =", results["err_house_residual"])
    print("||A_init * x_qr - b_init||_2 =", results["err_qr_residual"])
    print("||x_householder - s||_2 / ||s||_2 =", results["err_house_relative"])
    print("||x_qr - s||_2 / ||s||_2 =", results["err_qr_relative"])
    print("Reference threshold from the PDF = 1e-6")
    print()

    print("5. Inverse matrix")
    print("A_inv_householder =\n", results["A_inv_house"])
    print("A_inv_library =\n", results["A_inv_lib"])
    print()
    print(np.linalg.norm(results["A_inv_house"] - results["A_inv_lib"],2))


def main():
    parser = argparse.ArgumentParser(description="Homework 3 - QR with Householder")
    parser.add_argument("--n", type=int, default=3, help="Size n used in random mode")
    parser.add_argument("--eps", type=float, default=1e-12, help="Computation tolerance")
    parser.add_argument("--random", action="store_true", help="Use random input instead of the small example")
    parser.add_argument("--seed", type=int, default=None, help="Seed for the random generator")
    args = parser.parse_args()

    if args.random:
        A, s = generate_random_problem(args.n, args.seed)
    else:
        A, s = example_problem()

    results = run_full(A, s, args.eps)
    print_results(results, args.eps)


if __name__ == "__main__":
    main()