# Shahin, Wissam, 310910401ESL231073, Wissam.shahin05@gmail.com, shukakah
# Craciun, Daniel, 310910401ESL231020, danielcraciun72@gmail.com, donnavant
# Estimated AI-assisted portion: 35% (prints + refinements)
# Bibliography:
# Chatgpt -> Mainly questions to ask about how to think of a solution + solution to some exercises to simulate a way of solving for the algorithm
# PDF for formulas, psuedo code snippets and explanations
import argparse
import numpy as np


def show_report(bundle, tol):
    # Prints everything clearly, so it is easier to compare with the homework items.
    np.set_printoptions(precision=10, suppress=True)
    print("=== Input data ===")
    print("A =\n", bundle["A"])
    print("s =\n", bundle["s"])
    print("eps =", tol)
    print()
    print("1. Vector b built from A and s")
    print("b =\n", bundle["b"])
    print()
    print("2. Householder QR decomposition")
    print("Q_house =\n", bundle["Q_house"])
    print("R_house =\n", bundle["R_house"])
    print("Q^T b from the Householder steps =\n", bundle["b_house"])
    print("Singular according to the diagonal test from the PDF =", bundle["singular_house"])
    print("Verification ||A - Q_house @ R_house||_2 =",
          np.linalg.norm(bundle["A"] - bundle["Q_house"] @ bundle["R_house"], 2))
    print()
    print("3. Solving Ax = b")
    print("x_qr =\n", bundle["x_qr"])
    print("x_householder =\n", bundle["x_householder"])
    print("||x_qr - x_householder||_2 =", bundle["diff_x"])
    print()
    print("4. Required errors")
    print("||A_init * x_householder - b_init||_2 =", bundle["err_house_residual"])
    print("||A_init * x_qr - b_init||_2 =", bundle["err_qr_residual"])
    print("||x_householder - s||_2 / ||s||_2 =", bundle["err_house_relative"])
    print("||x_qr - s||_2 / ||s||_2 =", bundle["err_qr_relative"])
    print("Reference threshold from the PDF = 1e-6")
    print()
    print("5. Inverse matrix")
    print("A_inv_householder =\n", bundle["A_inv_house"])
    print("A_inv_library =\n", bundle["A_inv_lib"])
    print()
    print(np.linalg.norm(bundle["A_inv_house"] - bundle["A_inv_lib"], 2))


def sample_case():
    # The example problem that was at the end of the PDF, purely for testing
    mat = np.array([
        [0.0, 0.0, 4.0],
        [1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0],
    ])
    vec = np.array([3.0, 2.0, 1.0])
    return mat, vec


def make_inverse_from_qr(upper, q_transpose, tol=1e-12):
    # Builds the inverse one column at a time
    upper = np.asarray(upper, dtype=float)
    q_transpose = np.asarray(q_transpose, dtype=float)
    count = upper.shape[0]
    if diagonal_is_singular(upper, tol):
        raise np.linalg.LinAlgError("Singular matrix: the PDF says to stop when some |r_ii| <= eps.")
    inverse = np.zeros((count, count), dtype=float)
    # Solves the inverse part from page 6 of the PDF
    for col in range(count):
        rhs_col = np.zeros(count, dtype=float)
        # Takes the needed column from Q^T like provided in the PDF
        for row in range(count):
            rhs_col[row] = q_transpose[row, col]
        sol_col = solve_upper_system(upper, rhs_col, tol)
        # Places the result in column j of the inverse
        for row in range(count):
            inverse[row, col] = sol_col[row]
    return inverse


# Library thing
def builtin_qr(mat):
    left, upper = np.linalg.qr(np.asarray(mat, dtype=float))
    return left, upper


# Checking singularity fully according to the PDF
def diagonal_is_singular(upper, tol=1e-12):
    upper = np.asarray(upper, dtype=float)
    count = upper.shape[0]
    # Uses the diagonal test from pages 4 and 6 from the PDF
    for idx in range(count):
        if abs(upper[idx, idx]) < tol:
            return True
    return False


# Solves the final upper triangular system in the same order provided by the PDF
def solve_upper_system(upper, rhs, tol=1e-12):
    upper = np.asarray(upper, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    count = upper.shape[0]
    sol = np.zeros(count, dtype=float)
    # Solves the back substitution part from the pages 2 and 6 of the PDF
    for row in range(count - 1, -1, -1):
        if abs(upper[row, row]) < tol:
            raise np.linalg.LinAlgError(f"Singular matrix: |R[{row},{row}]| <= eps, so the system cannot be solved.")
        acc = 0.0
        # Uses the already found values (like in the PDF back substitution step)
        for col in range(row + 1, count):
            acc += upper[row, col] * sol[col]
        sol[row] = (rhs[row] - acc) / upper[row, row]
    return sol


# Runs the homework in the same main order as the statement
def execute_all(mat, vec, tol=1e-12):
    mat_copy = np.array(mat, dtype=float, copy=True)
    vec = np.array(vec, dtype=float, copy=True)
    if mat_copy.ndim != 2 or mat_copy.shape[0] != mat_copy.shape[1]:
        raise ValueError("Matrix A must be square.")
    if vec.ndim != 1 or vec.shape[0] != mat_copy.shape[0]:
        raise ValueError("Vector s must have size n.")
    # Solves item 1
    rhs = assemble_b(mat_copy, vec)
    # Solves item 2
    house_q, house_r, house_qt, house_rhs = householder_steps(mat_copy, rhs, tol)
    flagged = diagonal_is_singular(house_r, tol)
    if flagged:
        raise np.linalg.LinAlgError("Singular matrix: the PDF says to stop when some |r_ii| <= eps after Householder.")
    # Solves the Householder answer for item 3
    via_householder = solve_upper_system(house_r, house_rhs, tol)
    # Solves the library answer for item 3
    lib_q, lib_r = builtin_qr(mat_copy)
    lib_rhs = lib_q.T @ rhs
    via_qr = solve_upper_system(lib_r, lib_rhs, tol)
    # Solves the comparison between the 2 for the item 3
    delta_x = np.linalg.norm(via_qr - via_householder, 2)
    # Solves the first error for item 4
    err_house_residual = np.linalg.norm(mat_copy @ via_householder - rhs, 2)
    # Solves the second error for item 4
    err_qr_residual = np.linalg.norm(mat_copy @ via_qr - rhs, 2)
    # Solves the third error for item 4
    err_house_relative = np.linalg.norm(via_householder - vec, 2) / np.linalg.norm(vec, 2)
    # Solves the fourth error for item 4
    err_qr_relative = np.linalg.norm(via_qr - vec, 2) / np.linalg.norm(vec, 2)
    # Solves item 5
    inv_house = make_inverse_from_qr(house_r, house_qt, tol)
    inv_lib = np.linalg.inv(mat_copy)
    return {
        "A": mat_copy,
        "s": vec,
        "b": rhs,
        "Q_house": house_q,
        "R_house": house_r,
        "Qt_house": house_qt,
        "b_house": house_rhs,
        "Q_lib": lib_q,
        "R_lib": lib_r,
        "b_lib": lib_rhs,
        "x_householder": via_householder,
        "x_qr": via_qr,
        "diff_x": delta_x,
        "err_house_residual": err_house_residual,
        "err_qr_residual": err_qr_residual,
        "err_house_relative": err_house_relative,
        "err_qr_relative": err_qr_relative,
        "A_inv_house": inv_house,
        "A_inv_lib": inv_lib,
        "singular_house": flagged,
    }


def assemble_b(mat, vec):
    # Builds b from A and s, as it was asked from the PDF
    mat = np.asarray(mat, dtype=float)
    vec = np.asarray(vec, dtype=float)
    count = mat.shape[0]
    rhs = np.zeros(count, dtype=float)
    # Solves item 1 from the PDF
    for row in range(count):
        acc = 0.0
        # Continues item 1 from the PDF
        for col in range(count):
            acc += vec[col] * mat[row, col]
        rhs[row] = acc
    return rhs


# Goes through the Householder method step by step
def householder_steps(source_mat, source_rhs, tol=1e-12):
    mat = np.array(source_mat, dtype=float, copy=True)
    rhs = np.array(source_rhs, dtype=float, copy=True)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("Matrix A must be square.")
    if rhs.ndim != 1 or rhs.shape[0] != mat.shape[0]:
        raise ValueError("Vector b must have size n.")
    count = mat.shape[0]
    q_transpose = np.eye(count, dtype=float)
    # Follows the Householder steps from pages 2-5 of the PDF
    for pivot in range(count - 1):
        sigma = 0.0
        # Builds the helper value that is used for this step from the PDF
        for row in range(pivot, count):
            sigma += mat[row, pivot] * mat[row, pivot]
        if sigma <= tol:
            break
        lead = np.sqrt(sigma)
        if mat[pivot, pivot] > 0:
            lead = -lead
        beta = sigma - lead * mat[pivot, pivot]
        reflector = np.zeros(count, dtype=float)
        reflector[pivot] = mat[pivot, pivot] - lead
        # Fills the rest of u for the current step provided for from the PDF
        for row in range(pivot + 1, count):
            reflector[row] = mat[row, pivot]
        # Updates A for the current Householder step
        for col in range(pivot, count):
            gamma_num = 0.0
            # Builds the helper value that is used on this column
            for row in range(pivot, count):
                gamma_num += reflector[row] * mat[row, col]
            gamma = gamma_num / beta
            # Applies the current step to A now
            for row in range(pivot, count):
                mat[row, col] = mat[row, col] - gamma * reflector[row]
        mat[pivot, pivot] = lead
        # Clears the values under the diagonal explanation provided by the PDF
        for row in range(pivot + 1, count):
            mat[row, pivot] = 0.0
        gamma_num = 0.0
        # Updates b in the same order as the PDF
        for row in range(pivot, count):
            gamma_num += reflector[row] * rhs[row]
        gamma = gamma_num / beta
        # Applies the current step to b now
        for row in range(pivot, count):
            rhs[row] = rhs[row] - gamma * reflector[row]
        # Updates Q^T in the same order as the PDF
        for col in range(count):
            gamma_num = 0.0
            # Builds the helper value that is used on this column of Q^T
            for row in range(pivot, count):
                gamma_num += reflector[row] * q_transpose[row, col]
            gamma = gamma_num / beta
            # Applies the current step to Q^T
            for row in range(pivot, count):
                q_transpose[row, col] = q_transpose[row, col] - gamma * reflector[row]
    # Provided by chatGPT as an enchancement to print better, it cleans the small numbers
    for row in range(count):
        for col in range(count):
            if abs(mat[row, col]) < tol:
                mat[row, col] = 0.0
            if abs(q_transpose[row, col]) < tol:
                q_transpose[row, col] = 0.0
    upper = np.triu(mat)
    q_mat = q_transpose.T
    return q_mat, upper, q_transpose, rhs


def build_random_case(size, seed=None):
    # radnom data for homework
    rng = np.random.default_rng(seed)
    while True:
        mat = rng.standard_normal((size, size))
        mat += size * np.eye(size)
        if np.linalg.matrix_rank(mat) == size:
            break
    vec = rng.standard_normal(size)
    return mat, vec


def main():
    parser = argparse.ArgumentParser(description="Homework 3 - QR with Householder")
    parser.add_argument("--n", type=int, default=3, help="Size n used in random mode")
    parser.add_argument("--eps", type=float, default=1e-12, help="Computation tolerance")
    parser.add_argument("--random", action="store_true", help="Use random input instead of the small example")
    parser.add_argument("--seed", type=int, default=None, help="Seed for the random generator")
    parsed = parser.parse_args()
    if parsed.random:
        mat, vec = build_random_case(parsed.n, parsed.seed)
    else:
        mat, vec = sample_case()
    report = execute_all(mat, vec, parsed.eps)
    show_report(report, parsed.eps)


if __name__ == "__main__":
    main()
