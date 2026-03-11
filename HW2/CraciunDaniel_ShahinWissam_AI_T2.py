import numpy as np
from scipy.linalg import lu, lu_factor, lu_solve

# A = BB^T
def generate_matrix(n: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    B = rng.random((n, n))
    B += n * np.eye(n) #This helps with numerical stability, helps with large n curtesy of ChatGPT
    return B @ B.T

# Ax = b, basically for test data, no provided info was on this.
def generate_vector(n: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(n)


def ldlt_inplace(A: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    A = np.asarray(A, dtype=float)

    n, m = A.shape
    if n != m:
        raise ValueError("Matrix must be square")
    if eps <= 0:
        raise ValueError("eps must be > 0")
    if not np.allclose(A, A.T, atol=eps, rtol=0.0):
        raise ValueError("Matrix must be symmetric")

    d = np.zeros(n, dtype=float) # Diagonal

    for p in range(n):
        s = 0.0
        # Computing D_p, using the provided formula from the pdf (9)
        for k in range(p):
            s += d[k] * A[p, k] * A[p, k]

        dp = A[p, p] - s
        if abs(dp) <= eps:
            raise ValueError(f"LDLT failed at step {p}: pivot too small")

        d[p] = dp
        # Computing l_ip, using the provided fomula from the pdf (10)
        for i in range(p + 1, n):
            s = 0.0
            for k in range(p):
                s += d[k] * A[i, k] * A[p, k]

            A[i, p] = (A[i, p] - s) / dp

    return d

# Solves L_z = b
def forward_substitution_unit_lower(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    z = np.zeros(n, dtype=float)
    # Uses the PDF formula (4), l_ii is equal to w1, so not divided by it.
    for i in range(n):
        s = 0.0
        for j in range(i):
            s += A[i, j] * z[j]
        z[i] = b[i] - s

    return z

# Solved D_y = z
def diagonal_substitution(d: np.ndarray, z: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    n = len(d)
    y = np.zeros(n, dtype=float)
    # Purely the y_i = z_i/d_i
    for i in range(n):
        if abs(d[i]) <= eps:
            raise ValueError(f"Zero diagonal element in D at index {i}")
        y[i] = z[i] / d[i]

    return y

#Solves L^T*x = y
def backward_substitution_unit_upper_from_lower(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    x = np.zeros(n, dtype=float)
    #Solves similar to Forward, however for the lower triangular half, not the upper.
    for i in range(n - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, n):
            s += A[j, i] * x[j]
        x[i] = y[i] - s

    return x

# The main 3 operations, in the part 2 -> x_chol
def solve_with_ldlt_storage(A: np.ndarray, d: np.ndarray, b: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    z = forward_substitution_unit_lower(A, b)
    y = diagonal_substitution(d, z, eps)
    x = backward_substitution_unit_upper_from_lower(A, y)
    return x

# This mainly solves the Det(A) = Det(L)*Det(D)*Det(L^T), but as L and L^T both have their diagonals as 1, this means we just need
# Det(D).
def determinant_from_ldlt(d: np.ndarray) -> float:
    det_a = 1.0
    for value in d:
        det_a *= value
    return float(det_a)

# Abusing symmetry, we can make the lower half, by using the upper half, as we know a_ij = a_ji, thats why we do the j < i, we use the a[j,i[
def matvec_original_from_ldlt_storage(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    y = np.zeros(n, dtype=float)

    for i in range(n):
        s = 0.0
        for j in range(i):
            s += A[j, i] * x[j]
        s += A[i, i] * x[i]
        for j in range(i + 1, n):
            s += A[i, j] * x[j]
        y[i] = s

    return y


def solve_assignment(A: np.ndarray, b: np.ndarray, eps: float = 1e-10) -> dict:
    A_init = np.array(A, dtype=float, copy=True)
    b = np.array(b, dtype=float, copy=True).reshape(-1)

    n, m = A_init.shape
    if n != m:
        raise ValueError("Matrix must be square")
    if b.shape[0] != n:
        raise ValueError("Vector b must have length n")

    P, L_lu, U_lu = lu(A_init)
    lu_fac, piv = lu_factor(A_init)
    x_lib = lu_solve((lu_fac, piv), b)

    A_ldlt = A_init.copy()
    d = ldlt_inplace(A_ldlt, eps)

    x_chol = solve_with_ldlt_storage(A_ldlt, d, b, eps)

    residual = matvec_original_from_ldlt_storage(A_ldlt, x_chol) - b
    residual_norm = np.linalg.norm(residual, ord=2)
    solution_diff_norm = np.linalg.norm(x_chol - x_lib, ord=2)

    return {
        "P": P,
        "L_lu": L_lu,
        "U_lu": U_lu,
        "A_init": A_init,
        "A_ldlt": A_ldlt,
        "d": d,
        "b": b,
        "x_lib": x_lib,
        "x_chol": x_chol,
        "det_A": determinant_from_ldlt(d),
        "residual_norm": residual_norm,
        "solution_diff_norm": solution_diff_norm,
    }

# GPT provided prints:
def print_result(result: dict) -> None:
    np.set_printoptions(precision=10, suppress=True, linewidth=200)

    print("=== A_init ===")
    print("A_init =")
    print(result["A_init"])
    print("b =")
    print(result["b"])
    print()

    print("=== LU from library ===")
    print("P =")
    print(result["P"])
    print("L =")
    print(result["L_lu"])
    print("U =")
    print(result["U_lu"])
    print()

    print("=== LDLT in packed storage ===")
    print("A after LDLT (strict lower triangle = L, upper triangle + diagonal = A_init):")
    print(result["A_ldlt"])
    print("d =")
    print(result["d"])
    print()

    print("=== Solutions ===")
    print("x_lib =")
    print(result["x_lib"])
    print("x_chol =")
    print(result["x_chol"])
    print()

    print("=== Determinant ===")
    print("det(A) =", result["det_A"])
    print()

    print("=== Verification ===")
    print("||A_init x_chol - b||_2 =", result["residual_norm"])
    print("||x_chol - x_lib||_2   =", result["solution_diff_norm"])
    print("Residual < 1e-8 :", result["residual_norm"] < 1e-8)
    print("Diff < 1e-9     :", result["solution_diff_norm"] < 1e-9)

def main() -> None:
    n = int(input("n = ").strip())
    t = int(input("t for eps = 10^(-t), t = ").strip())
    eps = 10.0 ** (-t)

    A = generate_matrix(n, seed=12345)
    b = generate_vector(n, seed=54321)

    result = solve_assignment(A, b, eps)
    print_result(result)


if __name__ == "__main__":
    main()