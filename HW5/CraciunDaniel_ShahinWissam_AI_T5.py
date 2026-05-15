# Shahin, Wissam, 310910401ESL231073, Wissam.shahin05@gmail.com, shukakah
# Craciun, Daniel, 310910401ESL231020, danielcraciun72@gmail.com, donnavant
# Estimated AI-assisted portion: 35% (questions, print/output wording, and small refinements)
# Bibliography:
# GPT: used mainly for questions about how to approach the solution and for checking/refining explanations/prints
# Course PDF: main source for formulas, pseudo-code, algorithms, and homework requirements

import numpy as np

SHOW_MORE = False

def eps_from_t(power: int) -> float:
    return 10.0 ** (-power)

# For p = n we generate a symmetric positive definite matrix because Jacobi needs symmetry, and the Cholesky sequence needs the factorization
def make_square_A(size: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(size, size))
    ready = raw @ raw.T
    ready += size * np.eye(size)
    return ready

# For p > n the homework only asks for a rectangular matrix, so this is good enough, it makes sure with this small diagonal shift that the first n rows
# help A^T A can stay invertible
def make_tall_A(rows: int, cols: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ready = rng.normal(size=(rows, cols))
    for i in range(cols):
        ready[i, i] += cols
    return ready

# This function just comes from basically homework 2 implementation
def ldlt_in_place(work: np.ndarray, eps: float) -> np.ndarray:
    work = np.asarray(work, dtype=float)
    size = work.shape[0]
    d_values = np.zeros(size, dtype=float)
    for col in range(size):
        carry = 0.0
        # Current Diagonal Term the d_p
        for k in range(col):
            carry += d_values[k] * work[col, k] * work[col, k]
        d_now = work[col, col] - carry
        if abs(d_now) <= eps:
            raise ValueError("Factorization stopped because a diagonal value became too small.")
        if d_now < 0.0:
            raise ValueError("Factorization stopped because the matrix is not positive definite.")
        d_values[col] = d_now
        # Lower Entries the l_ip
        for row in range(col + 1, size):
            carry = 0.0
            for k in range(col):
                carry += d_values[k] * work[row, k] * work[col, k]
            work[row, col] = (work[row, col] - carry) / d_now
    return d_values

# From the Homework 2 above function we rebuild the Cholesky lower factor that Homework 5 needs
def get_cholesky_l(source: np.ndarray, eps: float) -> np.ndarray:
    packed = np.array(source, dtype=float, copy=True)
    d_values = ldlt_in_place(packed, eps)
    size = packed.shape[0]
    lower = np.zeros((size, size), dtype=float)
    for i in range(size):
        lower[i, i] = np.sqrt(d_values[i])
        for j in range(i):
            # This actuallt makes the diagonal for the Cholesky factor
            lower[i, j] = packed[i, j] * np.sqrt(d_values[j])
    return lower

# It searches in the matrix for the off-diagonal entry with the largest absolute valuie (ONLY IN LOWER TRIANGULAR PART)
def biggest_offdiag(board: np.ndarray) -> tuple[float, int, int]:
    size = board.shape[0]
    best = 0.0
    found_i = 0
    found_j = 0
    for i in range(1, size):
        for j in range(i):
            seen = abs(board[i, j])
            if seen > best:
                best = seen
                found_i = i
                found_j = j
    # Best should become lower to a point it goes under the eps
    return best, found_i, found_j

# This runs the jacobi method ONLY for the square (so basically p = n)
def run_jacobi(start_matrix: np.ndarray, eps: float, limit: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    live = np.array(start_matrix, dtype=float, copy=True)
    size = live.shape[0]
    # U=I_n
    vectors = np.eye(size, dtype=float)
    turns = 0
    biggest, p, q = biggest_offdiag(live)
    # Continue so long the matrix is not diagonal enough
    while biggest > eps and turns <= limit:
        app = live[p, p]
        aqq = live[q, q]
        apq = live[p, q]
        #Rotation Parameters, given from the PDF of homework 5
        alpha = (app - aqq) / (2.0 * apq)
        sign_alpha = 1.0 if alpha >= 0.0 else -1.0
        tangent = -alpha + sign_alpha * np.sqrt(alpha * alpha + 1.0)
        cosine = 1.0 / np.sqrt(1.0 + tangent * tangent)
        sine = tangent * cosine
        # Updates the working matrix per loop. The jacobi update formula
        for j in range(size):
            if j == p or j == q:
                continue
            old_pj = live[p, j]
            old_qj = live[q, j]
            live[p, j] = cosine * old_pj + sine * old_qj
            live[j, p] = live[p, j]
            live[q, j] = -sine * old_pj + cosine * old_qj
            live[j, q] = live[q, j]
        live[p, p] = app + tangent * apq
        live[q, q] = aqq - tangent * apq
        live[p, q] = 0.0
        live[q, p] = 0.0
        # This updates the eigenvector Matrix, basically turns rotations into vectors which becomes the matrix U
        for i in range(size):
            old_ip = vectors[i, p]
            old_iq = vectors[i, q]
            vectors[i, p] = cosine * old_ip + sine * old_iq
            vectors[i, q] = -sine * old_ip + cosine * old_iq
        turns += 1
        biggest, p, q = biggest_offdiag(live)
    return np.diag(live).copy(), vectors, live, turns

# this does the given formula from the pdf verification (||A_init@U - U @ lamba||)
def jacobi_residual(start_A: np.ndarray, lambdas: np.ndarray, vectors: np.ndarray) -> float:
    lambda_matrix = np.diag(lambdas)
    # We want it to be small
    return float(np.linalg.norm(start_A @ vectors - vectors @ lambda_matrix))

# This is the second square-case sequence
def run_cholesky_sequence(start_matrix: np.ndarray, eps: float, limit: int) -> tuple[np.ndarray, int]:
    last_A = np.array(start_matrix, dtype=float, copy=True)
    # Each step counts a lower factor basically last_A = lower * lower^T
    for step_no in range(1, limit + 1):
        lower = get_cholesky_l(last_A, eps)
        # A^(k+1) = L^T_k L_k
        next_A = lower.T @ lower
        # PDF-given stopping condition
        if np.linalg.norm(next_A - last_A) < eps:
            return next_A, step_no
        last_A = next_A
    return last_A, limit

# This computes onlyS what the statement asks for in the p > n branch.
def svd_values(tall_A: np.ndarray, eps: float) -> dict:
    # A = USV^T
    u_left, sigma, v_right_t = np.linalg.svd(tall_A, full_matrices=True)
    rows, cols = tall_A.shape
    rank_by_formula = int(np.sum(sigma > eps))
    positive_sigmas = sigma[sigma > eps]
    if positive_sigmas.size == 0:
        cond_by_formula = np.inf
    else:
        #PDF formula of k_2(A)
        cond_by_formula = float(np.max(sigma) / np.min(positive_sigmas))
    # Library ones for comparison
    rank_from_lib = int(np.linalg.matrix_rank(tall_A))
    cond_from_lib = float(np.linalg.cond(tall_A))
    # Moore-Penrose pseudoinverse basically the S is built by inverting the positive SINGULAR values
    s_inv = np.zeros((cols, rows), dtype=float)
    for i in range(rank_by_formula):
        s_inv[i, i] = 1.0 / sigma[i]
    v_right = v_right_t.T
    # A^I = V S^I U^T
    pinv_moore = v_right @ s_inv @ u_left.T
    #A^T A
    gram = tall_A.T @ tall_A
    # A^J
    pinv_normal = np.linalg.inv(gram) @ tall_A.T
    # if invertible low, if not, high.
    diff_norm_1 = float(np.linalg.norm(pinv_moore - pinv_normal, 1))
    return {
        "sigma": sigma,
        "rank_by_formula": rank_by_formula,
        "rank_from_lib": rank_from_lib,
        "cond_by_formula": cond_by_formula,
        "cond_from_lib": cond_from_lib,
        "pinv_moore": pinv_moore,
        "pinv_normal": pinv_normal,
        "diff_norm_1": diff_norm_1,
        "u_left": u_left,
        "v_right_t": v_right_t,
    }

# Generated by AI: full print block for the p = n output formatting.
# These are the homework prints for p = n. -- AI generated comments
def print_square_output(start_A: np.ndarray,
                     lambdas: np.ndarray,
                     vectors: np.ndarray,
                     check_norm: float,
                     final_board: np.ndarray,
                     jacobi_steps: int,
                     cholesky_steps: int) -> None:
    print()
    print("=== Case p = n ===")
    print("A_init =")
    print(start_A)
    print()
    print("Approximate eigenvalues =")
    print(lambdas)
    print()
    print("Approximate eigenvectors (columns of U) =")
    print(vectors)
    print()
    print("||A_init * U - U * Lambda|| =")
    print(check_norm)
    print()
    print("Last computed matrix from the sequence A(k) =")
    print(final_board)
    print()
    print("Form of the last matrix:")
    print("The matrix is approximately diagonal.")
    print()
    print("What can be found in the last matrix:")
    print("Its diagonal values approximate the eigenvalues of the initial matrix.")
    print()
    if SHOW_MORE:
        print("=== Extra ===")
        print("Jacobi iterations =", jacobi_steps)
        print("Cholesky iteration steps =", cholesky_steps)
        print("||U^T U - I|| =")
        print(np.linalg.norm(vectors.T @ vectors - np.eye(vectors.shape[0])))
        print()

# Generated by AI: full print block for the p > n output formatting.
# These are the homework prints for p > n. -- AI generated comments
def print_tall_output(tall_A: np.ndarray, pack: dict) -> None:
    print()
    print("=== Case p > n ===")
    print("A =")
    print(tall_A)
    print()
    print("Singular values of matrix A =")
    print(pack["sigma"])
    print()
    print("Rank of matrix A (implemented formula) =")
    print(pack["rank_by_formula"])
    print()
    print("Rank of matrix A (library) =")
    print(pack["rank_from_lib"])
    print()
    print("Conditioning number of matrix A (implemented formula) =")
    print(pack["cond_by_formula"])
    print()
    print("Conditioning number of matrix A (library) =")
    print(pack["cond_from_lib"])
    print()
    print("Moore-Penrose pseudo-inverse A^I =")
    print(pack["pinv_moore"])
    print()
    print("Least squares pseudo-inverse A^J =")
    print(pack["pinv_normal"])
    print()
    print("||A^I - A^J||_1 =")
    print(pack["diff_norm_1"])
    print()
    if SHOW_MORE:
        print("=== Extra ===")
        print("U from SVD =")
        print(pack["u_left"])
        print()
        print("V^T from SVD =")
        print(pack["v_right_t"])
        print()

# This is the full Tema 5 run.
def main() -> None:
    np.set_printoptions(precision=10, suppress=True, linewidth=220)
    p = int(input("p = ").strip())
    n = int(input("n = ").strip())
    power = int(input("t for eps = 10^(-t), t = ").strip())
    eps = eps_from_t(power)
    limit = int(input("kmax = ").strip())
    if p < n:
        raise ValueError("The statement covers only p = n and p > n, so p must be >= n.")
    seed_raw = input("random seed (press Enter for default seed) = ").strip()
    if seed_raw == "":
        seed_no = 12345
    else:
        seed_no = int(seed_raw)
    # Square Case p = n
    if p == n:
        start_A = make_square_A(n, seed_no)
        lambdas, vectors, _, jacobi_steps = run_jacobi(start_A, eps, limit)
        check_norm = jacobi_residual(start_A, lambdas, vectors)
        final_board, cholesky_steps = run_cholesky_sequence(start_A, eps, limit)
        print_square_output(
            start_A=start_A,
            lambdas=lambdas,
            vectors=vectors,
            check_norm=check_norm,
            final_board=final_board,
            jacobi_steps=jacobi_steps,
            cholesky_steps=cholesky_steps,
        )
    # Tall branch (p > n)
    else:
        tall_A = make_tall_A(p, n, seed_no)
        pack = svd_values(tall_A, eps)
        print_tall_output(tall_A, pack)

if __name__ == "__main__":
    main()
