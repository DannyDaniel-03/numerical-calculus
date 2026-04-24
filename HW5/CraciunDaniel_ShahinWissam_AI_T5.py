import numpy as np

SHOW_EXTRA = False

def make_eps(power: int) -> float:
    return 10.0 ** (-power)

# For p = n we generate a symmetric positive definite matrix because Jacobi needs symmetry, and the Cholesky sequence needs the factorization
def build_square_matrix(size: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(size, size))
    ready = raw @ raw.T
    ready += size * np.eye(size)
    return ready

# For p > n the homework only asks for a rectangular matrix, so this is good enough, it makes sure with this small diagonal shift that the first n rows
# help A^T A can stay invertible
def build_tall_matrix(rows: int, cols: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ready = rng.normal(size=(rows, cols))
    for i in range(cols):
        ready[i, i] += cols
    return ready

# This function just comes from basically homework 2 implementation
def ldlt_steps_in_place(work: np.ndarray, eps: float) -> np.ndarray:
    work = np.asarray(work, dtype=float)
    size = work.shape[0]
    diag_bucket = np.zeros(size, dtype=float)

    for col in range(size):
        carry = 0.0
        # Current Diagonal Term the d_p
        for k in range(col):
            carry += diag_bucket[k] * work[col, k] * work[col, k]

        diag_now = work[col, col] - carry
        if abs(diag_now) <= eps:
            raise ValueError("Factorization stopped because a diagonal value became too small.")
        if diag_now < 0.0:
            raise ValueError("Factorization stopped because the matrix is not positive definite.")

        diag_bucket[col] = diag_now

        # Lower Entries the l_ip
        for row in range(col + 1, size):
            carry = 0.0
            for k in range(col):
                carry += diag_bucket[k] * work[row, k] * work[col, k]

            work[row, col] = (work[row, col] - carry) / diag_now

    return diag_bucket

# From the Homework 2 above function we rebuild the Cholesky lower factor that Homework 5 needs
def recover_cholesky_lower(source: np.ndarray, eps: float) -> np.ndarray:
    packed = np.array(source, dtype=float, copy=True)
    diag_bucket = ldlt_steps_in_place(packed, eps)
    size = packed.shape[0]

    lower = np.zeros((size, size), dtype=float)
    for i in range(size):
        lower[i, i] = np.sqrt(diag_bucket[i])
        for j in range(i):
            # This actuallt makes the diagonal for the Cholesky factor
            lower[i, j] = packed[i, j] * np.sqrt(diag_bucket[j])

    return lower

# It searches in the matrix for the off-diagonal entry with the largest absolute valuie (ONLY IN LOWER TRIANGULAR PART)
def pick_biggest_outside_diagonal(board: np.ndarray) -> tuple[float, int, int]:
    size = board.shape[0]
    best = 0.0
    where_i = 0
    where_j = 0

    for i in range(1, size):
        for j in range(i):
            seen = abs(board[i, j])
            if seen > best:
                best = seen
                where_i = i
                where_j = j

    # Best should become lower to a point it goes under the eps
    return best, where_i, where_j

# This runs the jacobi method ONLY for the square (so basically p = n)
def jacobi_run(start_matrix: np.ndarray, eps: float, limit: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    live = np.array(start_matrix, dtype=float, copy=True)
    size = live.shape[0]
    # U=I_n
    vectors = np.eye(size, dtype=float)

    turns = 0
    biggest, p, q = pick_biggest_outside_diagonal(live)

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
        biggest, p, q = pick_biggest_outside_diagonal(live)

    return np.diag(live).copy(), vectors, live, turns

# this does the given formula from the pdf verification (||A_init@U - U @ lamba||)
def check_jacobi_relation(InitialA: np.ndarray, lambdas: np.ndarray, vectors: np.ndarray) -> float:
    Lambda = np.diag(lambdas)
    # We want it to be small
    return float(np.linalg.norm(InitialA @ vectors - vectors @ Lambda))

# This is the second square-case sequence
def cholesky_loop(start_matrix: np.ndarray, eps: float, limit: int) -> tuple[np.ndarray, int]:
    old_board = np.array(start_matrix, dtype=float, copy=True)

    # Each step counts a lower factor basically old_board = lower * lower^T
    for step_count in range(1, limit + 1):
        lower = recover_cholesky_lower(old_board, eps)
        # A^(k+1) = L^T_k L_k
        new_board = lower.T @ lower

        # PDF-given stopping condition
        if np.linalg.norm(new_board - old_board) < eps:
            return new_board, step_count

        old_board = new_board

    return old_board, limit

# This computes onlyS what the statement asks for in the p > n branch.
def svd_branch_values(RectA: np.ndarray, eps: float) -> dict:
    # A = USV^T
    LeftU, sigma, RightVt = np.linalg.svd(RectA, full_matrices=True)
    rows, cols = RectA.shape

    hand_rank = int(np.sum(sigma > eps))
    only_positive = sigma[sigma > eps]

    if only_positive.size == 0:
        hand_cond = np.inf
    else:
        #PDF formula of k_2(A)
        hand_cond = float(np.max(sigma) / np.min(only_positive))

    # Library ones for comparison
    lib_rank = int(np.linalg.matrix_rank(RectA))
    lib_cond = float(np.linalg.cond(RectA))

    # Moore-Penrose pseudoinverse basically the S is built by inverting the positive SINGULAR values
    InvS = np.zeros((cols, rows), dtype=float)
    for i in range(hand_rank):
        InvS[i, i] = 1.0 / sigma[i]

    RightV = RightVt.T
    # A^I = V S^I U^T
    pseudo_moore = RightV @ InvS @ LeftU.T

    #A^T A
    gram = RectA.T @ RectA
    # A^J
    pseudo_least = np.linalg.inv(gram) @ RectA.T

    # if invertible low, if not, high.
    norm_diff_1 = float(np.linalg.norm(pseudo_moore - pseudo_least, 1))

    return {
        "sigma": sigma,
        "hand_rank": hand_rank,
        "lib_rank": lib_rank,
        "hand_cond": hand_cond,
        "lib_cond": lib_cond,
        "pseudo_moore": pseudo_moore,
        "pseudo_least": pseudo_least,
        "norm_diff_1": norm_diff_1,
        "LeftU": LeftU,
        "RightVt": RightVt,
    }

# These are the homework prints for p = n. -- AI generated comments
def show_square_case(InitialA: np.ndarray,
                     lambdas: np.ndarray,
                     vectors: np.ndarray,
                     check_norm: float,
                     final_board: np.ndarray,
                     jacobi_steps: int,
                     cholesky_steps: int) -> None:
    print()
    print("=== Case p = n ===")
    print("InitialA =")
    print(InitialA)
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

    if SHOW_EXTRA:
        print("=== Extra ===")
        print("Jacobi iterations =", jacobi_steps)
        print("Cholesky iteration steps =", cholesky_steps)
        print("||U^T U - I|| =")
        print(np.linalg.norm(vectors.T @ vectors - np.eye(vectors.shape[0])))
        print()

# These are the homework prints for p > n. -- AI generated comments
def show_tall_case(RectA: np.ndarray, pack: dict) -> None:
    print()
    print("=== Case p > n ===")
    print("A =")
    print(RectA)
    print()

    print("Singular values of matrix A =")
    print(pack["sigma"])
    print()

    print("Rank of matrix A (implemented formula) =")
    print(pack["hand_rank"])
    print()

    print("Rank of matrix A (library) =")
    print(pack["lib_rank"])
    print()

    print("Conditioning number of matrix A (implemented formula) =")
    print(pack["hand_cond"])
    print()

    print("Conditioning number of matrix A (library) =")
    print(pack["lib_cond"])
    print()

    print("Moore-Penrose pseudo-inverse A^I =")
    print(pack["pseudo_moore"])
    print()

    print("Least squares pseudo-inverse A^J =")
    print(pack["pseudo_least"])
    print()

    print("||A^I - A^J||_1 =")
    print(pack["norm_diff_1"])
    print()

    if SHOW_EXTRA:
        print("=== Extra ===")
        print("U from SVD =")
        print(pack["LeftU"])
        print()
        print("V^T from SVD =")
        print(pack["RightVt"])
        print()

# This is the full Tema 5 run.
def main() -> None:
    np.set_printoptions(precision=10, suppress=True, linewidth=220)

    p = int(input("p = ").strip())
    n = int(input("n = ").strip())
    power = int(input("t for eps = 10^(-t), t = ").strip())
    eps = make_eps(power)
    limit = int(input("kmax = ").strip())
    if p < n:
        raise ValueError("The statement covers only p = n and p > n, so p must be >= n.")

    seed_text = input("random seed (press Enter for default seed) = ").strip()
    if seed_text == "":
        seed_value = 12345
    else:
        seed_value = int(seed_text)

    # Square Case p = n
    if p == n:
        InitialA = build_square_matrix(n, seed_value)

        lambdas, vectors, _, jacobi_steps = jacobi_run(InitialA, eps, limit)
        check_norm = check_jacobi_relation(InitialA, lambdas, vectors)
        final_board, cholesky_steps = cholesky_loop(InitialA, eps, limit)

        show_square_case(
            InitialA=InitialA,
            lambdas=lambdas,
            vectors=vectors,
            check_norm=check_norm,
            final_board=final_board,
            jacobi_steps=jacobi_steps,
            cholesky_steps=cholesky_steps,
        )

    # Tall branch (p > n)
    else:
        RectA = build_tall_matrix(p, n, seed_value)
        pack = svd_branch_values(RectA, eps)
        show_tall_case(RectA, pack)

if __name__ == "__main__":
    main()
