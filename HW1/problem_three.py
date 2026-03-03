import time
import numpy as np


def reduce_to_principal_tan_domain(x: float, mic: float = 1e-12) -> float:
    """
    Reduce x using tan periodicity so the result is in (-pi/2, pi/2),
    Returns a value in (-pi/2, pi/2) or raises ValueError for points where tan is undefined.
    """
    pi = np.pi
    half_pi = pi / 2

    r = (x + half_pi) % pi - half_pi  # r in (-pi/2, pi/2]

    # Handle singularities, use tolerance since floating point
    if abs(abs(r) - half_pi) <= mic:
        raise ValueError("tan is undefined at x = (pi/2) + k*pi")

    return r


def tan_continued_fraction_lentz(x: float, eps: float = 1e-10, mic: float = 1e-12, max_iter: int = 10_000) -> float:
    """
    Approximate tan(x) using the continued fraction and modified Lentz method from the PDF.

    Max iterations added by ChatGPT, for safety.
    """
    x = reduce_to_principal_tan_domain(x, mic=mic)

    b0 = 1.0
    f = b0
    if f == 0.0:
        f = mic
    C = f
    D = 0.0

    a = -(x * x)

    for j in range(1, max_iter + 1):
        b = 2.0 * j + 1.0

        #D
        D = b + a * D
        if D == 0.0:
            D = mic
        D = 1.0 / D

        #C
        C = b + a / C
        if C == 0.0:
            C = mic

        delta = C * D
        f *= delta

        if abs(delta - 1.0) < eps:
            break

    return x / f


def tan_polynomial(x: float, mic: float = 1e-12) -> float:
    """
    Approximate tan(x) using the polynomial from the PDF.
    Singularity and division safety suggested by ChatGPT
    """
    x = reduce_to_principal_tan_domain(x, mic=mic)

    pi = np.pi
    quarter_pi = pi / 4
    half_pi = pi / 2

    # Antisymmetry
    if x < 0.0:
        return -tan_polynomial(-x, mic=mic)

    # Reduce x from [pi/4, pi/2) into (0, pi/4]
    if x >= quarter_pi:
        t = tan_polynomial(half_pi - x, mic=mic)
        if abs(t) <= mic:
            # extremely close to singularity or underflow; avoid division blowup
            return np.sign(t) * (1.0 / mic) if t != 0.0 else (1.0 / mic)
        return 1.0 / t

    # Polynomial on [0, pi/4)
    c1 = 0.33333333333333333
    c2 = 0.133333333333333333
    c3 = 0.053968253968254
    c4 = 0.0218694885361552

    x2 = x * x
    x3 = x2 * x
    x5 = x3 * x2
    x7 = x5 * x2
    x9 = x7 * x2

    return x + c1 * x3 + c2 * x5 + c3 * x7 + c4 * x9


if __name__ == "__main__":
    # Settings
    N = 10_000
    eps = 1e-10  # corresponds to 10^-p in the PDF
    mic = 1e-12

    # Generate x in (-pi/2, pi/2) (avoid endpoints)
    half_pi = np.pi / 2
    rng = np.random.default_rng(42) # get the reference? :)
    xs = rng.uniform(-half_pi + 1e-8, half_pi - 1e-8, size=N)

    # Reference using numpy
    t0 = time.perf_counter()
    tan_np = np.tan(xs)
    t_np = time.perf_counter() - t0

    # Continued fraction (scalar per x, as said)
    t0 = time.perf_counter()
    tan_cf = np.array([tan_continued_fraction_lentz(float(x), eps=eps, mic=mic) for x in xs], dtype=float)
    t_cf = time.perf_counter() - t0

    # Polynomial (scalar per x)
    t0 = time.perf_counter()
    tan_poly = np.array([tan_polynomial(float(x), mic=mic) for x in xs], dtype=float)
    t_poly = time.perf_counter() - t0

    # Errors
    err_cf = np.abs(tan_np - tan_cf)
    err_poly = np.abs(tan_np - tan_poly)


    def summarize(name: str, errs: np.ndarray, elapsed: float):
        return {
            "method": name,
            "time_sec": elapsed,
            "mean_abs_err": float(np.mean(errs)),
            "median_abs_err": float(np.median(errs)),
            "max_abs_err": float(np.max(errs)),
            "p95_abs_err": float(np.percentile(errs, 95)),
        }


    summary = [
        summarize("numpy.tan (reference)", np.zeros_like(xs), t_np),
        summarize("continued_fraction_lentz", err_cf, t_cf),
        summarize("polynomial_degree_9", err_poly, t_poly),
    ]

    # Pretty print (provided by ChatGPT, including the summary form)
    print(f"N={N}, eps={eps}, mic={mic}")
    print("-" * 90)
    for s in summary:
        print(
            f"{s['method']:>26} | time={s['time_sec']:.6f}s"
            f" | mean={s['mean_abs_err']:.3e}"
            f" | median={s['median_abs_err']:.3e}"
            f" | p95={s['p95_abs_err']:.3e}"
            f" | max={s['max_abs_err']:.3e}"
        )
