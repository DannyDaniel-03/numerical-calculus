import numpy as np
from pathlib import Path
from typing import TypedDict

up = Path(__file__).parent
DATA_DIR = up / "data"


class SystemData(TypedDict):
    b: np.ndarray
    d0: np.ndarray
    d1: np.ndarray
    d2: np.ndarray
    x: np.ndarray
    n: int
    p: int
    q: int


def load_one_array(path: str | Path) -> np.ndarray:
    return np.loadtxt(path, dtype=float)


def load_data(eps: float) -> dict[str, SystemData]:
    systems: dict[str, SystemData] = {}

    for i in range(1, 6):
        b = load_one_array(DATA_DIR / f"b_{i}.txt")
        d0 = load_one_array(DATA_DIR / f"d0_{i}.txt")
        d1 = load_one_array(DATA_DIR / f"d1_{i}.txt")
        d2 = load_one_array(DATA_DIR / f"d2_{i}.txt")

        if np.any(np.abs(d0) < eps):
            print(f"Skipping system{i}: d0 contains values too close to zero for eps = {eps}.")
            continue

        d0_size = d0.size
        b_size = b.size

        if d0_size != b_size:
            print(f"Skipping system{i}: d0 size is {d0_size}, but b size is {b_size}.")
            continue

        n = b_size
        p = n - d1.size
        q = n - d2.size

        systems[f"system{i}"] = {
            "b": b,
            "d0": d0,
            "d1": d1,
            "d2": d2,
            "n": n,
            "p": p,
            "q": q,
            "x": np.zeros(n, dtype=float),
        }

    return systems


def check_diagonal_dominance(system: SystemData) -> bool:
    d0 = system["d0"]
    d1 = system["d1"]
    d2 = system["d2"]
    n = system["n"]
    p = system["p"]
    q = system["q"]

    for i in range(n):
        s = 0.0

        if i - p >= 0:
            s += abs(d1[i - p])
        if i + p < n:
            s += abs(d1[i])

        if i - q >= 0:
            s += abs(d2[i - q])
        if i + q < n:
            s += abs(d2[i])

        if abs(d0[i]) <= s:
            return False

    return True


def gauss_seidel(system: SystemData, eps: float, kmax: int = 10_000) -> bool:
    b = system["b"]
    d0 = system["d0"]
    d1 = system["d1"]
    d2 = system["d2"]
    n = system["n"]
    p = system["p"]
    q = system["q"]

    system["x"].fill(0.0)
    x = system["x"]

    old_settings = np.seterr(over='raise', invalid='raise')
    try:
        for k in range(kmax):
            x_old = x.copy()

            for i in range(n):
                s = 0.0

                if i - p >= 0:
                    s += d1[i - p] * x[i - p]
                if i + p < n:
                    s += d1[i] * x_old[i + p]

                if i - q >= 0:
                    s += d2[i - q] * x[i - q]
                if i + q < n:
                    s += d2[i] * x_old[i + q]

                x[i] = (b[i] - s) / d0[i]

            dx = np.linalg.norm(x - x_old, ord=np.inf)

            if dx < eps:
                return True

            if dx > 1e10 or not np.isfinite(dx):
                x.fill(0.0)
                return False

        x.fill(0.0)
        return False

    except FloatingPointError:
        x.fill(0.0)
        return False

    finally:
        np.seterr(**old_settings)


def compute_y(system: SystemData) -> np.ndarray | None:
    x = system["x"]

    if np.all(x == 0.0):
        return None

    d0 = system["d0"]
    d1 = system["d1"]
    d2 = system["d2"]
    n = system["n"]
    p = system["p"]
    q = system["q"]

    y = np.zeros(n, dtype=float)

    for i in range(n):
        s = d0[i] * x[i]

        if i - p >= 0:
            s += d1[i - p] * x[i - p]
        if i + p < n:
            s += d1[i] * x[i + p]

        if i - q >= 0:
            s += d2[i - q] * x[i - q]
        if i + q < n:
            s += d2[i] * x[i + q]

        y[i] = s

    return y


def residual_inf_norm(system: SystemData, y: np.ndarray) -> float:
    return float(np.linalg.norm(y - system["b"], ord=np.inf))


def main():
    t = int(input("t for eps = 10^(-t), t = ").strip())
    eps = 10.0 ** (-t)

    systems = load_data(eps)
    for system_name, system in systems.items():
        print(f"System {system_name}:")
        print(f"n = {system['n']}")
        print(f"p = {system['p']}")
        print(f"q = {system['q']}")
        print(f"is diagonally dominant: {check_diagonal_dominance(system)}")
        print()

    for system_name, system in systems.items():
        converged = gauss_seidel(system, eps)

        print(f"{system_name}:")
        if not converged:
            print("Did not converge.")
        else:
            print("Converged.")
            print("x =", system["x"])

            y = compute_y(system)
            if y is None:
                print("y could not be computed because x is the zero vector.")
            else:
                print("y =", y)
                print("||AxGS - b||inf =", residual_inf_norm(system, y))
        print()


if __name__ == "__main__":
    main()
