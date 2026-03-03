from HW1.problem_one import get_u_min
import random


def check_addition_associativity():
    u = get_u_min()
    x = 1.0
    y = z = u / 10

    ls = (x + y) + z
    rs = x + (y + z)
    is_equal = ls == rs

    print("(x + y) + z =", ls)
    print("x + (y + z) =", rs)
    print("Is equal:", is_equal)
    print()


def find_multiplication_nonassociativity():
    iterations = 0
    while True:
        iterations += 1
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        z = random.uniform(0, 1)

        ls = (x * y) * z
        rs = x * (y * z)
        is_equal = ls == rs

        if not is_equal:
            print("Found non-associativity after", iterations, "iterations")
            print("x =", x, "y =", y, "z =", z)
            print("(x * y) * z =", ls)
            print("x * (y * z) =", rs)
            print()
            break


if __name__ == "__main__":
    check_addition_associativity()
    find_multiplication_nonassociativity()

