def get_u_min():
    u = 1.0
    m = 0

    while 1.0 + u != 1.0:
        u /= 10.0
        m += 1

    return u * 10.0


if __name__ == "__main__":
    print("u =", get_u_min())
