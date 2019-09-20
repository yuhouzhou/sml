import numpy as np


def irr(p, c):
    poly = np.append(c[::-1], -p)
    res = np.roots(poly)
    rate = 1 / res - 1
    rate = rate.item(np.argmin(np.abs(rate))).real
    return rate


if __name__ == '__main__':
    n = 40
    c, p = 100 * np.arange(42, n + 42), 80000
    print('IRR =', irr(p, c))
