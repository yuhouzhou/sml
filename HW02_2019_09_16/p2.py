import numpy as np
from scipy.optimize import brentq


def pv(irr, c, m, n, d=0, f = 1):
    """

    :param irr: yield to maturity
    :param c: annual coupon rate
    :param m: number of payments per year
    :param n: time to maturity
    :param d: discount rate
    :param f: par value
    :return: present value or price
    """
    # coupon payment
    C = f * c / m
    # period
    i = np.arange(1, (m * n) + 1)
    # present value of the bond
    return np.sum(C / (1 + irr / m) ** i) + (f / (1 + irr / m) ** (n * m)) - d*f


if __name__ == '__main__':
    c = 0.1
    m = 2
    n = 10
    d = 0.75

    irr = brentq(pv, 0, 1, args=(c, m, n, d))

    print("yield to maturity = {:.4f}%".format(irr*100))
