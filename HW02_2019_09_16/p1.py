import numpy as np
from scipy.optimize import bisect, newton, brent
from sys import exit


def my_bisect(f, a=0, b=1, tol=1e-12, maxiter=100):
    """

    :param f: python function return a number
    :param a: one side of the start interval
    :param b: the other side of the start interval
    :param tol: tolerance about the precision
    :param maxiter: maximum iteration
    :return: the root
    """
    # assume a < b
    if a > b:
        tmp = a
        a = b
        b = tmp

    for i in range(1, maxiter + 1):
        # mid point of (a, b)
        x = (a + b) / 2
        f_x = f(x)
        if f_x == 0 or (b - a) / 2 < tol:
            return x
        if f(a) * f_x > 0:
            a = x
        else:
            b = x
        if i == maxiter:
            print("Maximum iteration reached!")
            exit(1)


def my_newton(f, x0=0.5, tol=1.48e-08, maxiter=50):
    pass


def secant(poly):
    pass


def my_brent(poly):
    pass


if __name__ == '__main__':
    def f(x):
        n = 300
        c, p = 120 * np.arange(10, n + 10), 15000
        coe = c[::-1]
        index = np.arange(1, len(c) + 1)[::-1]
        f_x = np.dot(coe, x ** index) - p
        return f_x


    print(my_bisect(f), bisect(f, 0, 1))
