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


def my_newton(f, fprime, x0=0.5, tol=1.48e-08, maxiter=500):
    """

    :param f: python function return a number
    :param fprime: python function return the derivative value of f
    :param x0: initial guess
    :param tol: tolerance about the precision
    :param maxiter: maximum iteration
    :return: the root
    """
    x = x0
    for i in range(1, maxiter + 1):
        if f(x) == 0 or abs(f(x)) < tol:
            return x
        else:
            x_last = x
            # Newton-Raphson
            x = x_last - f(x_last) / fprime(x_last)
        if i == maxiter:
            print("Maximum iteration reached!")
            exit(1)


def my_secant(poly):
    pass


def my_brent(poly):
    pass

def f_maker(c, p):
    """

    :param c: np.array. cash flow
    :param p: float. price at sold
    :return:
    """
    coe = c[::-1]
    index = np.arange(1, len(c) + 1)[::-1]

    def f(x):
        f_x = np.dot(coe, x ** index) - p
        return f_x

    def fprime(x):
        fprime_x = np.dot(coe, index * x ** (index - 1))
        return fprime_x

    return f, fprime

if __name__ == '__main__':
    n = 300
    c, p = 120 * np.arange(10, n + 10), 15000

    f, fprime = f_maker(c, p)

    print(my_bisect(f), bisect(f, 0, 1))
    print(my_newton(f, fprime), newton(f, 0.5, maxiter=500, fprime=fprime))
