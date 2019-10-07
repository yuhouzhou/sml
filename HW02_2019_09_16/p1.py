import numpy as np
from scipy.optimize import bisect, newton, brentq
from sys import exit
from timeit import Timer


"""
GTM:  given point 6
"""

def my_bisect(f, x0, x1, tol=1e-12, maxiter=50):
    """

    :param f: python function return a number
    :param x0: float. one side of the start interval
    :param x1: float. the other side of the start interval
    :param tol: float. tolerance about the precision
    :param maxiter: int. maximum iteration
    :return: float. the root
    """
    # assume a < b
    if x0 > x1:
        tmp = x0
        x0 = x1
        x1 = tmp

    for i in range(1, maxiter + 1):
        # mid point of (a, b)
        x = (x0 + x1) / 2
        f_x = f(x)
        if f_x == 0 or (x1 - x0) < tol:
            return x
        if f(x0) * f_x > 0:
            x0 = x
        else:
            x1 = x
        if i == maxiter:
            print("Maximum iteration reached!")
            exit(1)


def my_newton(f, x0, fprime, tol=1.48e-08, maxiter=50):
    """

    :param f: python function return a number
    :param x0: float. initial guess
    :param fprime: python function return the derivative value of f
    :param tol: float. tolerance about the precision
    :param maxiter: int. maximum iteration
    :return: float. the root
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


def my_secant(f, x0, x1, tol=1.48e-08, maxiter=50):
    """

    :param f: python function return a number
    :param x0: float. one side of the initial interval
    :param x1: float. the other side of the initial interval
    :param tol: float. the tolerance of the precision
    :param maxiter: int. maximum iteration
    :return: float. the root
    """
    for i in range(1, maxiter + 1):
        # secant
        f_x0, f_x1 = f(x0), f(x1)
        x = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
        x0 = x1
        x1 = x
        f_x = f(x)
        if f_x == 0 or abs(f_x) < tol:
            return x
        if i == maxiter:
            print("Maximum iteration reached!")
            exit(1)


def my_brent(f, x0, x1, tol=1.48e-08, maxiter=50):
    """

    :param f: python function return a number
    :param x0: float. one side of the initial interval
    :param x1: float. the other side of the initial interval
    :param tol: float. the tolerance of the precision
    :param maxiter: int. maximum iteration
    :return: float. the root

    reference: https://en.wikipedia.org/wiki/Brent%27s_method#Algorithm
    """
    def inverse_quadratic_interpolation_step(f, x0, x1, x2):
        def term(f, a, b, c):
            return (a * f(b) * f(c)) / ((f(a) - f(b)) * (f(a) - f(c)))
        return term(f, x0, x1, x2) + term(f, x1, x0, x2) + term(f, x2, x0, x1)

    def secant_step(f, x0, x1):
        return x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))

    def bisect_step(x0, x1):
        return (x0 + x1) / 2

    assert (f(x0) * f(x1)) < 0

    if f(x0) * f(x1) >= 0:
        exit(1)
    if abs(f(x0)) < abs(f(x1)):
        tmp = x0
        x0 = x1
        x1 = tmp

    x2 = x0
    mflag = 1

    for i in range(maxiter):

        if f(x0) != f(x2) and f(x1) != f(x2):
            x = inverse_quadratic_interpolation_step(f, x0, x1, x2)
        else:
            x = secant_step(f, x0, x1)
        if ((3 * x0 + x1) / 4 > x > x1) or \
                (mflag == 1 and abs(x - x1) >= abs(x1 - x2)) or \
                (mflag == 0 and abs(x - x1) >= abs(x2 - x3)) or \
                (mflag == 1 and abs(x1 - x2) < tol) or \
                (mflag == 0 and abs(x2 - x3) < tol):
            x = bisect_step(x0, x1)
            mflag = 1
        else:
            mflag = 0

        x3 = x2
        x2 = x1

        if f(x0) * f(x) < 0:
            x1 = x
        else:
            x0 = x

        if abs(f(x0)) < abs(f(x1)):
            tmp = x0
            x0 = x1
            x1 = tmp

        if f(x0) == 0 or abs(x1 - x0) < tol:
            return x


def f_maker(c, p):
    """

    :param c: np.array. cash flow
    :param p: float. price at sold
    :return: function. f and df taking x as an argument
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


def irr(root):
    return 1 / root - 1


if __name__ == '__main__':
    n = 300
    c, p = 120 * np.arange(10, n + 10), 15000

    f, fprime = f_maker(c, p)

    # print(my_bisect(f, 0.5, 0.9), bisect(f, 0.5, 0.9))
    # print(my_newton(f, 0.5, fprime, maxiter=500),
    #       newton(f, 0.5, maxiter=500, fprime=fprime))
    # print(my_secant(f, 0.5, 0.9), newton(f, 0.9))
    # print(my_brent(f, 0.5, 0.9), brentq(f, 0.5, 0.9))

    """ GTM: I changed the function to the available brentq see the right value...."""
    print("IRR = {:.3f}%".format(irr(brentq(f, 0, 1)) * 100))

    """GTM: you wrote the function irr but forgot to use it...
    """

    t0 = Timer('my_bisect(f, .5, .9)', 'from __main__ import my_bisect, f')
    print("my_bisect:\n\tvalue:{:.7f}, time:{:.5f} sec".format(irr(my_bisect(f, .5, .9)), t0.timeit(number=100)))
    t1 = Timer('my_newton(f, .5, fprime, maxiter=500)', 'from __main__ import my_newton, f, fprime')
    print("my_newton:\n\tvalue:{:.7f}, time:{:.5f} sec".format(irr(my_newton(f, .5, fprime, maxiter=500)), t1.timeit(number=100)))
    t2 = Timer('my_secant(f, .5, .9)', 'from __main__ import my_secant, f')
    print("my_secant:\n\tvalue:{:.7f}, time:{:.5f} sec".format(irr(my_secant(f, .5, .9)), t2.timeit(number=100)))
    t3 = Timer('my_brent(f, .5, fprime, maxiter=500)', 'from __main__ import my_prent, f')
    print("my_brent:\n\tvalue:{:.7f}, time:{:.5f} sec".format(irr(my_brent(f, .5, .9)), t2.timeit(number=100)))
