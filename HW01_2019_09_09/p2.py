import numpy as np


"""
GTM: There is a mistake in formulation of pv
given point: 3.5
"""

def pv(c, m, n, r):
    return c * (1 - (1 + r / m) ** (-n * m)) / (1-1/(1+r / m))


def c(fv, m, n, r):
    return (fv * (r / m)) / ((1 + r / m) ** (n * m + 1) - (1 + r / m))
    # return fv / ((((1 + r/m)**(n*m) - 1) / (r / m)) * (1 + r/m))
    # return (fv*(r/m)) / (((1+r/m)**(n*m)-1)*(1+r/m))


if __name__ == '__main__':
    fv = pv(c=1500, m=12, n=30, r=0.02)
    a = c(fv=fv, m=12, n=10, r=0.02)
    print('A =', a)
