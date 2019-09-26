import numpy as np
import matplotlib.pyplot as plt
from p2 import pv
from scipy.optimize import brent


def fv_horizon(r, m, n, f, horizon, c=0):
    """

    :param r: yield to maturity
    :param m: number of payments per year
    :param n: time to maturity
    :param f: par value
    :param horizon: horizon date
    :param c: annual coupon rate
    :return: horizon price
    """
    return pv(irr=r, c=c, m=m, n=n, f=f) * (1 + r / m) ** (horizon * m)


if __name__ == '__main__':
    n = 30
    c = 0.1
    m = 1
    f = 1
    horizon = 10

    r_lst = np.linspace(0, 0.3, 1000)
    horizon_lst = [fv_horizon(r, m, n, f, horizon, c) for r in r_lst]

    yield_min = brent(fv_horizon, args=(m, n, f, horizon, c))
    fv_horizon_min = fv_horizon(yield_min, m, n, f, horizon, c)

    plt.plot(r_lst, horizon_lst)
    plt.xlabel("Yield")
    plt.ylabel("Horizon Price")
    plt.title("Horizon Price vs. Yield")
    plt.scatter(yield_min, fv_horizon_min, marker='x')
    plt.text(s=str(round(yield_min, 4))+', '+str(round(fv_horizon_min, 4)),
                 x=yield_min+0.005, y=fv_horizon_min-0.08)
    plt.tight_layout()
    plt.savefig('p7.pdf')
    plt.show()

    print("When yield = {:.4f} and par value = 1, minimum horizon price = {:.4f}".format(yield_min, fv_horizon_min))
