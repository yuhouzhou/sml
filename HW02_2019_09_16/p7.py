import numpy as np
import matplotlib.pyplot as plt
from p2 import pv
from scipy.optimize import brent


def fv_horizon(r, m, n, f, horizon, c=0):
    return pv(irr=r, c=c, m=m, n=n, f=f) * (1 + r / m) ** (horizon * m)


if __name__ == '__main__':
    n = 30
    c = 0.1
    m = 1
    f = 1
    horizon = 10

    r_lst = np.linspace(0, 0.3, 1000)
    horizon_lst = [fv_horizon(r, m, n, f, horizon, c) for r in r_lst]
    plt.plot(r_lst, horizon_lst)
    plt.xlabel("Yield")
    plt.ylabel("Horizon Price")
    plt.title("Horizon Price vs. Yield")
    plt.tight_layout()
    plt.savefig('p7.pdf')
    plt.show()

    yield_min = brent(fv_horizon, args=(m, n, f, horizon, c))
    fv_horizon_min = fv_horizon(yield_min, m, n, f, horizon, c)
    print("When yield = {:.4f} and par value = 1, minimum horizon price = {:.4f}".format(yield_min,fv_horizon_min))
