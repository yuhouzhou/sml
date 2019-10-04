import numpy as np
import matplotlib.pyplot as plt
from p2 import pv

def forward_v(r, c, m, n, f):
    """

    :param r: yield to maturity
    :param c: annual coupon rate
    :param m: number of payments per year
    :param n: years to maturity
    :param d: discount rate
    :param f: par value
    :return: forward value
    """
    return pv(irr=r, c=c, m=m, n=n, f=f) * (1 + r / m) ** (n*m)

if __name__ == '__main__':
    r_lst = [0.06, 0.08, 0.1]
    c = 0.08
    m = 2
    f = 1

    years = np.arange(0, 16)

    for r in r_lst:
        forward_v_lst = [forward_v(r, c, m, n, f) for n in years]
        plt.plot(years, forward_v_lst, label='$r = {}$'.format(r))

    plt.xlabel("Years to Maturity")
    plt.ylabel("Forward Value")
    plt.title("Forward Value vs. Years to Maturity")
    plt.legend()
    plt.tight_layout()
    plt.savefig('p6.pdf')
    plt.show()
