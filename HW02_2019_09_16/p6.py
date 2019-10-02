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
    # r = 0.08
    r_lst = [0.06, 0.08, 0.1]
    c = 0.08
    # c_lst = [0.06, 0.08, 0.1]
    m = 2
    f = 1

    years = np.arange(0, 16)

    # r = 0.08
    #
    # r_lst_dict = {'r: 6% - 8%': np.linspace(r, 0.06,  n * m),
    #              'r: 8% - 8%' : np.linspace(r, r, n * m),
    #              'r: 8% - 10%' : np.linspace(r, 0.1, n * m)}
    #
    # for key ,r_lst in r_lst_dict.items():
    #     forward_v_lst = [pv(irr=r, c=c, m=m, n=n, f=f) * (1 + r / m) ** (i + 1)  for i, r in enumerate(r_lst)]
    #     plt.plot(np.multiply(years, 1/2), forward_v_lst, label='${}$'.format(key.replace('%', '\%')))

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
