import numpy as np
import matplotlib.pyplot as plt
from p2 import pv

def forward_v(r, c, m, n, f, i):
    return pv(irr=r, c=c, m=m, n=n, f=f) * (1 + r / m) ** (i + 1)

if __name__ == '__main__':
    r = 0.08
    c = 0.08
    n = 15
    m = 2
    f = 1

    periods = np.arange(1, m * n + 1)

    # r_lst_dict = {'r: 6% - 8%': np.linspace(r, 0.06,  n * m),
    #              'r: 8% - 8%' : np.linspace(r, r, n * m),
    #              'r: 8% - 10%' : np.linspace(r, 0.1, n * m)}

    # for i, (key ,r_lst) in enumerate(r_lst_dict.items()):
    #     forward_v_lst = [price * (1 + r / m) ** (i + 1) for price in [pv(irr=r, c=c, m=m, n=n, f=f) for r in r_lst]]
    #     plt.plot(np.multiply(period, 1/2), forward_v_lst, label='${}$'.format(key.replace('%', '\%')))

    for r in [0.06, 0.08, 0.1]:
        forward_v_lst = [forward_v(r, c, m, n, f, i) for i in periods]
        plt.plot(np.multiply(periods, 1 / 2), forward_v_lst, label='$r = {}$'.format(r))

    plt.xlabel("Time to Maturity (Years)")
    plt.ylabel("Forward Value")
    plt.title("Forward Value vs. Time to Maturity")
    plt.legend()
    plt.tight_layout()
    plt.savefig('p6.pdf')
    plt.show()
