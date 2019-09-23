import matplotlib.pyplot as plt
import numpy as np
from p2 import pv

if __name__ == "__main__":
    f = 1000
    m = 1
    n = 10
    c = 0.08
    r_lst = np.arange(0, 0.3, 0.01)

    price_lst = [pv(irr=r, c=c, m=m, n=n, f=f) for r in r_lst]
    plt.plot(r_lst, price_lst)
    plt.xlabel("Yields")
    plt.ylabel("Price")
    plt.title("Price of the Bond vs. Yields")
    plt.tight_layout()
    plt.savefig('p4.pdf')
    plt.show()

