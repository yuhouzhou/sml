import matplotlib.pyplot as plt
import numpy as np
from p2 import pv

"""
GTM:  given points 2
"""

if __name__ == "__main__":
    r = 0.06
    f = 1000
    m = 2
    c_lst = [0.02, 0.06, 0.12]
    years = np.arange(0, 11)

    for c in c_lst:
        price_lst = [pv(irr=r, c=c, m=m, n=year, f=f) for year in years]
        plt.plot(years, price_lst, label='c $= {:.0f}\%$'.format(c * 100))

    """GTM: OR since you are asked time to maturity, 
    reverse the x axis.... """
    plt.xlim(max(years),0)
    plt.xlabel("Time to Maturity (Years)")
    plt.ylabel("Price")
    plt.title("Price vs. Time to Maturity for Level Coupon Bonds")
    plt.legend()
    plt.tight_layout()
    plt.savefig('p3.pdf')
    plt.show()

