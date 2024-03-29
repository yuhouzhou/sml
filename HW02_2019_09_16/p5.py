import matplotlib.pyplot as plt
import numpy as np


"""
GTM: Due to wrong formulation -0.5
given points 2.5
"""

def volatility(c, y, n, m, f=1):
    """

    :param c: annual coupon rate
    :param y: yield to maturity
    :param n: years to maturity
    :param m: number of payments per year
    :param f: par value
    :return: price volatility
    """
    C = f * c / m
    # formula from textbook "Financial Engineering and Computation" P32 (4.1)
    # The same with the formula on the lecture note, but with a negative sign.
    # this include the influence of f
    return - ((C / y) * n * m - (C / (y ** 2)) * ((1 + y) ** (n * m + 1) - (1 + y)) - n * m * f) / \
           ((C / y) * ((1 + y) ** (n * m + 1) - (1 + y)) + f * (1 + y))
    # The formula presented on the lecture note only works when f = 1.
    # In this problem f = 1000
    # f = 1
    # C = f * c / m
    # return ((C / y) * n * m - (C / (y ** 2)) * (1 + y) * ((1 + y) ** (n * m) - 1) - n * m) / \
    #        ((C / y) * (1 + y) * ((1 + y) ** (n * m) - 1) + (1 + y))


if __name__ == "__main__":
    r = 0.06
    f = 1000
    m = 2
    c_lst = [0.02, 0.06, 0.12]
    years = np.arange(0, 101)

    for c in c_lst:
        """ GTM: you need to update interest rate semiannually """
        volat_lst = [volatility(c=c, y=r/m, n=year, m=m, f=f) for year in years]
        plt.plot(years, volat_lst, label='c $= {:.0f}\%$'.format(c * 100))
    
    """GTM: since you are asked time to maturity, reverse the x axis.... """
    plt.xlim(100, 0)
    plt.xlabel("Time to Maturity (years)")
    plt.ylabel("Volatility")
    plt.title("Volatility vs. Time to Maturity for Level Coupon Bonds")
    plt.legend()
    plt.tight_layout()
    #plt.savefig('p5.pdf')
    plt.show()
