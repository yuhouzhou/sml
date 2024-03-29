from HW04_2019_10_07.p1 import binomial_tree, payoff
# from p1 import binomial_tree, payoff
import numpy as np
from scipy.stats import norm, stats
import matplotlib.pyplot as plt

"""
GTM: In loglog plot, we do not compute the logarithmic value of
     the error or periods. Loglog plot sketch the graph in logarithmic
     scale so that order of scale can be detected easily. When one looks
     at the axes, it is visible that actual error and period values are
     replaced on them in log-scaling.
     wrong loglog plotting...
     plotting the error scale wrong... (-0.5)
given points: 5.5
"""

def black_scholes(r, sigma, S, K, T):
    """
    Black-Scholes formula
    :param r: the risk-free period interest rate
    :param sigma: the volatility
    :param S: the initial stock price
    :param K: the strike price
    :param T: the maturity
    :return: the price of the option at time T = 0
    """
    x = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    return S * norm.cdf(x) - K * np.exp(-r * T) * norm.cdf(x - sigma * np.sqrt(T))


if __name__ == '__main__':
    r = 0.03
    sigma = 0.5
    S = 1
    K = 1.2
    T = 1

    # print(binomial_tree(payoff=payoff, n=100, rp=r, sigma=sigma, S=S, K=K, T=T))
    # print(black_scholes(r=r, sigma=sigma, S=S, K=K, T=T))

    #n_lst = range(1, 501, 1)
    """ GTM: choosing well distributed points depending on data 
    helps much more... """
    n_lst = np.array(1.3**np.arange(1,25),dtype=int)
    c_tree_arr = np.asarray([binomial_tree(payoff=payoff, n=n, rp=r, sigma=sigma, S=S, K=K, T=T) for n in n_lst])
    c_black_arr = np.ones(len(n_lst)) * black_scholes(r=r, sigma=sigma, S=S, K=K, T=T)
    Err_lst = np.absolute(c_tree_arr - c_black_arr)

    slope, intercept, r_value, p_value, std_err = stats.linregress(np.log(n_lst), np.log(Err_lst))
    print("Rate of Convergence:", slope)

    # plt.scatter(n_lst, logErr_lst)
    plt.plot(n_lst, Err_lst,'*')
    """GTM: you are supposed to plot a reference lien as well... """
    plt.loglog(n_lst, 0.01/n_lst, label='linear fit')
    plt.title('Logarithm of the Error vs. n')
    plt.xlabel('Steps of Binomial Trees $n$')
    plt.ylabel('Error')
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    #plt.savefig('p2.pdf')
    plt.show()

    """Discussion
    The plot shows that the logarithm of the error constantly decreases according to the increment of n. However, my 
    implementation causes an oscillation of errors, and the plot shows a repeated pattern of butterflies. This pattern
    can be used to accelerate the computing, i.e. with small n, as shown on th plot when n is around 100, the binomial
    tree method already gets pretty good result, and we thus don't need higher n steps, which saves much computing time.
    """
