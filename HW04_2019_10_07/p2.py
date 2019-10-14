from HW04_2019_10_07.p1 import binomial_tree, payoff
import numpy as np
from scipy.stats import norm, stats
import matplotlib.pyplot as plt


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

    n_lst = range(1, 501, 1)
    c_tree_arr = np.asarray([binomial_tree(payoff=payoff, n=n, rp=r, sigma=sigma, S=S, K=K, T=T) for n in n_lst])
    c_black_arr = np.ones(len(n_lst)) * black_scholes(r=r, sigma=sigma, S=S, K=K, T=T)
    logErr_lst = np.log(np.absolute(c_tree_arr - c_black_arr))

    slope, intercept, r_value, p_value, std_err = stats.linregress(n_lst[:100], logErr_lst[:100])
    print("Rate of Convergence:", slope)

    # plt.scatter(n_lst, logErr_lst)
    plt.plot(n_lst, logErr_lst)
    plt.title('Logarithm of the Error vs. n')
    plt.xlabel('Steps of Binomial Trees $n$')
    plt.ylabel('Logarithm of the Error')
    plt.tight_layout()
    plt.savefig('p2.pdf')
    plt.show()

    """Discussion
    The plot shows that the logarithm of the error constantly decreases according to the increment of n. However, my 
    implementation causes an oscillation of errors, and the plot shows a repeated pattern of butterflies. This pattern
    can be used to accelerate the computing, i.e. with small n, as shown on th plot when n is around 100, the binomial
    tree method already gets pretty good result, and we thus don't need higher n steps, which saves much computing time.
    """
