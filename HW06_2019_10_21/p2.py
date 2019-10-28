import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import stats
from HW06_2019_10_21.p1a import geom_brownian
from HW04_2019_10_07.p1 import payoff
from HW04_2019_10_07.p2 import black_scholes

if __name__ == '__main__':
    X = 0.9
    T = 1
    r = mu = 0.05
    sigma = 0.3
    S0 = 1

    price_black = black_scholes(r=r, sigma=sigma, S=S0, K=X, T=T)

    diff_lst = []
    # i_lst = np.arange(1, 1000)
    i_lst = np.array(1.3 ** np.arange(1, 25), dtype=int)

    for i in i_lst:
        seeds = np.arange(i)

        # Monte Carlo Simulation
        # Sample random paths for S in a risk-neutral world.
        S_lst = [geom_brownian(N=500, mu=mu, sigma=sigma, seed=seed)[-1] for seed in seeds]
        # Calculate the payoff from the derivative.
        pf_lst = [payoff(S=S, K=X) for S in S_lst]
        # Calculate the mean of the sample payoffs to get an estimate of the expected payoff in a risk-neutral world.
        pf_mean = np.mean(pf_lst)
        # Discount this expected payoff at the risk-free rate to get an estimate of the value of the derivative.
        # price = pf_mean / (1 + r)
        price = pf_mean / np.exp(r * T)

        diff = np.abs(price_black - price)
        diff_lst.append(diff)

    slope, intercept, r_value, p_value, std_err = stats.linregress(np.log(i_lst), np.log(diff_lst))
    print("Rate of Convergence:", slope)

    # plt.rc('figure', figsize=(14, 10))
    plt.plot(i_lst, diff_lst, '*')
    plt.loglog(i_lst, 0.54 / i_lst, label='linear fit')
    plt.xlabel('number of samples')
    plt.ylabel('deviation')
    plt.title('Deviation between GBM Monte-Carlo and Black-Scholes vs. Number of Samples')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('p2.pdf')
    plt.show()