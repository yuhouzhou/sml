import numpy as np
import matplotlib.pyplot as plt


"""
GTM:
given poitns: see part 1c
"""

def binomial_s(N, rp, sigma, S, T, seed):
    """Calculate Stock Price using Binomial Tree

    Args:
        N: the number of steps
        rp: the risk-free period interest rate
        sigma: the volatility
        S: the initial stock price
        T: the maturity
        seed: seed

    Returns:
        binomial path
    """
    # Calculating the increase rate and decrease rate
    np.random.seed(seed)
    u = np.exp(sigma * np.sqrt(T / N))
    d = 1 / u
    p = (np.exp(rp * T / N) - d) / (u - d)
    ud = np.random.choice([0, 1], size=(N - 1), p=[1 - p, p])
    dw = (ud == 1) * u + (ud != 1) * d
    path = np.cumprod(dw)
    return np.insert(path, 0, S)


def draw_subplot(b_arr, mean_lst, std_lst, steps, mu, sigma):
    # plot mean and standard deviation
    plt.plot(steps, mean_lst, c='r', label='empirical mean')
    plt.plot(steps, mean_lst - std_lst, c='g', label='empirical standard deviation')
    plt.plot(steps, mean_lst + std_lst, c='g')
    # plot 10 sample paths
    n_sample = 10
    for i, b in enumerate(b_arr[0:n_sample - 1]):
        if i == 0:
            plt.plot(steps, b, c='b', label='Binomial path')
        else:
            plt.plot(steps, b, c='b')

    plt.xlabel('$t$')
    plt.ylabel('$S$')
    plt.title('Ensemble of Binomial Paths $S$, with $\mu = {}, \sigma = {}$'.format(mu, sigma))
    plt.legend()


if __name__ == '__main__':
    N = 500
    T = 1
    S = 1
    seeds = np.arange(0, 1000)
    steps = np.linspace(0, 1, N)

    # compute binomial paths
    mu0 = 0.2
    sigma0 = 0.6
    b_arr0 = np.asarray([binomial_s(N=N, rp=mu0, sigma=sigma0, S=S, T=T, seed=seed) for seed in seeds])
    # compute mean and std. of every step
    mean_arr0 = np.mean(b_arr0, axis=0)
    std_arr0 = np.std(b_arr0, axis=0)

    # compute binomial paths
    mu1 = 0.6
    sigma1 = 0.2
    b_arr1 = np.asarray([binomial_s(N=N, rp=mu1, sigma=sigma1, S=S, T=T, seed=seed) for seed in seeds])
    # compute mean and std. of every step
    mean_arr1 = np.mean(b_arr1, axis=0)
    std_arr1 = np.std(b_arr1, axis=0)

    plt.rc('figure', figsize=(14, 10))

    plt.subplot(2, 1, 1)
    draw_subplot(b_arr0, mean_arr0, std_arr0, steps, mu0, sigma0)
    plt.subplot(2, 1, 2)
    draw_subplot(b_arr1, mean_arr1, std_arr1, steps, mu1, sigma1)

    plt.tight_layout()
    plt.savefig('p1b.pdf')
    plt.show()
