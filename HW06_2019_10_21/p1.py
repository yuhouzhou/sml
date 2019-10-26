import numpy as np
import matplotlib.pyplot as plt


def geom_brownian(N, mu, sigma, seed):
    """Generate a geometric Brownian path

    Args:
        N: number of steps
        mu: expected return rate
        sigma: volatility
        seed: seed

    Returns:
        a geometric Brownian path
    """
    np.random.seed(seed)
    dt = 1 / N
    S0 = 1
    ds = np.exp((mu - sigma ** 2 / 2) * dt + sigma * np.random.normal(loc = 0, scale = np.sqrt(dt), size=N - 1))
    path = S0*ds.cumprod()
    return np.insert(path, 0, S0)



def _draw_subplot(b_arr, mean_lst, std_lst, steps, mu, sigma):
    # plot mean and standard deviation
    plt.plot(steps, mean_lst, c='crimson', label='empirical mean')
    plt.plot(steps, mean_lst - std_lst, c='gold', label='empirical standard deviation')
    plt.plot(steps, mean_lst + std_lst, c='gold')
    # plot 10 sample paths
    n_sample = 10
    for i, b in enumerate(b_arr[0:n_sample - 1]):
        if i == 0:
            plt.plot(steps, b, c='c', label='geometric Brownian path')
        else:
            plt.plot(steps, b, c='c')

    plt.xlabel('$t$')
    plt.ylabel('$W(t)$')
    plt.title('Ensemble of Geometric Brownian Paths $W(t)$, with $\mu = {}, \sigma = {}$'.format(mu, sigma))
    plt.legend()


if __name__ == '__main__':
    N = 500
    n_path = 1000
    seeds = range(0, n_path)
    steps = np.linspace(0, 1, N)

    # compute brownian paths
    mu0 = 0.2
    sigma0 = 0.6
    b_arr0 = np.asarray([geom_brownian(N=N, mu=mu0, sigma=sigma0, seed=seed) for seed in seeds])
    # compute mean and std. of every step
    mean_lst0 = np.mean(b_arr0, axis=0)
    std_lst0 = np.std(b_arr0, axis=0)

    # compute brownian paths
    mu1 = 0.6
    sigma1 = 0.2
    b_arr1 = np.asarray([geom_brownian(N=N, mu=mu1, sigma=sigma1, seed=seed) for seed in seeds])
    # compute mean and std. of every step
    mean_lst1 = np.mean(b_arr1, axis=0)
    std_lst1 = np.std(b_arr1, axis=0)

    plt.rc('figure', figsize=(14, 10))

    plt.subplot(2, 1, 1)
    _draw_subplot(b_arr0, mean_lst0, std_lst0, steps, mu0, sigma0)
    plt.subplot(2, 1, 2)
    _draw_subplot(b_arr1, mean_lst1, std_lst1, steps, mu1, sigma1)

    plt.tight_layout()
    plt.savefig('p1a.pdf')
    plt.show()
