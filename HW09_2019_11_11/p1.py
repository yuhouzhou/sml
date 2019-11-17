import numpy as np
import matplotlib.pyplot as plt
from HW06_2019_10_21.p1a import geom_brownian, _draw_subplot


def ornstein_uhlenbeck(N, T, X0, c, mu, sigma, seed):
    np.random.seed(seed)
    dt = T / N
    Xt = np.ones(N) * X0
    for i, x in enumerate(Xt[:-1]):
        Xt[i + 1] = x + (mu * (1 - c * (np.log(x)) + sigma ** 2 / 2)) * x * dt + sigma * x * np.random.normal(
            loc=0, scale=1) * np.sqrt(dt)
    return Xt


if __name__ == '__main__':
    N = 500
    T = 1
    X0 = 1
    c_lst = [0, 1]
    mu = 0.8
    sigma = 0.2
    seeds = range(50)
    steps = np.linspace(0, T, N)

    plt.rc('figure', figsize=(14, 10))
    for c in c_lst:
        Xt_lst0 = [ornstein_uhlenbeck(N=N, T=T, X0=X0, c=c, mu=mu, sigma=sigma, seed=seed) for seed in seeds]
        mean_lst0 = np.mean(Xt_lst0, axis=0)
        std_lst0 = np.std(Xt_lst0, axis=0)
        plt.subplot(211)
        _draw_subplot(Xt_lst0, mean_lst0, std_lst0, steps, mu, sigma)
        plt.title('Ensemble of Exponential Ornstein Uhlenbeck Path, with $\mu={}, \sigma={}, c={}$'.format(mu, sigma, c))

        Xt_lst1 = [geom_brownian(N=N, mu=mu, sigma=sigma, seed=seed) for seed in seeds]
        mean_lst1 = np.mean(Xt_lst1, axis=0)
        std_lst1 = np.std(Xt_lst1, axis=0)
        plt.subplot(212)
        _draw_subplot(Xt_lst1, mean_lst1, std_lst1, steps, mu, sigma)

        plt.show()

'''Discussion
When c=0, Exponential Ornstein Uhlenbeck process is the same with GBM.
When c=1, Exponential Ornstein Uhlenbeck process has smaller range of s, 
but the similar drift and variance behavior.
'''