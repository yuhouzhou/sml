import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def binomial_s(N, rp, sigma, S, T):
    """Calculate Stock Price using Binomial Tree

    Args:
        N: the number of steps
        rp: the risk-free period interest rate
        sigma: the volatility
        S: the initial stock price
        T: the maturity

    Returns:
        stock price lattice
    """
    # Calculating the increase rate and decrease rate
    u = np.exp(sigma * np.sqrt(T / N))
    d = 1 / u

    # Calculating a Stock Price Lattice for the Underlying Asset Price
    y = x = np.arange(N + 1)
    xx, yy = np.meshgrid(x, y)
    return S * u ** xx * d ** yy


def path_generator(lattice, seed):
    np.random.seed(seed)
    coordinate = [0, 0]
    path = []
    assert lattice.shape[0] == lattice.shape[1]
    x_lst = np.random.randint(0, 2, lattice.shape[0])
    for x in x_lst:
        path.append(lattice[coordinate[0], coordinate[1]])
        coordinate[x] += 1
    return pd.Series(path)


def diag_index(n):
    x_arr = np.arange(n).astype(int)
    y_arr = (np.ones(n) * (n - 1) - x_arr).astype(int)
    return x_arr, y_arr


def meanNstd_arr(lattice):
    def mean_n(lattice, n):
        di = diag_index(n)
        return np.mean(lattice[di])

    def std_n(lattice, n):
        di = diag_index(n)
        return np.std(lattice[di])

    assert lattice.shape[0] == lattice.shape[1]
    N = lattice.shape[0]

    mean_lst = []
    std_lst = []
    for n in range(1, N + 1):
        mean_lst.append(mean_n(lattice, n))
        std_lst.append(std_n(lattice, n))
    mean_arr = np.asarray(mean_lst)
    std_arr = np.asarray(std_lst)
    return mean_arr, std_arr


def draw_subplot(path_df, mean_arr, std_arr, rp, sigma, T, N):
    plt.plot(np.linspace(0, T, N), mean_arr, c='r', label='mean')
    plt.plot(np.linspace(0, T, N), mean_arr - std_arr, c='g', label='standard deviation')
    plt.plot(np.linspace(0, T, N), mean_arr + std_arr, c='g')

    path_df.columns = np.arange(0, path_df.shape[1])
    for column in path_df.columns[::100]:
        if column == 0:
            plt.plot(np.linspace(0, T, N), path_df[column], c='b', label='binomial tree path')
        else:
            plt.plot(np.linspace(0, T, N), path_df[column], c='b')

    plt.xlabel('$t$')
    plt.ylabel('$S$')
    plt.title('Ensemble of Binomial Tree Paths $S$, with $rp = {}, \sigma = {}$'.format(rp, sigma))
    plt.legend()


if __name__ == '__main__':
    N = 500
    T = 1
    S = 1
    seeds = np.arange(0, 1000)

    sigma0 = 0.2
    rp0 = 0.6
    lattice0 = binomial_s(N - 1, rp0, sigma0, S, T)
    path_df0 = pd.DataFrame()
    for seed in seeds:
        path_df0 = pd.concat([path_df0, path_generator(lattice0, seed)], axis=1)
    mean_arr0 = path_df0.mean(axis = 1)
    std_arr0 = path_df0.std(axis = 1)

    sigma1 = 0.6
    rp1 = 0.2
    lattice1 = binomial_s(N - 1, rp1, sigma1, S, T)
    path_df1 = pd.DataFrame()
    for seed in seeds:
        path_df1 = pd.concat([path_df1, path_generator(lattice1, seed)], axis=1)
    mean_arr1 = path_df1.mean(axis=1)
    std_arr1 = path_df1.std(axis=1)

    plt.rc('figure', figsize=(14, 10))

    plt.subplot(2, 1, 1)
    draw_subplot(path_df0, mean_arr0, std_arr0, rp0, sigma0, T, N)

    plt.subplot(2, 1, 2)
    draw_subplot(path_df1, mean_arr1, std_arr1, rp1, sigma1, T, N)

    plt.tight_layout()
    plt.savefig('p1b.pdf')
    plt.show()

    """Description
    From fig p1c, subplot 1 shows that when mu = 0.2, sigma = 0.6, GBM and Binomial Tree produce similar results.
    When mu = 0.6, sigma 0.2, they are different, because binomial tree model can't incorporate drift rate.
    """
