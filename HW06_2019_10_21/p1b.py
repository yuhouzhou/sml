import numpy as np
import matplotlib.pyplot as plt


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


def path_generator(lattice):
    coordinate = [0, 0]
    path = []
    assert lattice.shape[0] == lattice.shape[1]
    x_lst = np.random.randint(0, 2, lattice.shape[0])
    for x in x_lst:
        path.append(lattice[coordinate[0], coordinate[1]])
        coordinate[x] += 1
    return path


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


def draw_subplot(lattice, mean_arr, std_arr, rp, sigma, T, N):
    plt.plot(np.linspace(0, T, N), mean_arr, c='r', label='mean')
    plt.plot(np.linspace(0, T, N), mean_arr - std_arr, c='g', label='standard deviation')
    plt.plot(np.linspace(0, T, N), mean_arr + std_arr, c='g')

    for _ in range(10):
        if _ == 0:
            plt.plot(np.linspace(0, T, N), path_generator(lattice), c='b', label='binomial tree path')
        else:
            plt.plot(np.linspace(0, T, N), path_generator(lattice), c='b')

    plt.yscale("log")
    plt.xlabel('$t$')
    plt.ylabel('$S$')
    plt.title('Ensemble of Binomial Tree Paths $S$, with $rp = {}, \sigma = {}$'.format(rp, sigma))
    plt.legend()


if __name__ == '__main__':
    N = 500
    T = 1
    S = 1

    sigma0 = 0.2
    rp0 = 0.6
    lattice0 = binomial_s(N - 1, rp0, sigma0, S, T)
    mean_arr0, std_arr0 = meanNstd_arr(lattice0)

    sigma1 = 0.6
    rp1 = 0.2
    lattice1 = binomial_s(N - 1, rp1, sigma1, S, T)
    mean_arr1, std_arr1 = meanNstd_arr(lattice1)

    plt.rc('figure', figsize=(14, 10))

    plt.subplot(2, 1, 1)
    draw_subplot(lattice0, mean_arr0, std_arr0, rp0, sigma0, T, N)

    plt.subplot(2, 1, 2)
    draw_subplot(lattice1, mean_arr1, std_arr1, rp1, sigma1, T, N)

    plt.tight_layout()
    plt.savefig('p1b.pdf')
    plt.show()
