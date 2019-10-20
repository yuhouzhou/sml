import numpy as np
import matplotlib.pyplot as plt


def brownian(N, seed):
    """Generate a standard brownian path

    Args:
        N: time steps in [0, 1]
        seed: seed

    Returns:
        one standard brownian path
    """
    np.random.seed(seed)
    dt = 1 / N
    dz = np.random.normal(loc=0, scale=1, size=N - 1) * np.sqrt(dt)
    path = np.cumsum(dz)
    return np.insert(path, 0, 0)


if __name__ == '__main__':
    N = 500
    n_path = 1000
    seeds = range(0, n_path)

    # compute brownian paths
    b_arr = np.asarray([brownian(N=500, seed=seed) for seed in seeds])
    # compute mean and std. of every step
    mean_lst = np.mean(b_arr, axis=0)
    std_lst = np.std(b_arr, axis=0)

    steps = np.linspace(0, 1, N)
    # plot mean and standard deviation
    plt.plot(steps, mean_lst, c='crimson', label='empirical mean')
    plt.plot(steps, mean_lst - std_lst, c='gold', label='empirical standard deviation')
    plt.plot(steps, mean_lst + std_lst, c='gold')
    # plot 10 sample paths
    n_sample = 10
    for i, b in enumerate(b_arr[:n_sample - 1]):
        if i == 0:
            plt.plot(steps, b, c='c', label='standard Brownian path')
        else:
            plt.plot(steps, b, c='c')

    plt.xlabel('$t$')
    plt.ylabel('$W(t)$')
    plt.title('Ensemble of Standard Brownian Paths $W(t)$')
    plt.legend()

    plt.savefig('p4.pdf')
    plt.show()
