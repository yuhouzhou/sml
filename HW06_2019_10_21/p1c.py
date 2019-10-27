import p1a, p1b
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    N = 500
    n_path = 1000
    seeds = range(0, n_path)
    steps = np.linspace(0, 1, N)
    T = 1
    S = 1


    # compute brownian paths
    rp0 = mu0 = 0.2
    sigma0 = 0.6
    b_arr0 = np.asarray([p1a.geom_brownian(N=N, mu=mu0, sigma=sigma0, seed=seed) for seed in seeds])
    # compute mean and std. of every step
    mean_lst0 = np.mean(b_arr0, axis=0)
    std_lst0 = np.std(b_arr0, axis=0)
    # compute binomial
    lattice0 = p1b.binomial_s(N - 1, rp0, sigma0, S, T)
    mean_arr0, std_arr0 = p1b.meanNstd_arr(lattice0)

    # compute brownian paths
    rp1 = mu1 = 0.6
    sigma1 = 0.2
    b_arr1 = np.asarray([p1a.geom_brownian(N=N, mu=mu1, sigma=sigma1, seed=seed) for seed in seeds])
    # compute mean and std. of every step
    mean_lst1 = np.mean(b_arr1, axis=0)
    std_lst1 = np.std(b_arr1, axis=0)
    # compute binomial
    lattice1 = p1b.binomial_s(N - 1, rp1, sigma1, S, T)
    mean_arr1, std_arr1 = p1b.meanNstd_arr(lattice1)

    plt.rc('figure', figsize=(14, 10))

    plt.subplot(2, 1, 1)
    p1a._draw_subplot(b_arr0, mean_lst0, std_lst0, steps, mu0, sigma0)
    p1b.draw_subplot(lattice0, mean_arr0, std_arr0, rp0, sigma0, T, N)
    plt.title('Geometric Brownian Paths vs. Binomial Tree Paths, with $\mu$=0.2, $\sigma$=0.6')

    plt.subplot(2, 1, 2)
    p1a._draw_subplot(b_arr1, mean_lst1, std_lst1, steps, mu1, sigma1)
    p1b.draw_subplot(lattice1, mean_arr1, std_arr1, rp1, sigma1, T, N)
    plt.title('Geometric Brownian Paths vs. Binomial Tree Paths, with $\mu$=0.6, $\sigma$=0.2')

    plt.tight_layout()
    plt.savefig('p1c.pdf')
    plt.show()
