import p1a, p1b
import numpy as np
import matplotlib.pyplot as plt

"""
GTM: You should provide better comments.
    Good to mention: the geometric Brownian motion is the 
    model underlying the binomial tree Stock prices, if 
    correctly calibrated, and for large N
    
    - I noticed you made some changes after deadline, you 
    are not allowed to do that. If I cannot notice the extend of
    changes you can loose all points. I am reporting everything.
    Here, I wrote you grade but I will report this as well.
    
given poitns: 10
"""

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
    b_lst0 = np.asarray([p1a.geom_brownian(N=N, mu=mu0, sigma=sigma0, seed=seed) for seed in seeds])
    # compute mean and std. of every step
    mean_lst0 = np.mean(b_lst0, axis=0)
    std_lst0 = np.std(b_lst0, axis=0)
    # compute binomial
    b_arr0 = np.asarray([p1b.binomial_s(N=N, rp=mu0, sigma=sigma0, S=S, T=T, seed=seed) for seed in seeds])
    # compute mean and std. of every step
    mean_arr0 = np.mean(b_arr0, axis=0)
    std_arr0 = np.std(b_arr0, axis=0)

    # compute brownian paths
    rp1 = mu1 = 0.6
    sigma1 = 0.2
    b_lst1 = np.asarray([p1a.geom_brownian(N=N, mu=mu1, sigma=sigma1, seed=seed) for seed in seeds])
    # compute mean and std. of every step
    mean_lst1 = np.mean(b_lst1, axis=0)
    std_lst1 = np.std(b_lst1, axis=0)
    # compute binomial
    b_arr1 = np.asarray([p1b.binomial_s(N=N, rp=mu1, sigma=sigma1, S=S, T=T, seed=seed) for seed in seeds])
    # compute mean and std. of every step
    mean_arr1 = np.mean(b_arr1, axis=0)
    std_arr1 = np.std(b_arr1, axis=0)

    plt.rc('figure', figsize=(14, 10))

    plt.subplot(2, 1, 1)
    p1a._draw_subplot(b_lst0, mean_lst0, std_lst0, steps, mu0, sigma0)
    p1b.draw_subplot(b_arr0, mean_arr0, std_arr0, steps, mu0, sigma0)
    plt.title('Geometric Brownian Paths vs. Binomial Tree Paths, with $\mu$=0.2, $\sigma$=0.6')

    plt.subplot(2, 1, 2)
    p1a._draw_subplot(b_lst1, mean_lst1, std_lst1, steps, mu1, sigma1)
    p1b.draw_subplot(b_arr1, mean_arr1, std_arr1, steps, mu1, sigma1)
    plt.title('Geometric Brownian Paths vs. Binomial Tree Paths, with $\mu$=0.6, $\sigma$=0.2')

    plt.tight_layout()
    plt.savefig('p1c.pdf')
    plt.show()

    """Description
    From fig p1c,  GBM and Binomial Tree produce similar results. 
    Their means and standard deviations overlap.
    Look at details, GBM looks a bit smoother.
    """
