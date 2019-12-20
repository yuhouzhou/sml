"""GTM: not having precise path definition"""
import sys
sys.path.append('../')
from HW05_2019_10_14.p3 import j2x
from HW06_2019_10_21.p1a import geom_brownian
import numpy as np
import matplotlib.pyplot as plt

"""
GTM: Problem 1)           
     part e)Task: QQ-plot vs. normal dist of log-returns, dists of two noisy log   (+2)
                  returns in one graph, comment.
        well done...          
"""     

if __name__ == '__main__':
    k = 10
    N = 2 ** k
    dt = 1 / 2 ** k
    mu = 0.2
    sigma = 0.4
    seed = 5
    gbm0 = geom_brownian(N=N + 1, mu=mu, sigma=sigma, seed=seed)
    g_noise = np.sqrt(dt) * np.random.normal(0, 1, N + 1)
    f = 0.3
    p_noise = np.sqrt(dt) * np.sin(2 * np.pi * f * np.arange(N + 1))
    gbm1 = gbm0 + g_noise
    gbm2 = gbm0 + p_noise

    n_sample = np.sort(j2x(np.random.normal(loc=0, scale=1, size=N)))
    g_sample = np.sort(j2x(np.diff(np.log(gbm1))))
    p_sample = np.sort(j2x(np.diff(np.log(gbm2))))
    gbm_sample = np.sort(j2x(np.diff(np.log(gbm0))))

    plt.rc('figure', figsize=(14, 10))
    plt.plot(n_sample, n_sample, label='Diagonal')
    plt.scatter(n_sample, g_sample, s=5, label='Gaussian noise')
    plt.scatter(n_sample, p_sample, s=5, label='high frequency periodic perturbation')
    plt.scatter(n_sample, gbm_sample, s=5, label='no noise')
    plt.title('Q-Q Plot')
    plt.xlabel('scaled normal distribution')
    plt.ylabel('scaled log-returns')
    plt.legend()
    plt.grid(linestyle='--')
    # plt.savefig('p1e.pdf')
    plt.show()

    """
    Log return of GBM without noise follow normal distribution.
    Adding a Gaussian noise does not change the distribution of the log return.
    Adding a high frequency periodic perturbation changes the distribution of log return.
    """
