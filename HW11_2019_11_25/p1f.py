from HW06_2019_10_21.p1a import geom_brownian
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

if __name__ == '__main__':
    k = 10
    N = 2 ** k
    dt = 1 / 2 ** k
    mu = 0.2
    sigma = 0.4
    seed = 5
    gbm0 = geom_brownian(N=N + 1, mu=mu, sigma=sigma, seed=seed)
    g_noise = np.sqrt(dt) * np.random.normal(0, 1, N + 1)
    p_noise = np.sqrt(dt) * np.sin(2 * np.pi * 20e6 * np.arange(N + 1))
    gbm1 = gbm0 + g_noise
    gbm2 = gbm0 + p_noise

    # plt.subplot(311)
    # plt.acorr(np.log(gbm0), usevlines=True, normed=True, maxlags=N, lw=2)
    # plt.subplot(312)
    # plt.acorr(np.log(gbm1), maxlags=N)
    # plt.subplot(313)
    # plt.acorr(np.log(gbm2), maxlags=N)
    # plt.show()

    ax0 = plt.subplot(311)
    plot_acf(np.log(gbm0), ax=ax0, lags = N)
    ax1 = plt.subplot(312)
    plot_acf(np.log(gbm1), ax=ax1, lags = N)
    ax2 = plt.subplot(313)
    plot_acf(np.log(gbm2), ax=ax2, lags = N)
    # plt.savefig('p1f.pdf')
    plt.show()

    """
    No matter with noise or not, the autocorrelation graph shows when dt is small, 
    there is autocorrelation between nearby r_i, so increase dt to get more reliable estimate sigma.
    """
