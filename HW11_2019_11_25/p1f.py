"""GTM: not having precise path definition"""
import sys
sys.path.append('../')
from HW06_2019_10_21.p1a import geom_brownian
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

""" Last part is waiting....
GTM: Problem 1)
     part a) +6
     part b) +4
     part c) +4
     part d) +2
     part e) +2
     part f)Task: autocorr. of time-series of log-returns, and two noisy version   (+2)
                  in one plot, comment.
        
     Subtract overall point for not proving proper path definition.. -1

given points: 19
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

    # plt.subplot(311)
    # plt.acorr(np.diff(np.log(gbm0)))
    # plt.subplot(312)
    # plt.acorr(np.diff(np.log(gbm1)))
    # plt.subplot(313)
    # plt.acorr(np.diff(np.log(gbm2)))
    # plt.show()

    ax0 = plt.subplot(311)
    plot_acf(np.diff(np.log(gbm0)), ax=ax0)
    ax1 = plt.subplot(312)
    plot_acf(np.diff(np.log(gbm1)), ax=ax1)
    ax2 = plt.subplot(313)
    plot_acf(np.diff(np.log(gbm2)), ax=ax2)
    plt.tight_layout()
    # plt.savefig('p1f.pdf')
    plt.show()

    """
    The log return of GBM without noise is not auto-correlated;
    The log return of GBM with Gaussian noise is not auto-correlated when lags > 1, if lags = 1, it shows a slight auto-correlation;
    The log return of GBM with a high frequency perturbation noise is auto-correlated, showing a periodic pattern.
    """
