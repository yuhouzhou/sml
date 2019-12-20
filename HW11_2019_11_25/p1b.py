#from HW11_2019_11_25.p1a import sme_v
from p1a import sme_v
"""GTM: not having precise path definition"""
import sys
sys.path.append('../')
from HW06_2019_10_21.p1a import geom_brownian
import numpy as np
import matplotlib.pyplot as plt

"""
GTM: Problem 1)
     part b)Task: geom Brown, mean and std of estimates, loglog plot, variance     (+4)
                  for sigma_hat, comment
            well done...
"""     

gbm_v = np.vectorize(geom_brownian, otypes=[list])


def gbm_sigma_mu(k, gbm, N):
    return sme_v(np.array(range(k + 1)), gbm=gbm, N=N)


gsm_v = np.vectorize(gbm_sigma_mu, otypes=[list])

if __name__ == '__main__':
    k = 10
    N = 2 ** k
    mu = 0.2
    sigma = 0.4
    seeds = np.array(range(3000))
    gbms = gbm_v(N=N + 1, mu=mu, sigma=sigma, seed=seeds)

    gbms_sigma_mu = np.concatenate(gsm_v(k, gbms, N))
    gbms_sigma = gbms_sigma_mu[::2]
    gbms_mu = gbms_sigma_mu[1::2]
    sigma_mean = np.mean(gbms_sigma, axis=0)
    mu_mean = np.mean(gbms_mu, axis=0)
    sigma_std = np.std(gbms_sigma, axis=0)
    mu_std = np.std(gbms_mu, axis=0)
    sigma_var_sqrt = np.sqrt(sigma_mean ** 2 / (2 * 2 ** np.arange(k + 1)))

    plt.loglog(2 ** np.arange(k + 1), sigma_mean, '*', label='$\hat{\sigma}$ mean')
    plt.loglog(2 ** np.arange(k + 1), mu_mean, 'X', label='$\hat{\mu}$ mean')
    plt.loglog(2 ** np.arange(k + 1), sigma_std, 'o', label='$\hat{\sigma}$ std')
    plt.loglog(2 ** np.arange(k + 1), mu_std, 'v', label='$\hat{\mu}$ std')
    plt.loglog(2 ** np.arange(k + 1), sigma_var_sqrt, 'D', label='$\hat{\sigma}$ std (formula)')
    plt.xlabel('$N$')
    plt.title('$\hat{\sigma}\ &\ \hat{\mu}$')
    plt.legend(loc=3)
    #plt.savefig('p1b.pdf')
    plt.show()

    """
    My statistics reproduce the result.
    The variance of the estimate for µ stay nearly constant as N increases.
    """
