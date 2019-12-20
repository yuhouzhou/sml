"""GTM: not having precise path definition"""
import sys
sys.path.append('../')
from HW06_2019_10_21.p1a import geom_brownian
import numpy as np
import matplotlib.pyplot as plt


"""
GTM: Problem 1)
     part a)Task: generate geom Brow, find estimates, plot semilogx, comment       (+6)
            well done... 
"""     

"""GTM: for reproducible data"""
np.random.seed(1)

def sigma_mu_est(k, gbm, N):
    dt = 1 / 2 ** k
    step = int(N * dt)
    s_coarse = gbm[::step]
    r_arr = np.diff(np.log(s_coarse))
    r_mean = np.mean(r_arr)
    if len(r_arr) > 1:
        sigma_r = np.std(r_arr, ddof=1)
    else:
        sigma_r = 0
    sigma_est = sigma_r / np.sqrt(dt)
    mu_est = r_mean / dt + sigma_est ** 2 / 2
    return sigma_est, mu_est


sme_v = np.vectorize(sigma_mu_est, excluded=['gbm'])

if __name__ == '__main__':
    k = 10
    N = 2 ** k
    mu = 0.2
    sigma = 0.4
    seed = 1
    gbm = geom_brownian(N=N + 1, mu=mu, sigma=sigma, seed=seed)

    sigma_mu = sme_v(np.array(range(k + 1)), gbm=gbm, N=N)

    plt.semilogx(2 ** np.arange(k + 1), sigma_mu[0], 'X', label='$\sigma$', basex=2)
    plt.semilogx(2 ** np.arange(k + 1), sigma_mu[1], 'o', label='$\mu$', basex=2)
    plt.xlabel('$dt$')
    plt.ylabel('value')
    plt.title('Semilogx')
    plt.legend()
    # plt.savefig('p1a.pdf')
    plt.show()

    """
    sigma converges to the correct estimation; mu does not.
    """
