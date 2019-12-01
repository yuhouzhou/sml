from HW06_2019_10_21.p1a import geom_brownian
from HW11_2019_11_25.p1a import sigma_mu_est, sme_v
from HW11_2019_11_25.p1b import gbm_v
import numpy as np
import matplotlib.pyplot as plt

sme_v2 = np.vectorize(sigma_mu_est)

if __name__ == '__main__':
    k = 10
    N = 2 ** k
    mu = 0.2
    sigma = 0.4
    seed = 1

    gbm = geom_brownian(N=N + 1, mu=mu, sigma=sigma, seed=seed)
    sigma_est, mu_est = sigma_mu_est(k, gbm=gbm, N=N)

    seeds = range(1000)
    gbms = gbm_v(N=N + 1, mu=mu_est, sigma=sigma_est, seed=seeds)
    sigma_mu_est_lst = sme_v2(k, gbm=gbms, N=N)
    sigma_est_lst = sigma_mu_est_lst[0]
    mu_est_lst = sigma_mu_est_lst[1]

    plt.rc('figure', figsize=(14, 10))
    plt.subplot(121)
    plt.hist(sigma_est_lst, 40)
    plt.axvline(x=np.mean(sigma_est_lst), linestyle='--', color='y', Label='$\sigma$ mean')
    plt.axvline(x=np.mean(sigma_est_lst + np.std(sigma_est_lst)), linestyle='--', color='c', label='$\sigma$ std')
    plt.axvline(x=np.mean(sigma_est_lst - np.std(sigma_est_lst)), linestyle='--', color='c')
    plt.axvline(x=sigma, linestyle='--', color='r', Label='$\sigma$ true')
    plt.xlabel('$\sigma$')
    plt.legend()
    plt.subplot(122)
    plt.hist(mu_est_lst, 40)
    plt.axvline(x=np.mean(mu_est_lst), linestyle='--', color='y', Label='$\mu$ mean')
    plt.axvline(x=np.mean(mu_est_lst + np.std(mu_est_lst)), linestyle='--', color='c', label='$\mu$ std')
    plt.axvline(x=np.mean(mu_est_lst - np.std(mu_est_lst)), linestyle='--', color='c')
    plt.axvline(x=mu, linestyle='--', color='r', Label='$\mu$ true')
    plt.xlabel('$\mu$')
    plt.legend()
    plt.tight_layout()
    # plt.savefig('p1c.pdf')
    plt.show()
