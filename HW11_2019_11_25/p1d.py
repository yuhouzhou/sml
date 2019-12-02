from HW06_2019_10_21.p1a import geom_brownian
from HW11_2019_11_25.p1a import sme_v
import numpy as np
import matplotlib.pyplot as plt

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

    plt.rc('figure', figsize=(14, 10))

    plt.subplot(121)
    gbm = gbm0 + g_noise
    sigma_mu = sme_v(np.array(range(k + 1)), gbm=gbm, N=N)
    sigma_est_lst = sigma_mu[0]
    mu_est_lst = sigma_mu[1]
    plt.semilogx(2 ** np.arange(k + 1), sigma_est_lst, label='$\sigma$', basex=2)
    plt.semilogx(2 ** np.arange(k + 1), mu_est_lst, label='$\mu$', basex=2)
    plt.xlabel('$dt$')
    plt.ylabel('value')
    plt.title('GBM with Gaussian Noise')
    plt.legend()

    plt.subplot(122)
    gbm = gbm0 + p_noise
    sigma_mu = sme_v(np.array(range(k + 1)), gbm=gbm, N=N)
    sigma_est_lst = sigma_mu[0]
    mu_est_lst = sigma_mu[1]
    plt.semilogx(2 ** np.arange(k + 1), sigma_est_lst, label='$\sigma$', basex=2)
    plt.semilogx(2 ** np.arange(k + 1), mu_est_lst, label='$\mu$', basex=2)
    plt.xlabel('$dt$')
    plt.ylabel('value')
    plt.title('GBM with High Freq Periodic Perturbation')
    plt.legend()

    plt.tight_layout()
    # plt.savefig('p1d.pdf')
    plt.show()

    """
    Sigma and Mu in GBM with Gaussian noise and a periodic noise do not converge to the correct value; 
        When N gets larger, the values of them increase.
    """

