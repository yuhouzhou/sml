import numpy as np
import matplotlib.pyplot as plt

def brownian(N, T, S0, seed):
    """Generate a standard brownian path

    Args:
        N: time steps in [0, 1]
        seed: seed

    Returns:
        one standard brownian path
    """
    np.random.seed(seed)
    dt = T / N
    dz = np.random.normal(loc=0, scale=1, size=N - 1) * np.sqrt(dt)
    dz = np.insert(dz, 0, S0)
    path = np.cumsum(dz)
    return path

def ito(N, T, S0, seed):
    w_t = brownian(N, T, S0=S0, seed=seed)
    w_0 = w_t[0:N - 1]
    w_1 = w_t[1:N]
    Ito = w_0 * (w_1 - w_0)
    return np.cumsum(Ito)


def stratonovich(N, T, S0, seed):
    w_t = brownian(N, T, S0=S0, seed=seed)
    w_0 = w_t[0:N - 1]
    w_1 = w_t[1:N]
    Stratonovich = (w_1+w_0)/2 * (w_1 - w_0)
    return np.cumsum(Stratonovich)


if __name__ == "__main__":
    N = 10000
    T = 1
    S0 = 0
    seed = 1

    w_t = brownian(N=N, T=T, S0=S0, seed=seed)
    Ito = ito(N=N, T=T, S0=S0, seed=seed)
    Stratonovich = stratonovich(N=N, T=T, S0=S0, seed=seed)

    # # Kloeden, Peter E.; Platen, Eckhard (1992). Numerical solution of stochastic differential equations.
    # # Applications of Mathematics. Berlin, New York: Springer-Verlag. ISBN 978-3-540-54062-5.
    # ito2strat = 1 / 2 * np.linspace(0, T, N-1) + Ito

    plt.plot(np.linspace(0, T, N), w_t, label='Brownian Motion')
    plt.plot(np.linspace(0, T, N-1), Ito, label='Ito Integral')
    plt.plot(np.linspace(0, T, N-1), Stratonovich, label='Stratonovich Integral')
    # plt.plot(np.linspace(0, T, N-1), ito2strat)
    plt.xlabel('$t$')
    plt.ylabel('$s$')
    plt.title('Brownian Motion & its Ito Integration and Stratonovich Integration')
    plt.legend()
    plt.savefig('p3.pdf')
    plt.show()

    """Description
    According observation, the result of Stratonovich Integral is larger than Ito Integral's.
    According to Kloeden et al. (1992) the amount is 0.5 * np.linspace(0, T, N-1).
    """