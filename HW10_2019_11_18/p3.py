import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.linalg import solve_banded
from scipy import stats


def black_scholes(rp, sigma, X, T, S):
    x = (np.log(S / X) + (rp + sigma ** 2 / 2) * T) / (sigma * T ** (1 / 2))
    return S * norm.cdf(x) - X * np.exp(-rp * T) * norm.cdf(x - sigma * T ** (1 / 2))


def black_scholes_e(x_max, M, N, sigma, r, S0, K):
    """explicit ﬁnite diﬀerence scheme

    Args:
        x_max: the maximum of the asset price
        M: the number of stock price interval
        N: the number of time interval
        sigma: volatility of the underlying asset
        r: risk-free interest rate
        S0: the initial stock price
        K: strike price of the option

    Returns:
        the price grid
    """

    dx = (2 * x_max) / N
    dt = 1 / M
    x = np.linspace(-x_max, x_max, N + 1)
    S = np.exp(x)
    S_max = np.exp(x_max)

    """Boundary conditions
    The particular derivative that is obtained when the equation is solved depends on the boundary conditions that are
    used. These specify the values of the derivative at the boundaries of possible values of S and t.
    In our case, the key boundary condition is f = max(S - K, 0) when t = T.
    """

    # boundary condition when stock price equals maximum of stock price
    V_m_N = (S_max - K) * np.ones(M)
    # boundary condition when stock price equals zero
    V_m_0 = np.zeros(M)
    # boundary condition when time equals T, over all possibles S
    V_M_n = np.fmax(S - K, np.zeros(N + 1))

    grid = np.zeros((N + 1, M))
    grid[:, -1] = V_M_n
    grid[0, :] = V_m_0
    grid[N, :] = V_m_N

    a = dt * (sigma ** 2 / (2 * dx ** 2) - (r - sigma ** 2 / 2) / (2 * dx))
    b = 1 - dt * (sigma ** 2 / dx ** 2 + r)
    c = dt * (sigma ** 2 / (2 * dx ** 2) + (r - sigma ** 2 / 2) / (2 * dx))

    for m in range(M - 1, 0, -1):
        grid[1:N - 1, m - 1] = a * grid[0:N - 2, m] + b * grid[1:N - 1, m] + c * grid[2:N, m]

    return grid

def black_scholes_i(x_max, M, N, sigma, r, S0, K):
    """implicit ﬁnite diﬀerence scheme

    Args:
        x_max: the maximum of the asset price
        M: the number of stock price interval
        N: the number of time interval
        sigma: volatility of the underlying asset
        r: risk-free interest rate
        S0: the initial stock price
        K: strike price of the option

    Returns:
        the price grid
    """

    dx = (2 * x_max) / N
    dt = 1 / M
    x = np.linspace(-x_max, x_max, N + 1)
    S = np.exp(x)
    S_max = np.exp(x_max)

    V_m_N = (S_max - K) * np.ones(M)
    V_m_0 = np.zeros(M)
    V_M_n = np.fmax(S - K, np.zeros(N + 1))

    grid = np.zeros((N + 1, M))
    grid[:, -1] = V_M_n
    grid[0, :] = V_m_0
    grid[N, :] = V_m_N

    a = (-sigma ** 2 / 2 * dt / dx ** 2 + (r - sigma ** 2 / 2) * dt / (2 * dx))
    b = (sigma ** 2 / dx ** 2 * dt + r * dt + 1)
    c = (-sigma ** 2 * dt / (2 * dx ** 2) - (r - sigma ** 2 / 2) * dt / (2 * dx))

    a_lst = np.ones(N - 1) * a
    b_lst = np.ones(N - 1) * b
    c_lst = np.ones(N - 1) * c

    for m in range(M - 1, 0, -1):
        strip = np.zeros(N - 1)
        strip[-1] = grid[N, m - 1] * a
        D = grid[1:N, m] - strip
        pre = solve_banded((1, 1), np.r_[[a_lst], [b_lst], [c_lst]], D)
        grid[1:N, m - 1] = pre

    return grid


def err(x_max, M, N, sigma, r, S0, K, bs):
    """

    Args:
        x_max: the maximum of the asset price
        M: the number of stock price interval
        N: the number of time interval
        sigma: volatility of the underlying asset
        r: risk-free interest rate
        S0: the initial stock price
        K: strike price of the option
        bs: the method function used

    Returns:
        the error
    """
    x = np.linspace(-x_max, x_max, N + 1)
    S = np.exp(x)
    grid = bs(x_max, M, N, sigma, r, S0, K)
    return abs(black_scholes(r, sigma, K, 1, S[int(N / 2)]) - grid[int(N / 2), 0])

err_vectorized = np.vectorize(err)

if __name__ == '__main__':
    x_max = 8
    M = np.arange(1, 100, 2)
    delta_t = 1 / M
    N = 400
    sigma = 0.3
    r = 0.05
    S0 = 1
    K = 0.8

    explicit_error = err_vectorized(x_max, M, N, sigma, r, S0, K, black_scholes_e)
    implicit_error = err_vectorized(x_max, M, N, sigma, r, S0, K, black_scholes_i)

    fig = plt.figure()
    plt.loglog(delta_t[::-1], explicit_error[::-1], '*', label='Explicit error')
    plt.title('Error vs. $\Delta t$')
    plt.xlabel('$\Delta t$')
    plt.ylabel('$error$')
    plt.legend()
    plt.show()
    print('Method stability depending on dx is Visible')

    plt.loglog(delta_t[::-1], explicit_error[::-1], '*', label='Explicit error')
    plt.loglog(delta_t[::-1], implicit_error[::-1], '*', label='Implicit error')
    plt.title('Error vs. $\Delta t$')
    plt.xlabel('$\Delta t$')
    plt.ylabel('$error$')
    plt.legend()
    plt.show()

    plt.loglog(delta_t[::-1], implicit_error[::-1], '*', label='Implicit error')
    plt.title('Error vs. $\Delta t$')
    plt.xlabel('$\Delta t$')
    plt.ylabel('$error$')
    plt.legend()
    plt.show()

    M = np.array(1.7 ** np.arange(3, 20), dtype=int)
    dt = 1 / M
    N = np.array(1.7 ** np.arange(3, 20) - 1, dtype=int)
    error_convergence_implicit = []

    for i in range(len(M)):
        error = err(x_max, M[i], N[i], sigma, r, S0, K, black_scholes_i)
        error_convergence_implicit.append(error)

    slope, intercept, r_value, p_value, std_err = stats.linregress(np.log(dt), np.log(error_convergence_implicit))

    plt.plot(np.log(dt), np.log(error_convergence_implicit), '*', label='log Error')
    plt.plot(np.log(dt), np.log(dt) * slope + intercept, label='Slope equals ' + str(round(slope, 2)))
    plt.title('Order of Convergence')
    plt.xlabel('log(dt)')
    plt.ylabel('$log(error)$')
    plt.legend()
    plt.show()
    print('Rate of convergence of the implicit method = {}'.format(slope))
