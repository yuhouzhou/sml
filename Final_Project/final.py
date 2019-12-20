import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date
import os
import sys
from scipy.stats import norm
from statsmodels.graphics.tsaplots import plot_acf


def j2x(j):
    """standardization

    Args:
        j: the original value of the distribution

    Returns:
        the standardized value of the distribution

    """
    return (j - np.mean(j)) / np.std(j)


def payoff(S, K):
    """
    Calculate the payoff of a call option
    :param S: the stock price
    :param K: the strike price
    :return: the payoff
    """
    return np.maximum(0, S - K)


def black_scholes(r, sigma, S, K, T):
    """
    Black-Scholes formula
    :param r: the risk-free period interest rate
    :param sigma: the volatility
    :param S: the initial stock price
    :param K: the strike price
    :param T: the maturity
    :return: the price of the option at time T = 0
    """
    x = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    return S * norm.cdf(x) - K * np.exp(-r * T) * norm.cdf(x - sigma * np.sqrt(T))

black_scholes_v = np.vectorize(black_scholes)

if __name__ == '__main__':
    name = 'TSLA'
    # today = date.today().strftime("%Y-%m-%d")
    today = '2019-12-17'

    # feature toggle
    FOR_FINAL = True
    if not FOR_FINAL:
        # add the path of Final_Project to $PYTHONPATH
        from Final_Project.data.data import generate_data
        # 1mo, 3mo, 1y, 3y, max
        stock_period = '3y'
        generate_data(name=name, day=today, stock_period=stock_period)

    # change the working directory to where this script locates
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    path_data = './data/' + today + '-' + name + '/'
    df_stock = pd.read_pickle(path_data + 'stock-' + today + '.pkl')

    """A
    
    """
    stock_history = df_stock['Close']
    N = len(stock_history)
    r_arr = np.diff(np.log(stock_history))

    n_sample = np.sort(j2x(np.random.normal(loc=0, scale=1, size=N - 1)))
    gbm_sample = np.sort(j2x(r_arr))

    plt.rc('figure', figsize=(14, 10))

    # Q-Q plot
    plt.plot(n_sample, n_sample, label='Diagonal')
    plt.scatter(n_sample, gbm_sample, s=5, label=name + ' stock price')
    plt.title('Q-Q Plot')
    plt.xlabel('scaled normal distribution')
    plt.ylabel('scaled log-returns')
    plt.legend()
    plt.grid(linestyle='--')
    plt.show()

    # autocorrelation
    plot_acf(r_arr)
    plt.show()

    # estimate volatility
    dt = 1 / N
    r_mean = np.mean(r_arr)
    sigma_r = np.std(r_arr, ddof=1)
    sigma_est = sigma_r / np.sqrt(dt)
    print("estimated volatility", sigma_est)

    """B
    
    For EU investors, the risk free interest rate is -0.5% on December 17 2019 by ECB
    For US investors, the risk free interest rate is 1.53% on December 17 2019 by 3 Month US Treasury Bill Rate
    
    """
    r = 0.0153

    """C
    
    """
    # read all option data
    (_, _, files) = next(os.walk('./data/' + today + '-' + name))
    option_files = [f for f in files if 'callop' in f]
    print(option_files)

    # pricing
    print(black_scholes(r=r,sigma=sigma_est, S=stock_history[-1], K=295, T=7/365))

    """D
    
    Binomial Tree Method:
     > Advantages:
        * Can accommodate American-style as well as European-style derivatives
        * It is difficult to apply when the payoffs depend on the past history of the state variables 
          as well as on their current values
     > Disadvantages:
        * Become computationally very time consuming when three or more state variables are involved.
    
    Black-Scholes Formula:
     > Advantages:
     > Disadvantages:
    
    Monte-Carlo Simulation:
     > Advantages:
         * It can be used when the payoff depends on the path followed by the underlying variable S 
           as well as when it depends only on the final value of S. (For example, it can be used 
           when payoffs depend on the average value of S between time 0 and time T.) 
           Payoffs can occur at several times during the life of the derivative rather than all at the end.
         * Any stochastic process for S can be accommodated.
         * The procedure can also be extended to accommodate situations where the payoff from the derivative 
           depends on several underlying market variables.
     > Disadvantages:
        * It is computationally very time consuming.
        * Cannot easily handle situations where there are early exercise opportunities.
    
    Finite Difference Method:
     > Advantages:
        * They can handle American-style as well as European-style derivatives.
        * Finite difference methods can, at the expense of a considerable increase in computer time, 
          be used when there are several state variables.
        * The implicit finite difference method  has the advantage that the user 
          does not have to take any special precautions to ensure convergence.
     > Disadvantages:
        * Cannot easily be used in situations where the payoff from a derivative depends on 
          the past history of the underlying variable.
        * Become computationally very time consuming when three or more state variables are involved.
    """
