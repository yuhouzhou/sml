import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date, datetime
import os
from scipy.stats import norm
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import kurtosistest
import re
from dateutil.relativedelta import relativedelta


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
black_scholes_vv = np.vectorize(black_scholes_v, excluded=['T'], otypes=[list])

if __name__ == '__main__':
    # TSLA is non-dividend paying, which means its American call option is actually equal European call option.
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
        # download historical stock data and its call option data
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

    # Kurtosis
    print("Kurtosis:", kurtosistest(r_arr))
    """
    
    From the QQ plot, we see the standardized distribution of log returns of TSLA stock price history has heavy tails, 
    comparing to Gaussian distribution. This conclusion is different from what we got, when we use GBM to simulate stock
    price. This heavy tails behavior is studies in literature[1, 2, 3]. A leptokurtic (i.e., long-tailed) distribution is 
    more capable to model the log return. The above result from the Kurtosis test (p-value < 1e-21) also suggest that we 
    could confidently reject the null hypothesis that the kurtosis of the population from which the sample was drawn is 
    that of the normal distribution [1]. Even though Black-Scholes model failed to catch all the mathematical features 
    of the market and has assumptions violate the real life, it is still popularly used in practice [2].
    
    References:
    [ 1 ] Empirical properties of asset returns: stylized facts and statistical issues
        http://rama.cont.perso.math.cnrs.fr/pdf/empirical.pdf
    [ 2 ] Criticism of the Black-Scholes Model: But Why Is It Still Used? (The Answer Is Simpler than the Formula).
        https://mpra.ub.uni-muenchen.de/63208/1/MPRA_paper_63208.pdf
    [ 3 ] Understanding Asset Returns
        http://www.statslab.cam.ac.uk/~chris/papers/UAR.pdf
    
    """

    # autocorrelation
    plot_acf(r_arr)
    plt.show()

    """

    The autocorrelation plot suggest that the history of stock price is not auto-correlated.

    """

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
    Ts = [(datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', file_name).group(), '%Y-%m-%d').date() -
          datetime.strptime(today, '%Y-%m-%d').date()).days / 365.25
          for file_name in option_files]

    # pricing
    df_price = pd.DataFrame()
    for i, T in enumerate(Ts):
        df = pd.read_pickle('./data/' + today + '-' + name + '/' + option_files[i])
        price = black_scholes_v(r=r, sigma=sigma_est, S=stock_history[-1], K=df.strike, T=T)
        quote = df[['bid', 'ask']].mean(axis=1)
        df_price4T = pd.concat([df.strike, pd.DataFrame(price), quote], axis=1)
        df_price4T['maturity'] = T
        df_price = pd.concat([df_price, df_price4T], axis=0)
    df_price.columns = ['strike', 'price', 'quote', 'maturity']
    df_price.set_index(['maturity', 'strike'], inplace=True)
    df_price.sort_index(inplace=True)

    print(df_price)

    """D
    
    Binomial Tree Method:
     > Advantages:
        * Can accommodate American-style as well as European-style derivatives.
        * It is difficult to apply when the payoffs depend on the past history of the state variables 
          as well as on their current values.
        * It is mathematically intuitive and can be easily visualized.
        * In practice, it give investors insights when is the possible good time to early exercise, 
          by looking at the tree
     > Disadvantages:
        * Become computationally very time consuming when three or more state variables are involved.
    
    Black-Scholes Formula:
     > Advantages:
        * It is close-formed.
        * Black-Scholes model is fast -- it lets you calculate a very large number of option prices in a very short time.
     > Disadvantages:
        * Compare to Binomial Tree method, it is more mathematically involved.
        * Cannot price American options which can early exercise.
        * Requires no dividends are paid out during the life of the option.
        * Some requirements fails the reality.
    
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
