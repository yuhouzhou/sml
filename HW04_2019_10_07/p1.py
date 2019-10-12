import numpy as np


def binomial_tree(payoff, n, rp, sigma, S, K, T):
    """
    The binomial tree via backwards induction.
    Reference: https://www.goddardconsulting.ca/option-pricing-binomial-index.html
    :param payoff: a function that takes the stock price S (possibly a vector) and strike price K as arguments and
                    returns the payoff
    :param n: the number of steps,
    :param rp: the risk-free period interest rate,
    :param sigma: the volatility,
    :param S: the initial stock price,
    :param K: the strike price,
    :param T: the maturity.
    :return: the price of the option at time T = 0
    """
    # Calculating the increase rate and decrease rate
    u = np.exp(sigma * np.sqrt(T / n))
    d = 1 / u

    # Calculating a Stock Price Lattice for the Underlying Asset Price
    y = x = np.arange(n + 1)
    xx, yy = np.meshgrid(x, y)
    lattice = S * u ** xx * d ** yy

    # Calculating the Payoff at the Expiry
    payoff_vec = np.vectorize(payoff)
    poff = payoff_vec(lattice, K=K)

    # Discount the Payoffs by Backwards Induction
    dt = T / n
    p = (np.exp(rp * dt) - d) / (u - d)
    # Payoff at maturity
    poff_T = np.fliplr(poff).diagonal()

    def BI_cal(poff_T, p, rp, dt):
        """
        Backwards Induction (recursive)
        :param poff_T: payoff at maturity
        :param p: the probability of an upwards price movement
        :param rp: the risk-free interest rate
        :param dt: the step size between time slices of the model
        :return: the option price at time zero
        """
        poff_T = list(poff_T)
        if len(poff_T) > 2:
            # Discounted payoff list
            poff_d = []
            while len(poff_T) > 1:
                v_u = poff_T[0]
                v_d = poff_T[1]
                # backwards induction formula
                v_n = np.exp(-rp * dt) * (p * v_u + (1 - p) * v_d)
                poff_d.append(v_n)
                poff_T.pop(0)
            return BI_cal(poff_d, p, rp, dt)
        else:
            v_u = poff_T[0]
            v_d = poff_T[1]
            return np.exp(-rp * dt) * (p * v_u + (1 - p) * v_d)

    return BI_cal(poff_T, p, rp, dt)


def payoff(S, K):
    return np.maximum(0, S - K)


if __name__ == '__main__':
    print(binomial_tree(payoff=payoff, n=1000, rp=0.02, sigma=0.4, S=1, K=0.8, T=1))
