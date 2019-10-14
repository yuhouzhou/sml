import numpy as np
import matplotlib.pyplot as plt


"""
GTM:  
given point: 3
"""

def payoff(s, x, type, position):
    """

    :param s: float. stock price
    :param x: float. strike price
    :param type: str. type of options. 'put' or 'call'
    :param psotion: str. 'long' position or 'short' position
    :return: payoff
    """
    zero = np.zeros(len(s))
    if type == 'call' and position == 'long':
        return np.fmax(zero, s - x)
    elif type == 'put' and position == 'long':
        return np.fmax(zero, x - s)
    elif type == 'call' and position == 'short':
        return np.fmin(zero, x - s)
    elif type == 'put' and position == 'short':
        return np.fmin(zero, s - x)
    else:
        print('Invalid input!')


if __name__ == '__main__':
    s = np.arange(0, 200)
    x = 100
    types = ['call', 'put']
    positions = ['long', 'short']
    i = 0
    for type in types:
        for position in  positions:
            i += 1
            plt.subplot(len(types), len(positions), i)
            plt.plot(s, payoff(s=s, x=x, type=type, position=position))
            plt.title(position + ' a ' + type)
            plt.ylabel('Payoff')
            plt.xlabel('Price')
    plt.tight_layout()
    plt.savefig('p3.pdf')
    plt.show()
