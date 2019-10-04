from p3 import payoff
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    s = np.arange(30, 110)
    payoff_butterfly = payoff(s=s, x=50, type='call', position='long') +\
                       payoff(s=s, x=90, type='call', position='long') + \
                       payoff(s=s, x=70, type='call', position='short') * 2
    plt.plot(s, payoff_butterfly)
    plt.xlabel('Stock Price [$]')
    plt.ylabel('Payoff [$]')
    plt.title('Butterfly Spread')
    plt.savefig('p4.pdf')
    plt.show()