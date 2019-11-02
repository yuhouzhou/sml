import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
#from sklearn import linear_model


"""
GTM: Indeed, the slope must be very close to -1.
     The slope can be also found by plotting a line as I did.
given points: 5
"""

def stirling(n):
    return np.sqrt(2 * np.pi * n) * (n / np.e) ** n


def relative_err(true_v, comp_v):
    return (true_v - comp_v) / comp_v


if __name__ == '__main__':
    N = 100
    n_lst = np.arange(1, N+1)
    stirling_lst = stirling(n_lst)
    factorial_lst = factorial(n_lst)
    r_err_lst = relative_err(factorial_lst, stirling_lst)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Relative Error vs. n")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel("Relative Error")
    ax.plot(n_lst, r_err_lst,'*')
    ax.plot(n_lst,1./(12*n_lst), label = 'line 1/(12n)')
    #plt.savefig('p1.pdf')
    plt.legend()
    plt.show()

    clf = linear_model.Lasso(alpha=0.1)
    clf.fit(np.log(n_lst.reshape(-1, 1)), np.log(r_err_lst))
    print('Rate of Convergence:', clf.coef_)
    print('Coefficient of 1/n:',
        np.exp(np.log(N)+np.log(r_err_lst)[N-2])+clf.coef_*(np.log(N)-np.log(N-1)))

    """Comment
    The plot of the logarithm of the relative error is a straight line with slope -0.8850278.
    The next order of the Stirling approximation is f(n)/(12n). 
    """
