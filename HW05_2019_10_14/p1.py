import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
from sklearn import linear_model


def stirling(n):
    return np.sqrt(2 * np.pi * n) * (n / np.e) ** n


def relative_err(true_v, comp_v):
    return (true_v - comp_v) / comp_v


if __name__ == '__main__':
    n_lst = np.arange(1, 101)
    stirling_lst = stirling(n_lst)
    factorial_lst = factorial(n_lst)
    r_err_lst = relative_err(factorial_lst, stirling_lst)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Relative Error vs. n")
    ax.set_yscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel("Relative Error")
    ax.plot(n_lst, r_err_lst)
    plt.savefig('p1.pdf')
    plt.show()

    clf = linear_model.Lasso(alpha=0.1)
    clf.fit(n_lst.reshape(-1, 1), np.log(r_err_lst))
    print(clf.coef_)