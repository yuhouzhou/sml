from scipy.stats import binom
import matplotlib.pyplot as plt
import numpy as np


def x(n, p):
    j = np.arange(0, n + 1)
    n = np.ones(n + 1) * n
    q = 1 - p
    return (j - n * p) / np.sqrt(n * p * q)


def y(n, p):
    j = np.arange(0, n + 1)
    n = np.ones(n + 1) * n
    q = 1 - p
    return np.sqrt(n * p * q) * binom.pmf(j, n, p)


def norm_pdf(x):
    return 1 / np.sqrt(2 * np.pi) * np.e ** (-x ** 2 / 2)


def draw_subplot(ax, n, p):
    ax.plot(x(n, p), y(n, p), 'gx', label='Binomial')
    ax.plot(x(n, p), norm_pdf(x(n, p)), label='Gaussian')
    ax.legend()
    ax.set_title('$n = {}, p = {}$'.format(n, p))
    ax.set_xlabel('$x$')
    ax.set_ylabel('Probability Density')


if __name__ == '__main__':
    plt.rc('figure', figsize=(14, 10))

    fig = plt.figure()
    # fig.suptitle('Binomial Distribution vs. Gaussian Distribution')

    ax1 = fig.add_subplot(2, 2, 1)
    p = 0.5
    n = 10
    draw_subplot(ax1, n, p)

    ax2 = fig.add_subplot(2, 2, 2)
    p = 0.5
    n = 100
    draw_subplot(ax2, n, p)

    ax3 = fig.add_subplot(2, 2, 3)
    p = 0.2
    n = 10
    draw_subplot(ax3, n, p)

    ax4 = fig.add_subplot(2, 2, 4)
    p = 0.2
    n = 100
    draw_subplot(ax4, n, p)

    fig.tight_layout()

    plt.savefig('p2.pdf')
    plt.show()
