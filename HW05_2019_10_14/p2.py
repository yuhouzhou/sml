from scipy.stats import binom
import matplotlib.pyplot as plt
import numpy as np


def x(n, p):
    """Convert x coordinate of binomial distribution

    An approximation of B(n, p) is N(np, np(1-p))

    Args:
        n: number of trials
        p: success probability for each trial

    Returns:
        adjusted x coordinate

    Reference:
        https://en.wikipedia.org/wiki/Binomial_distribution#Normal_approximation

    """
    j = np.arange(0, n + 1)
    n = np.ones(n + 1) * n
    q = 1 - p
    return (j - n * p) / np.sqrt(n * p * q)


def y(n, p):
    """Convert y coordinate of binomial distribution

    An approximation of B(n, p) is N(np, np(1-p))

    Args:
        n: number of trials
        p: success probability for each trial

    Returns:
        adjusted y coordinate

    Reference:
        https://en.wikipedia.org/wiki/Binomial_distribution#Normal_approximation

    """
    j = np.arange(0, n + 1)
    n = np.ones(n + 1) * n
    q = 1 - p
    return np.sqrt(n * p * q) * binom.pmf(j, n, p)


def norm_pdf(x):
    """Compute the p.d.f of a Gaussian distribution with mean 0 variance 1

    Args:
        x: output of a random variable

    Returns:
        the probability density

    """
    return 1 / np.sqrt(2 * np.pi) * np.e ** (-x ** 2 / 2)


def draw_subplot(ax, n, p):
    """Draw subplot

    Args:
        ax: a matplotlib axes object
        n: number of trials
        p: success probability for each trial

    Returns:
        None

    """
    ax.plot(x(n, p), y(n, p), 'gx', label='Binomial')
    ax.plot(x(n, p), norm_pdf(x(n, p)), label='Gaussian')
    ax.legend()
    ax.set_title('$n = {}, p = {}$'.format(n, p))
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')


if __name__ == '__main__':
    plt.rc('figure', figsize=(14, 10))

    fig = plt.figure()
    # fig.suptitle('Binomial Distribution vs. Gaussian Distribution')

    ax1 = fig.add_subplot(2, 2, 1)
    draw_subplot(ax1, n=10, p=0.5)

    ax2 = fig.add_subplot(2, 2, 2)
    draw_subplot(ax2, n=100, p=0.5)

    ax3 = fig.add_subplot(2, 2, 3)
    draw_subplot(ax3, n=10, p=0.2)

    ax4 = fig.add_subplot(2, 2, 4)
    draw_subplot(ax4, n=100, p=0.2)

    fig.tight_layout()

    plt.savefig('p2.pdf')
    plt.show()

    """Comment
    The basic approximation generally improves as n increases and is better when p is not near to 0 or 1.
    """
