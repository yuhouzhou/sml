import numpy as np
import matplotlib.pyplot as plt


def j2x(j):
    return (j - np.mean(j)) / np.std(j)


if __name__ == '__main__':
    n, p = 10000, 0.5
    b_sample = np.sort(j2x(np.random.binomial(n=n, p=p, size=n)))
    n_sample = np.sort(j2x(np.random.normal(loc=n, scale=p, size=n)))

    # plt.rc('figure', figsize=(14, 10))
    plt.plot(n_sample, n_sample, label='Diagonal')
    plt.scatter(n_sample, b_sample, s=5, c='g', label='Q-Q Plot')
    plt.title('Q-Q Plot')
    plt.xlabel('scaled normal distribution')
    plt.ylabel('scaled binomial distribution')
    plt.legend()
    plt.grid(linestyle='--')

    plt.savefig('p3.pdf')
    plt.show()

    """Comment
    According to the Q-Q plot, we can see normal distribution and binomial distribution are similar, when n is large 
    (here n = 10000), because most of the points fall on the diagonal. 
    """
