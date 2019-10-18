from scipy.stats import binom
import matplotlib.pyplot as plt
import numpy as np

def x(j, n, p):
    q = 1 - p
    return (j - n * p) / np.sqrt(n * p * q)

def y(j, n, p):
    q = 1 - p
    return np.sqrt(n * p * q) * binom.pmf(j, n, p)

if __name__ == '__main__':
    plt.plot()