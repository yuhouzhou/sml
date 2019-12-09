import numpy as np
from scipy.linalg import solve_banded
from scipy.sparse import diags
import timeit

"""
GTM: Since n is small, you are not able to make
     good comparison of computing times... -1
given points: 5
"""

def solve_tridiagonal(ab, b):
    """solve a tri-diagonal matrix

    Args:
        ab: a matrix in diagonal ordered form
        b: the right side of the augmented matrix

    Returns: the solution of the matrix

    """
    a_lst = ab[0]
    b_lst = ab[1]
    c_lst = ab[2]
    d_lst = b
    cp_lst = []
    dp_lst = []
    cp_lst.append(c_lst[0] / b_lst[0])
    dp_lst.append(d_lst[0] / b_lst[0])
    for i, (c, d) in enumerate(zip(c_lst[1:-1], d_lst[1:-1])):
        i = i + 1
        s = b_lst[i] - a_lst[i] * cp_lst[i - 1]
        cp_lst.append(c / s)
        dp_lst.append((d_lst[i] - a_lst[i] * dp_lst[i - 1]) / s)
    n = ab.shape[1] - 1
    dp_lst.append((d_lst[n] - a_lst[n] * dp_lst[n - 1]) / (b_lst[n] - a_lst[n] * cp_lst[n-1]))
    x_lst = []
    x_lst.append(dp_lst[-1])
    for i, (cp, dp) in enumerate(zip(cp_lst[-1::-1], dp_lst[-2::-1])):
        x_lst.append((dp - cp * x_lst[i]))
    return x_lst[::-1]

def to_ab(A, l, u):
    """Convert a banded matrix to matrix diagonal ordered form

    Args:
        A: a banded matrix
        l: number of non-zero lower diagonals
        u: number of non-zero upper diagonals

    Returns: the matrix in diagonal ordered form

    """
    ab = np.zeros(shape=(l + u + 1, A.shape[1]))
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] != 0:
                ab[u + i - j, j] = A[i, j]
    return ab

if __name__ == '__main__':
    n = 5
    a = c = 1
    b = -2
    u = 1
    l = 1
    l_and_u = (l, u)
    # generate the tri-diagonal matrix.
    A = diags(diagonals=[a, b, c], offsets=[-1, 0, 1], shape=(n, n)).toarray()
    # generate a random matrix
    b = np.random.rand(n)
    ab = to_ab(A, l, u)

    x0 = solve_banded(l_and_u=l_and_u, ab=ab, b=b)
    x1 = solve_tridiagonal(ab, b)

    # print(x0, x1)
    run = 10000

    T0 = timeit.Timer('solve_banded(l_and_u=(1, 1), ab=ab, b=b)', 'from __main__ import solve_banded, l_and_u, ab, b')
    print("scipy.linalg.solve_banded, {} runs:".format(run), T0.timeit(number=run), "s")

    T1 = timeit.Timer('solve_tridiagonal(ab, b)', 'from __main__ import solve_tridiagonal,ab, b')
    print("solve_tridiagonal, {} runs:".format(run), T1.timeit(number=run), "s")
