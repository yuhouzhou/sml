import numpy as np
import pandas as pd

"""
GTM: well written 
given points: 5
"""

def amortization(r, p, m, n):
    df = pd.DataFrame(index=range(m * n + 1))
    df.index.name = 'Month'

    c = p * ((r / m) / (1 - (1 + r / m) ** (-n * m)))

    c_lst = np.zeros(m * n + 1)
    i_lst = np.zeros(m * n + 1)
    p_r_lst = np.ones(m * n + 1) * p
    p_paid_lst = np.zeros(m * n + 1)

    for k in range(1, m * n + 1):
        c_lst[k] = c

        p_r_lst[k] = c * (1 - (1 + r / m) ** (-m * n + k)) / (r / m)

        i_lst[k] = p_r_lst[k - 1] * r / m

        p_paid_lst[k] = c - i_lst[k]

    df['Payment'] = c_lst
    df['Interest'] = i_lst
    df['Principal'] = p_paid_lst
    df['Remaining principal'] = p_r_lst
    df.loc['Total', :3] = df.sum(axis=0)

    return df


if __name__ == '__main__':
    print(amortization(p=400000, r=0.02, m=12, n=20))
