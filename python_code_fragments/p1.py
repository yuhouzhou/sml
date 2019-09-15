import numpy as np

c_lst = 100.0 * np.arange(1000,1800)
r = 0.03
x = 1/(1+r)

def w1(c_lst, x):
    pv = 0
    for i, c in enumerate(c_lst):
        pv += c * x**(i+1)
    return pv

def w4(c_lst, x):
    i = np.arange(1, len(c_lst)+1)
    return np.dot(c_lst, x**i)

print(w1(c_lst, x))
print(w4(c_lst, x))