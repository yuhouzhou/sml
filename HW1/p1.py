import numpy as np
import timeit

# explicit loop
def w1(c_lst, x):
    pv = 0
    for i, c in enumerate(c_lst):
        pv += c * x**(i+1)
    return pv

# Horner's scheme
def w2(c_lst, x):
    c = c_lst[0]
    if len(c_lst) > 1:
        c_lst = c_lst[1:]
        pv = (c + w2(c_lst, x))*x
    else:
        return c*x
    return pv

def w3(c_lst, x):
    pass

def w4(c_lst, x):
    pass

if __name__ == "__main__":
    c_lst = 100 * np.arange(3, 2003)
    y = 0.05
    x = 1 / (1 + y)
    w_lst = [w1, w2, w3, w4]

    for i, w in enumerate(w_lst):
        t = timeit.Timer('w{}(c_lst, x)'.format(i+1), 'from __main__ import w{},c_lst,x'.format(i+1))
        print("value:{}, time:{} sec".format(w(c_lst, x), t.timeit(number=1000)))
