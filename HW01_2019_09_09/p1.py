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

def w5(c_lst, x):
    pv = 0
    for i in np.arange(len(c_lst)-1, -1, -1):
        pv = c_lst[i]+(x*pv)
    pv *= x
    return pv

# polyval
def w3(c_lst, x):
    c_lst = np.append(c_lst[::-1], 0)
    return np.polyval(c_lst, x)

# dot product
def w4(c_lst, x):
    i = np.arange(1, len(c_lst) + 1)
    return np.dot(c_lst, x ** i)

if __name__ == "__main__":
    c_lst = 100 * np.arange(1000, 1800)
    r = 0.03
    x = 1 / (1 + r)

    w_dic = {'explicit loop':w1, "Horner's scheme (Recursive)":w2,
             "Horner's scheme (Iterative)": w5,
             "polyval":w3, "dot product":w4}

    for k, v in w_dic.items():
        t = timeit.Timer('{}(c_lst, x)'.format(v.__name__), 'from __main__ import {},c_lst,x'.format(v.__name__))
        print("{}:\n\tvalue:{}, time:{} sec".format(k, v(c_lst, x), t.timeit(number=1000)))
