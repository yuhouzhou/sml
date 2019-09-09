# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 12:28:44 2019

@author: Soeren Petrat
"""

from pylab import *

print("Hello World")

# math functions
print(cos(2))
print(exp(4))
print(3**2)
print(4<9)

#vectors
a = array([4,8])
print(a)
print(a[1])
print(len(a))

b = linspace(0,2,100)
print(exp(b))
plot(b,exp(b))

for i in cos(arange(5,9+1)):
    print(i)
    
def f(x):
    return(exp(cos(x)))

plot(b,f(b))

c1 = arange(0,10,2)
c2 = arange(9,19,2)
print(c1,c2)
print(c1*c2)
print(dot(c1,c2))

#matrices
d = array([[1,2],[3,4]])
print(d)
print(d*a)
print(dot(d,a))

print(ones((4,4)))
print(eye(4))
print(rand(4,4))

#timing
import time
t = time.clock()
b = linspace(0,10,10000)
for i in arange(1,101):
    dummy = f(b)
print(time.clock()-t," s")

#better timing
import timeit
T = timeit.Timer('f(b)', 'from __main__ import f,b')
print("100 Evaluations take ", T.timeit(100), " s")