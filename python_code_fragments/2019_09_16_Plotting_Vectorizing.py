# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 17:30:52 2019

@author: Soeren Petrat
"""


from pylab import *

### vectorizing functions (in the first argument; see also the SciPy documentation)	

def new_fct(a,b):
    if a > b:
        return a-b
    else:
        return a+b

v_new_fct = vectorize(new_fct)

print(v_new_fct([1,2,3,4,5],3))


### Plots

N = 1000 #Number of plot points

xmin = -2
xmax = 2

xx = linspace(xmin, xmax, N)

def f(x):
    return x**2
def g(x):
    return sin(5*x)

figure()
plot(xx,f(xx))
plot(xx,g(xx))
show()

rc('text', usetex=True) # Use TeX typesetting for all labels

figure()
plot(xx,f(xx), 'k', label="This is $x^2$")
plot(xx,g(xx), label="This is $\sin(5x)$")
plot(1.5,f(1.5),'ro')

xlabel('$x$')
ylabel('$f(x),g(x)$')
title("This is a test plot")
xlim(xmin,xmax)
annotate('Parabola',xy = (1,1),xytext = (-0.5,2),size = 18,arrowprops = dict(arrowstyle="->"))
legend(loc = 'upper center')
axvline(-1.5,linestyle=':')
axhline(0,color='g')

savefig('test_plot.pdf')

show()