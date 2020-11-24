#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 13:15:24 2017

@author: Kaina
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 11:28:15 2017

@author: Kaina
"""

import numpy as np
import matplotlib.pyplot as plt

fileinput = open('duck.dat', 'r')
N = 21
x = np.arange(0.9, 13.5, 0.1)

m = 5 
matrix = np.loadtxt('duck.dat')

X = np.ones((N, m+1)) ## create a N by m+1 matrix of 1's
X[:,1] = matrix[:,0]; ## put in X values of duck.dat file into column 1
for i in range(21):
    for j in range(2, m+1):
        X[i,j] = X[i,1]**j
        
XT = X.T ## traspose of X

f = matrix[:,1]; ## y values of the duck.dat values

a = np.linalg.solve(np.dot(XT,X), np.dot(XT,f))

#print a

polynomial = np.poly1d(np.flip(a,0))

y5 = polynomial(x)

###############################################################################
###############################################################################

m = 10
matrix = np.loadtxt('duck.dat')

X = np.ones((N, m+1)) ## create a N by m+1 matrix of 1's
X[:,1] = matrix[:,0]; ## put in X values of duck.dat file into column 1
for i in range(21):
    for j in range(2, m+1):
        X[i,j] = X[i,1]**j
        
XT = X.T ## traspose of X

f = matrix[:,1]; ## y values of the duck.dat values

a = np.linalg.solve(np.dot(XT,X), np.dot(XT,f))

#print a

polynomial = np.poly1d(np.flip(a,0))

y10 = polynomial(x)

###############################################################################
###############################################################################

m = 15
matrix = np.loadtxt('duck.dat')

X = np.ones((N, m+1)) ## create a N by m+1 matrix of 1's
X[:,1] = matrix[:,0]; ## put in X values of duck.dat file into column 1
for i in range(21):
    for j in range(2, m+1):
        X[i,j] = X[i,1]**j
#print matrix[:,0]
        
XT = X.T ## traspose of X

f = matrix[:,1]; ## y values of the duck.dat values

a = np.linalg.solve(np.dot(XT,X), np.dot(XT,f))

#print a

polynomial = np.poly1d(np.flip(a,0))

y15 = polynomial(x)

###############################################################################
###############################################################################

m = 20
matrix = np.loadtxt('duck.dat')

X = np.ones((N, m+1)) ## create a N by m+1 matrix of 1's
X[:,1] = matrix[:,0]; ## put in X values of duck.dat file into column 1
for i in range(21):
    for j in range(2, m+1):
        X[i,j] = X[i,1]**j
        
XT = X.T ## traspose of X

f = matrix[:,1]; ## y values of the duck.dat values

a = np.linalg.solve(X,f)

#print a

polynomial = np.poly1d(np.flip(a,0))

y20 = polynomial(x)

###############################################################################
###############################################################################

xvals = matrix[:,0]
yvals = matrix[:,1]
#print yvals, "\n"


a = np.ones(20)  ## delta n
for i in range(len(xvals)-2):
    a[i] = xvals[i+2] - xvals[i+1]
#print "below: ", a 


c = np.ones(20) ## delta n - 1
for i in range(len(xvals)-2):
    c[0] = 1
    c[i+1] = xvals[i+1] - xvals[i]
#print " \nabove: ", c

b = np.ones(21)
for i in range(len(xvals)-2):
    b[0] = 2
    b[i+1] = 2*(c[i+1]+a[i])
    b[20] = 2
#print "\nmiddle: ", b

f = np.ones(21)
for i in range(len(xvals)-2):
    f[i+1] = (((3*(a[i]))/(c[i+1]))*(yvals[i]-yvals[i+1])) + (((3*(c[i+1]))/(a[i]))*(yvals[i+2]-yvals[i+1]))
    f[0] = 1.5
    f[20] = -1.5  
#print "\nf: ",f
    
###############################################################################
###############################################################################

def TDMAsolver(a, b, c, d):
    '''
    TDMA solver, a b c d can be NumPy array type or Python list type.
    refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    and to http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
    '''
    nf = len(d) # number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, d)) # copy arrays
    for it in xrange(1, nf):
        mc = ac[it-1]/bc[it-1]
        bc[it] = bc[it] - mc*cc[it-1] 
        dc[it] = dc[it] - mc*dc[it-1]
        	    
    xc = bc
    xc[-1] = dc[-1]/bc[-1]

    for il in xrange(nf-2, -1, -1):
        xc[il] = (dc[il]-cc[il]*xc[il+1])/bc[il]

    return xc

gvals = TDMAsolver(a, b, c, f)

#print "\n", gvals


###############################################################################
###############################################################################

p0 = np.poly1d([2,-3,0,1])
p1 = np.poly1d([-2,3,0,0])
q0 = np.poly1d([1,-2,1,0])
q1 = np.poly1d([1,-1,0,0])

graph = np.ones([len(np.arange(0.9, 13.31, 0.01)),2])
graph[:,0] = np.arange(0.9, 13.31, 0.01)

for i in range(len(graph[:,0])):
    x = graph[i,0]
    exactmatch = False
    
    for j in range(len(matrix[:,0])):
        if(np.abs(matrix[j,0] - x) < 0.0001):
            exactmatch = True
            break
        elif(matrix[j,0] > x):
            j = j - 1
            break  
    if(exactmatch):
        graph[i,1] = matrix[j,1]
        continue
    
    t = (x - matrix[j,0]) / (matrix[j+1,0] - matrix[j,0])
    
    fn = matrix[j,1]
    fn1 = matrix[j+1,1]
    gn = gvals[j]
    gn1 = gvals[j+1]
    xn = matrix[j,0]
    xn1 = matrix[j+1,0]
    
    graph[i,1] = fn*p0(t) + fn1*p1(t) + (xn1-xn)*(gn*q0(t)+gn1*q1(t))
    

###############################################################################
###############################################################################

plt.figure(1,figsize=(20, 15))
plt.subplot(222)
plt.plot(matrix[:,0],matrix[:,1],'mo', label='original data')
plt.plot(graph[:,0], graph[:,1], label='Cubic Spline Interpolation')
plt.title('Cubic Spline Interpolation')
plt.legend()
plt.axis([0.0,14.0,0.0,3.0])
plt.grid(True)
plt.show()




