#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 14:42:23 2018

@author: Elliott
"""

#Exercise 2, Supplementary Task 1 - sensitivity to initial conditions

import numpy as np
from numpy import sin, pi
import scipy.integrate as integrate
import matplotlib.pyplot as plt

#Sets up function to be solved by odeint. y[0] = angular displacement, y[1] = angular velocity:

def derivatives(y,t,q,F,omega):
    return [y[1], -sin(y[0]) - q*y[1] + F*sin(omega*t)]

#Correct for wrap-around at theta > |pi|:

def wrap_correction(y):
    
    for i in range(len(y)):
        
        if y[i,0] >=0:
            y[i,0] = y[i,0] % (2*pi)
            if y[i,0] > pi:
                y[i,0] -= (2*pi)
        else:
            y[i,0] = -1 * ((-1 * y[i,0]) % (2*pi))
            if y[i,0] < -1 * pi:
                y[i,0] += (2*pi) 
    return y


#Initial condiditions and constants
    
q = 0.5
F= 1.2
l = 1
g = l
theta0 = 0.2
vel0 = 0.0
w = l/g
omega = 2/3

t = np.linspace(0.0,100.0,10000)

#Set up initial conditions with slightly different angular displacement

y0 = [theta0,vel0]
y1 = [theta0+0.00001,vel0]

#Solve ODE to calculate angular displacement and velocity for each set of initial conditions:

y0_soln = integrate.odeint(derivatives, y0,t,args=(q,F,omega,))
y1_soln = integrate.odeint(derivatives, y1,t,args=(q,F,omega,))
y0_soln_corr = wrap_correction(y0_soln)
y1_soln_corr = wrap_correction(y1_soln)


fig, ax1 = plt.subplots()
ax1.plot(t,y0_soln_corr[:,0],'-', color= "r", label = "Theta = 0.2")
ax1.plot(t,y1_soln_corr[:,0],'--', color= "g", label = "Theta 0.20002")
ax1.set_xlabel("Time / s")
ax1.set_ylabel("Angular displacement / rad")
ax1.set_title("Comparison of displacement for small variation in initial conditions")
ax1.legend(loc="lower left")
plt.savefig('Initial_comp.eps', format='eps', dpi=1000)
plt.show()