#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 14:42:23 2018

@author: Elliott
"""

#Exercise 2, Core Task 2 - vary q, F

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

#Initial condiditions and constants:
    
q = 0.5
F= 1.465
l = 1
g = l
theta0 = 0.2
vel0 = 0.0
w = l/g
omega = 2/3

t = np.linspace(0.0,100.0,10000)
y0=[theta0,vel0]

#Solve ODE to calculate angular displacement and velocity, and correct for wrap-around:

y = integrate.odeint(derivatives, y0,t,args=(q,F,omega,))
corrected_y = wrap_correction(y)
            
plt.plot(t,corrected_y[:,0])
plt.xlabel("Time / s")
plt.ylabel("Angular displacement / rad")
plt.title("Angular displacement as a function of time for q = 0.5, F = 1.465")
plt.savefig('Ang_X_05_1465.eps', format='eps', dpi=1000)


#plt.plot(t,corrected_y[:,1])
#plt.xlabel("Time / s")
#plt.ylabel("Angular velocity / rad/s")
#plt.title("Angular velocity as a function of time for q = 0.5, F = 1.465")
#plt.savefig('Ang_V_05_1465.eps', format='eps', dpi=1000)


#fig, ax1 = plt.subplots()
#
#ax1.plot(t,a[:,0], color= "r", label = "Theta")
#ax1.plot(t,a[:,1], color= "r", label = "Angular veclocity")
#ax1.legend(loc="lower right")