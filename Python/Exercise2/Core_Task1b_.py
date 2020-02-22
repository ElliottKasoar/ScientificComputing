#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 07:09:56 2018

@author: Elliott
"""

#Exercise 2, Core Task 1 - calculate values of time period

import numpy as np
from numpy import sin, pi
import scipy.integrate as integrate
import matplotlib.pyplot as plt

#Sets up function to be solved by odeint. y[0] = angular displacement, y[1] = angular velocity:

def derivatives(y,t,q,F,omega):
    return [y[1], -sin(y[0]) - q*y[1] - F*sin(omega*t)]


#Solves ODE for variable initial conditions (y0, via theta[i]):
    
def ODEsolution(theta, i, q, F,t,vel0,omega):
    
    y0=[theta[i],vel0]
    y = integrate.odeint(derivatives, y0,t,args=(q,F,omega))
    return y
 
    
#Calculates time period for a given ODE solution, based on angular displacement changing sign
    
def calc_time_period(soln,t):
    
    #Find where sign of angle changes and use to calculate time this occurs:
    
    sign = np.sign(soln[:,0])
    Tperiod_time = [0]

    for j in range(len(sign)):
        if j>0 and sign[j] != sign[j-1]:
            Tperiod_time = np.append(Tperiod_time,(2*0.5*(t[j] + t[j-1]))) 
    
    #Create array of time periods and average (weighted by time elapsed)
    
    length = len(Tperiod_time)
    
    if (length > 1):
        
        Tperiod_divn = np.divide(Tperiod_time[1:length-1],np.arange(1,length-1))
        Tperiod_weight_av = np.ma.average(Tperiod_divn,weights=range(1,length-1))
        #Tperiod_av = Tperiod[length-1]/(length-1)  - non-weighted average

    else:
        
        Tperiod_weight_av = 0
        
    return Tperiod_weight_av
    

#Initial condiditions and constants

q = 0
F = 0
l = 1
g = l
vel0 = 0.0
omega = 2/3
w = l/g

theta = np.linspace(0.0,pi,200)
t = np.linspace(0.0,2000.0,2000)
Tperiodtotal = np.zeros(len(theta))
Tperiod_weight_av = np.zeros(len(theta))

for i in range(len(theta)):

    soln = ODEsolution(theta, i, q, F,t,vel0,omega)     #Solve ODE for different initial angles
        
    Tperiod_weight_av[i] = calc_time_period(soln,t)     #Calculate time period for this solution
    
    #Determine time period at theta0 = pi/2 (assumes theta array is even in size):
    
    if i == len(theta)/2:
        print("Time period at pi/2:", Tperiod_weight_av[i])
    
print("Initial time period:",Tperiod_weight_av[1])

plt.plot(theta[1:len(theta)-2],Tperiod_weight_av[1:len(Tperiod_weight_av)-2])
plt.xlabel("Initial angle / rad")
plt.ylabel("Time period / s")
plt.title("Time period as a function of initial angle")
plt.savefig('T_period.eps', format='eps', dpi=1000)
plt.show()