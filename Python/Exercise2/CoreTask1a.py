#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 14:42:23 2018

@author: Elliott
"""

#Exercise 2, Core Task 1 - Check energy conservation and comparison of theoretical and calculated values

import numpy as np
from numpy import sin, cos
import scipy.integrate as integrate
import matplotlib.pyplot as plt

#Sets up function to be solved by odeint. y[0] = angular displacement, y[1] = angular velocity:

def derivatives(y,t,q,F,omega):
    return [y[1], -sin(y[0]) - q*y[1] - F*sin(omega*t)]

#Functions to calculate kinetic and potential energy:
    
def kinetic_energy(y):
    return 0.5*np.square(y[:,1])

def potential_energy(y):
    return -np.cos(y[:,0])

t = np.linspace(0.0,50.0,20000)

#Initial condiditions and constants

q = 0
F= 0
l = 1
g = l
theta0 = 0.01
vel0 = 0.0
omega = 2/3
w = l/g
y0=[0.01,vel0]


#Solve ODE to calculate angular displacement and velocity:

y = integrate.odeint(derivatives, y0,t,args=(q,F,omega))

#Calculate total energy and its components:

E = kinetic_energy(y) + potential_energy(y)
KE = kinetic_energy(y)
PE = potential_energy(y)

E_average = np.average(E)
E_variation = E[0] - E[len(E)-1]

E_frac_var = E_variation / E_average

print("Fractional change in energy = ", E_frac_var)

#Calculate theoretical solutions for angular displacement and velocity and compare to those calculated 

y0_theor = theta0 * cos(w * t)
y1_theor = -theta0 * w * sin(w * t)

#Angular displacement:

#fig, ax1 = plt.subplots()
#ax1.plot(t,y[:,0],'--', color= "r", label = "Calculated angular displacement")
#ax1.plot(t,y0_theor,':', color= "b",label = "Predicted angular displacement")
#ax1.legend(loc="lower right")
#plt.xlabel("Time / s")
#plt.ylabel("Angular displacement / rad")
#plt.title("Comparison of calculated and theoretical angular displacements")
#plt.savefig('Ang_dis.eps', format='eps', dpi=1000)
#plt.show()

#Angular velocity:

#fig, ax1 = plt.subplots()
#ax1.plot(t,y[:,1],'--', color= "g", label = "Calculated angular velocity")
#ax1.plot(t,y1_theor,':', color= "k",label = "Predicted angular velocity")
#ax1.legend(loc="lower right")
#plt.xlabel("Time / s")
#plt.ylabel("Angular velocity / rad/s")
#plt.title("Comparison of calculated and theoretical angular velocities")
#plt.savefig('Ang_vel.eps', format='eps', dpi=1000)
#plt.show()


#Total energy variation:

plt.plot(t,E)
plt.xlabel("Time / s")
plt.ylabel("Total energy / J")
plt.title("Total energy as a function of time", y=1.05)
plt.ylim(-0.999,-1.001) #Comment out for zoomed in plot
plt.savefig('Energy.eps', format='eps', dpi=1000)
plt.show()

#fig, ax1 = plt.subplots()
#ax1.plot(t,E,'--', color= "r", label = "Total E")
#ax1.plot(t,KE,':', color= "k",label = "KE")
#ax1.plot(t,PE,':', color= "k",label = "PE")
#ax1.legend(loc="lower right")
#plt.ylim(-2,2)
