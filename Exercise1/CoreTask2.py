#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 15:23:02 2018

@author: Elliott
"""
#Exercise 1, Core Task 2: Plot of Coru Spiral

from scipy.integrate import quad
import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt

#Functions to return cos and sin functions to be integrated

def sin_function(x):
  return sin((pi/2) * x**2)

def cos_function(x):
  return cos((pi/2) * x**2)

#Functions to integrate functions between 0 and an upper limit

def sin_result(i):
  sinquad1, sinquad2 = quad(sin_function,0,i)
  return sinquad1
  
def cos_result(i):
  cosquad1, cosquad2 = quad(cos_function,0,i)
  return cosquad1

#Create arrays for Fresnel integrals and calculate for different values of u
  
t = np.linspace(-10,10,1000)
cos_arr = np.zeros(len(t))
sin_arr = np.zeros(len(t))

for j in range(len(t)):
  sin_arr[j] = sin_result(t[j])
  cos_arr[j] = cos_result(t[j])

#Plot Cornu spiral
  
plt.plot(cos_arr,sin_arr)
plt.xlim(-1,1)
plt.ylim(-0.8,0.8)
plt.xlabel("C(u)")
plt.ylabel("S(u)")
plt.title("Plot of Cornu spiral")
plt.savefig("Cornu_spiral.eps",format='eps', dpi=1000)
plt.show()
