#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 15:39:19 2018

@author: Elliott
"""

#Exercise 1, Supplementary Task 2

from scipy.integrate import quad
import numpy as np
from numpy import sqrt, sin, cos, pi
import matplotlib.pyplot as plt


#Functions to return cos and sin functions to be integrated:

def sin_function(x):
  return sin((pi/2) * x**2)

def cos_function(x):
  return cos((pi/2) * x**2)

#Functions to integrate cos and sin functions between 0 and an upper limit:

def sin_result(i):
  sinquad1, sinquad2 = quad(sin_function,0,i)
  return sinquad1
  
def cos_result(i):
  cosquad1, cosquad2 = quad(cos_function,0,i)
  return cosquad1

#Constants:

width = 0.1 #slit width
D = 1   #distance to screen
points = 2000   #number of points to plot
max_x = 0.25    #maximum distance along screen
wavelength = 0.01   #wavelength of monochromatic light

#Arrays for position, amplitude and phase at a given point along the screen:

position = np.linspace(-max_x,max_x,points)
amp = np.zeros(len(position))
phase = np.zeros(len(position))

scale = sqrt(2/(wavelength*D))  #scale factor to use Fresnel functions for new integral

#Calculate integral between limits at different positions along the screen:

for i in range(len(position)):
  a = sin_result(scale*((width/2) - position[i]))
  b= cos_result(scale*((width/2) - position[i]))
  c = sin_result(scale*(-(width/2) - position[i]))
  d = cos_result(scale*(-(width/2) - position[i]))
  
  amp[i] = sqrt(((a-c)**2 + (b-d)**2))
  
  phase[i] = np.arctan((a-c)/(b-d))
  
normalised_amp = amp / np.max(amp)


#Plot normalised amplitude of diffraction pattern against distance along screen:
  
plt.plot(position,normalised_amp)
plt.xlabel("Distance along screen / m")
plt.ylabel("Normalised amplitude")
plt.title("Fresnel diffraction amplitude for distance to screen of 1m")
plt.savefig("Amp_1.eps",format='eps', dpi=1000)

#Plot phase of diffraction pattern against distance along screen:

#plt.plot(position,phase)
#plt.xlabel("Distance along screen / m")
#plt.ylabel("Phase / rad")
#plt.title("Phase of Fresnel diffraction for distance to screen of 1m")
#plt.savefig("Phase_1.eps",format='eps', dpi=1000)



plt.show()