#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 16:09:58 2018

@author: Elliott
"""

#Exercise 3A, Core Task 1 - Diffraction pattern from single slit in far field

import numpy as np
import numpy.fft as fft
from numpy import sinc, pi
import matplotlib.pyplot as plt

def edges(N,L,d,delta):
    edges = [(L - d)/2, (L + d)/2]  #edges of aperture slit
    edges_no = [edges[0]/delta,edges[1]/delta]    #edges of aperture in number space    
    edges_no_int = np.rint(edges_no).astype(int)    #convert to integers for array
    return edges_no_int[0], edges_no_int[1]

#Function to creates single slit aperture: 

def singleslit(N,L,d,delta):
    
    slit = np.zeros(N)  #aperture array
    slit_edge1, slit_edge2 = edges(N,L,d,delta) #edges of array
    slit[slit_edge1:slit_edge2] = 1   #set slit values in aperture array to 1
    return slit

#Calculates Fourier transform of aperture to determine amplitude of diffraction pattern

def fft_slit(N,delta,slit,lambda_0,D,L,d):
    
    j = np.arange(int(N))
    y = (j-(N/2))*lambda_0 * D / L
    
    ft_aperture = fft.fftshift(fft.fft(slit))
    
    intensity = (2*pi/(lambda_0*D)) * delta**2 * (abs(ft_aperture))**2  
    
    return intensity, y
    
#Constants:

N = 2 ** 22  #Size of array
L = 5.0 * 10 ** (-3)  #Total extent of the aperture
D = 1.0     #Distance to screen
d = 100.0 * 10 ** (-6)    #Slit width
lambda_0 = 500.0 * 10 **(-9)  #Wavelength

delta = L/N     #width of point in array

#Create aperture array and use to calculate intensity of diffraction pattern:

slit = singleslit(N,L,d,delta)
intensity, y = fft_slit(N,delta,slit,lambda_0,D,L,d)

#Calculate theoretical intensity for single slit:

sinc_arg = d * y / (lambda_0 * D)
int_theor = (2*pi/(lambda_0*D)) * (d * sinc(sinc_arg))**2    


#Plot caclculated and theoretical intensities:

fig, ax1 = plt.subplots()
ax1.plot(y,intensity,label = 'Calculated')        
ax1.plot(y,int_theor,'--', color= "r",label = "Theoretical")
ax1.legend(loc="upper right")
plt.xlim(-0.02,0.02)
plt.xlabel("Distance along the screen / m")
plt.ylabel("Intensity")
plt.title("Plot of diffraction pattern intensity for a single slit")
plt.savefig('Intensity.eps', format='eps', dpi=1000)
plt.show()
    


