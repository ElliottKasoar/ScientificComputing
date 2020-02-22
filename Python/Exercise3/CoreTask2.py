#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 15:02:58 2018

@author: Elliott
"""

#Exercise 3A, Core task 2 - Diffraction pattern for sinusoidal phase grating

import numpy as np
import numpy.fft as fft
from numpy import pi, sin
import matplotlib.pyplot as plt

#Calculate edges of aperture:

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

#Function to creates sinusoidal phase grating:

def sinslit(N,L,d,delta,s,m):
    
    slit = np.zeros(int(N),dtype=complex)  #aperture array
    slit_edge1, slit_edge2 = edges(N,L,d,delta) #edges of array
        
    #Adds sinusoidal phase correction within slit width:
    
    for i in range(slit_edge1,slit_edge2):
        if i<=(int(N/2)):
            phi = (m/2)*sin(((L/2) - (i*delta))*2*pi/s)
        else:
            phi = (m/2)*sin((-(L/2) + (i*delta))*2*pi/s)
            
        slit[i] = 1.0*np.exp(1j*phi)
        
    return slit

def fft_slit(N,delta,slit,lambda_0,D,L,d):
    
    j = np.arange(int(N))
    y = (j-(N/2))*lambda_0 * D / L
    
    ft_aperture = fft.fftshift(fft.fft(slit))
    
    intensity = (2*pi/(lambda_0*D)) * delta**2 * (abs(ft_aperture))**2  
    
    return intensity, y

#Constants:
    
N = 2 ** 22  #Size of array
L = 5.0 * 10 ** (-3)  #Total extent of the aperture
D = 10.0     #Distance to screen
d = 2.0 * 10 ** (-3)    #Slit width
lambda_0 = 500.0 * 10 **(-9)  #Wavelength
s = 100.0 * 10**(-6)
m = 8.0

delta = L/N     #width of point in array


#Create aperture array and use to calculate intensity of diffraction pattern:

slit = sinslit(N,L,d,delta,s,m)
intensity, y = fft_slit(N,delta,slit,lambda_0,D,L,d)

#Plot and save graph

plt.plot(y,intensity)
plt.xlim(-0.3,0.3)
plt.xlabel("Distance along the screen / m")
plt.ylabel("Intensity")
plt.title("Diffraction pattern for a sinusoidal phase grating slit")
plt.savefig('Sin.eps', format='eps', dpi=1000)
plt.show()
