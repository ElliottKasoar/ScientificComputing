#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 00:46:37 2018

@author: Elliott
"""

#Exercise 1, Core Task 1 & Supplementary Task 1

import numpy as np
from math import sqrt
from numpy import pi, random
import matplotlib.pyplot as plt

#Function to approximate integral and error through Monte-Carlo techniques

def singleintegral(N): 

  N_tot = 2**N #total number of Monte-Carlo samples
  s = pi/8  #integral limits
  V = (pi/8) **8    #volume over which we integrate
  
  #Summation of eight random numbers between 0 and 1:
  
  rand_1 = random.rand(N_tot)
  rand_2 = random.rand(N_tot)
  rand_3 = random.rand(N_tot)
  rand_4 = random.rand(N_tot)
  rand_5 = random.rand(N_tot)
  rand_6 = random.rand(N_tot)
  rand_7 = random.rand(N_tot)
  rand_8 = random.rand(N_tot)
  
  sum_arr = rand_1 + rand_2 + rand_3 + rand_4 + rand_5 + rand_6 + rand_7 + rand_8
  
  scaled_sum_arr = s*sum_arr   #Each number between 0 and s
  sin_arr = (10**6) * np.sin(scaled_sum_arr)   #evaluate sin
  sin_sqr_arr = np.square(sin_arr)   #evaluate square of sin
  
  #Calculate integral value and theoretical error
  
  average = np.average(sin_arr) 
  average_sqr = np.average(sin_sqr_arr)
  sigma = V*sqrt((average_sqr - average**2)/N_tot)
  integral = average * V
     
  return integral, sigma

#Caluclate gradient log-log plot with N points in each var array. Use data from 5th point (arbitrary)
    
def logloggradient(x,y,N):
    gradient, intercept = np.polyfit(np.log(x), np.log(y),1)
    return gradient

N=22    #log2 of number of Monte-Carlo samples
M = 50  #Number of values of N to investigate

mean_sd_arr = np.zeros(N)
alt_mean_sd_arr = np.zeros(N)
N_arr = np.zeros(N)
deviation_arr = np.zeros(N)
dev_sq_arr = np.zeros(N)
sdev_arr = np.zeros(N)  

#Loop over different values of N

for k in range(0,N):
  
  integral_arr = np.zeros(M)
  sd_arr = np.zeros(M)
  
  #Calculate integral and error for different values of N
  
  for j in range(0,M):
    integral_arr[j], sd_arr[j] = singleintegral(k)
          
  #Calculate average of theoretical errors from Monte-Carlo simulations
  mean_sd_arr[k] = np.average(sd_arr) / sqrt(M)

  #Calculate errors via scatter of Monte-Carlo simulations
  integral_av = np.average(integral_arr)
  deviation_arr = integral_arr - integral_av
  dev_sqr_arr = np.square(deviation_arr)
  sumsqr = np.sum(dev_sqr_arr)
  sdev_arr[k] = sqrt(sumsqr/M)
    
  if k == N-1:
      print("Integral = ", integral_av, "Standard deviation = ", sdev_arr[k])
  
  N_arr[k] = 2**k

#Plotting functions  
  
#1: Plot theoretical error against N in loglog space (first point not plotted since error = 0)
  
#plt.plot(np.log(N_arr), np.log(sdev_arr))
#plt.xlabel("log(N)")
#plt.ylabel("log(Error in integral)")
#plt.title("Estimate of integral errors against number of Monte-Carlo samples")
#plt.xlim(-1,15)
#plt.ylim(-4,4)
#plt.savefig("Sdev_N_plot.eps",format='eps', dpi=1000)
#grad = logloggradient(N_arr,sdev_arr,N)


#2: Plot to compare errors derived from scatter of Monte-Carlo simulations with theoretical error estimate
  
#plt.plot(mean_sd_arr[1:N-1],sdev_arr[1:N-1])
#plt.xlabel("Theoretical error from Monte-Carlo simulations")
#plt.ylabel("Error derived from scatter of simulations")
#plt.title("Comparison of errors calculated using scatter or theory")
#plt.savefig("Error_comparison_plot.eps",format='eps', dpi=1000)

#3 Same as 2, but in loglog space
  
plt.plot(np.log(mean_sd_arr[1:N-1]),np.log(sdev_arr[1:N-1]))
plt.xlabel("log(theoretical error from Monte-Carlo simulations)")
plt.ylabel("log(error derived from scatter of simulations)")
plt.title("Comparison of errors calculated using scatter or theory")
plt.savefig("Error_comparison_log_plot.eps",format='eps', dpi=1000)
grad = logloggradient(mean_sd_arr[2:N-1],sdev_arr[2:N-1],N)

print("Gradient = ",grad)

plt.show()

