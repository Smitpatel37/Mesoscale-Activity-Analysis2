# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 22:49:23 2024

@author: Smit3
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.signal as sig
import scipy.integrate as integrate
import scipy.stats as sc_stats
import seaborn as sns
import os
from scipy.optimize import curve_fit
import get_CCG_functions as gcg
from jitter import jitter

def inhomogeneous_poisson(lambda_t, T, dt):
    """
    Simulates an inhomogeneous Poisson process and returns a list of spike times.
    
    Parameters:
    - lambda_t: Function of time that returns the firing rate (intensity) at time t.
    - T: Total simulation time.
    - dt: Time step for discretization.
    
    Returns:
    - spikes: List of spike times.
    """
    # Maximum firing rate (for thinning)
    lambda_max = max(lambda_t(np.arange(0, T, dt)))  # Max of the rate function over time
    
    # Generate spikes from a homogeneous Poisson process with rate lambda_max
    spike_times_homogeneous = np.random.exponential(scale=1/lambda_max, size=int(T/dt))
    spike_times_homogeneous = np.cumsum(spike_times_homogeneous)
    spike_times_homogeneous = spike_times_homogeneous[spike_times_homogeneous < T]
    
    # Thinning based on actual rate function lambda_t
    spikes = []
    for spike_time in spike_times_homogeneous:
        if np.random.rand() < lambda_t(spike_time) / lambda_max:
            spikes.append(spike_time)
    
    return spikes  # Return spike times as a list

# Example rate function (lambda_t): A sinusoidal rate
def sinusoidal_rate(t):
    return 20 + 10 * np.sin(2 * np.pi * t / 1000)


with open('jitter_data_peak.pkl', 'rb') as f: 
    ccg = pickle.load(f)
    
corr_vec = ccg[0]
corr_vec_25 = ccg[1]
regions = ['left ALM','left Thalamus']
spikes_pooled_or = {}
spikes_pooled = {}
file0 = 'left ALM_notrialssub-480135_ses-20210226T161028_alloverlappedunits.pkl'
file1 = 'left Thalamus_notrialssub-480135_ses-20210226T161028_alloverlappedunits.pkl'
if os.path.isfile(file0)==True and os.path.isfile(file1)==True:
    #print('\nsession '+ (sub_id+session)+' crosscorrelation between ' +regions[0]+ ' and '+ regions[1])
    with open(file0, 'rb') as f:  # open a text file
        spikes_pooled_or[regions[0]] = pickle.load(f) # # 
    with open(file1, 'rb') as f:  # open a text file
        spikes_pooled_or[regions[1]] = pickle.load(f)
a=[]
b=[]

spikes_pooled_or[regions[0]] = np.expand_dims(spikes_pooled_or[regions[0]][52],axis=0)
spikes_pooled_or[regions[1]] = np.expand_dims(spikes_pooled_or[regions[1]][123],axis=0)

dt = 0.5  # Time step (in milliseconds)

# # Generate spikes
# spike_times_cx = inhomogeneous_poisson(sinusoidal_rate, int(spikes_pooled_or[regions[0]][0][-1])*100, dt)
# spike_times_th = inhomogeneous_poisson(sinusoidal_rate, int(spikes_pooled_or[regions[1]][0][-1])*100, dt)
# # Print the list of spike times
# # print(spike_times)




a.append(jitter(spikes_pooled_or[regions[0]],window_size=0.020))
b.append(jitter(spikes_pooled_or[regions[1]],window_size=0.020))
a.append(jitter(spikes_pooled_or[regions[0]],window_size=0.010))
b.append(jitter(spikes_pooled_or[regions[1]],window_size=0.010))
a.append(jitter(spikes_pooled_or[regions[0]],window_size=0.0015))
b.append(jitter(spikes_pooled_or[regions[1]],window_size=0.0015))
a.append(jitter(spikes_pooled_or[regions[0]],window_size=0.0025))
b.append(jitter(spikes_pooled_or[regions[1]],window_size=0.0025))

a= np.squeeze(a)
b = np.squeeze(b)
sparse1, sparse2 = gcg.getsparsematrix(a,b)
corr_vec_jitter, filt_time, ALM_FR, Thal_FR = gcg.cross_corr_sam(sparse1, sparse2)

 
fig, ax = plt.subplots(2, 3,figsize = (24,16))
ax = ax.flatten()
# First bar plot
ax[0].plot(filt_time,corr_vec, color='blue')
ax[0].set_title("Raw CCG")
ax[0].set_xlim(-0.020,0.020)

# Second bar plot
ax[1].plot(filt_time,(corr_vec-corr_vec_jitter[:,2,2]), color='orange')
ax[1].set_title("JitterCorrected 15ms CCG")
ax[1].set_xlim(-0.020,0.020)

ax[2].plot(filt_time,(corr_vec-corr_vec_jitter[:,1,1]), color='orange')
ax[2].set_title("JitterCorrected 10ms CCG")
ax[2].set_xlim(-0.020,0.020)

ax[3].plot(filt_time,(corr_vec-corr_vec_jitter[:,0,0]), color='orange')
ax[3].set_title("JitterCorrected 20ms CCG")
ax[3].set_xlim(-0.020,0.020)

ax[4].plot(filt_time,(corr_vec-corr_vec_jitter[:,3,3]), color='orange')
ax[4].set_title("JitterCorrected 25ms CCG")
ax[4].set_xlim(-0.020,0.020)
# Show the plots
plt.show()