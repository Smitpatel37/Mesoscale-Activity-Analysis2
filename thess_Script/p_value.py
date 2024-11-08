# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 14:53:57 2024

@author: Smit3
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy
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
    
    return spikes  

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
    #print('session '+ (sub_id+session)+' crosscorrelation between ' +regions[0]+ ' and '+ regions[1])
    with open(file0, 'rb') as f:  # open a text file
        spikes_pooled_or[regions[0]] = pickle.load(f) #
    with open(file1, 'rb') as f:  # open a text file
        spikes_pooled_or[regions[1]] = pickle.load(f)
a=[]
b=[]

spikes_pooled_or[regions[0]] = np.expand_dims(spikes_pooled_or[regions[0]][52],axis=0)
spikes_pooled_or[regions[1]] = np.expand_dims(spikes_pooled_or[regions[1]][123],axis=0)


def compute_normalized_coincidences(spike_train_1, spike_train_2, window):
    """
    Compute normalized spike coincidences between two spike trains within a specified time window.
    
    Parameters:
    - spike_train_1: List of spike times for the first neuron.
    - spike_train_2: List of spike times for the second neuron.
    - window: Time window for considering coincidences (in ms).
    
    Returns:
    - normalized_coincidences: Normalized count of spike coincidences within the window.
    """
    spike_train_1 = list(spike_train_1)
    spike_train_2 = list(spike_train_2)
    
    coincidences = 0
    for spike1 in spike_train_1:
        # Check if there is at least one spike in spike_train_2 within the coincidence window
        if any(abs(spike1 - spike2) <= window for spike2 in spike_train_2):
            coincidences += 1
    
    # Normalize by the total number of spikes in both spike trains
    normalized_coincidences = coincidences / (len(spike_train_1) + len(spike_train_2))
    return normalized_coincidences

# Define parameters
window = 5/1000 # 5 ms coincidence window
num_resamples = 2 # Number of resampled datasets

# Original data: two spike trains with different lengths
original_spike_train_1 = spikes_pooled_or[regions[0]] # Replace with actual data
original_spike_train_2 = spikes_pooled_or[regions[1]]   # Different number of spikes

# Compute the test statistic for the original data
original_coincidences = compute_normalized_coincidences(np.squeeze(original_spike_train_1), np.squeeze(original_spike_train_2), 
                                                        window)

# Generate resampled data and compute test statistic for each resample
resampled_coincidences = []
for _ in range(num_resamples):
    # Create jittered versions of the original spike trains
    resampled_train_1 = jitter(original_spike_train_1,0.020)
    resampled_train_2 = jitter(original_spike_train_2,0.020)
    # print(resampled_train_1)
    resampled_train_1 = np.squeeze(resampled_train_1)
    resampled_train_2 = np.squeeze(resampled_train_2)

    # Compute normalized coincidences for the resampled data
    coincidence_count = compute_normalized_coincidences(resampled_train_1, resampled_train_2, window)
    resampled_coincidences.append(coincidence_count)

# Calculate the p-value
p_value = sum(1 for x in resampled_coincidences if x >= original_coincidences) / num_resamples

print(f"Original Normalized Coincidences: {original_coincidences}")
print(f"p-value: {p_value}")

