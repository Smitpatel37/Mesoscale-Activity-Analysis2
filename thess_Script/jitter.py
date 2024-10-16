# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 16:25:03 2017

@author: Xiaoxuan Jia
"""
###

import numpy as np
from scipy import stats
import scipy


import numpy as np

def jitter(spike_times, window_size=0.025):
    randomized_spike_times = []

    # Iterate over each neuron's spike data (each row in the 2D spike_times array)
    for neuron_spikes in spike_times:
        # Initialize list to store randomized spike times for the current neuron
        randomized_neuron_spikes = []
        
        current_window_start = 0.0  # Start of the window
        current_window_end = window_size
        
        # List to hold spikes in the current window
        spikes_in_window = []
    
        # Iterate through spike times for the current neuron and divide into windows
        for spike_time in neuron_spikes:
            # If the spike exceeds the current window, generate new random spikes for the window
            if spike_time > current_window_end:
                # Generate the same number of random spikes within the current window
                num_spikes = len(spikes_in_window)
                if num_spikes > 0:
                    new_spikes = np.sort(np.random.uniform(current_window_start, current_window_end, num_spikes))
                    randomized_neuron_spikes.extend(new_spikes)
                
                # Move to the next window
                current_window_start = current_window_end
                current_window_end += window_size
                
                # Reset spikes_in_window and start collecting spikes for the new window
                spikes_in_window = []
    
            # Add the current spike to the current window
            spikes_in_window.append(spike_time)
    
        # For the last window, randomize the remaining spikes
        if spikes_in_window:
            num_spikes = len(spikes_in_window)
            new_spikes = np.sort(np.random.uniform(current_window_start, current_window_end, num_spikes))
            randomized_neuron_spikes.extend(new_spikes)
        
        # Append randomized spike times for this neuron to the result list
        randomized_spike_times.append(np.array(randomized_neuron_spikes))
  
    return randomized_spike_times

