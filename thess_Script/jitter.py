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
    """
    Randomize spike times within 25 ms windows. The new spike times within each window will be random 
    but in increasing order.
    
    Parameters:
    - spike_times: List or array of spike times (e.g., [0.1, 0.3, 0.7, 1.0])
    - window_size: Size of each window (e.g., 25 ms = 0.025 seconds).
    
    Returns:
    - randomized_spike_times: List of spike times where new random times are generated within each window.
    """
    # Initialize list for randomized spike times
    randomized_spike_times = []
    
    current_window_start = 0.0  # Start from time 0
    current_window_end = window_size
    
    # List to hold spikes in the current window
    spikes_in_window = []

    # Iterate through spike times and divide them into windows
    for spike_time in spike_times:
        # If the spike exceeds the current window, generate new random spikes for the window
        if spike_time > current_window_end:
            # Generate the same number of random spikes within the current window
            num_spikes = len(spikes_in_window)
            if num_spikes > 0:
                new_spikes = np.sort(np.random.uniform(current_window_start, current_window_end, num_spikes))
                randomized_spike_times.extend(new_spikes)
            
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
        randomized_spike_times.extend(new_spikes)

    return np.array(randomized_spike_times)

