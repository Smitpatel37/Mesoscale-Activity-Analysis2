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

def jitter(spike_times_2d, window_size):
    """
    Jitters a spike data series within specified windows.

    Args:
        spike_times: A 2D NumPy array containing arrays of spike times (in milliseconds).
        window_size: The size of each jitter window in milliseconds.

    Returns:
        A 1D NumPy array containing the jittered spike times.
    """

    all_jittered = []

    # Iterate over each row in the 2D array
    for spike_times in spike_times_2d:
        jittered_spike_times = np.copy(spike_times)
        jitt = []
     
        window_times = np.arange(0,spike_times[-1], window_size)
        
        # Iterate through windows based on spike times
        for start_time in window_times:
            end_time = start_time + window_size
            window_data = jittered_spike_times[(spike_times >= start_time) & (spike_times < end_time)]
            
            window_data = np.sort(window_data)
            
            # Shuffle the window data while ensuring it stays within the window
            for i in range(len(window_data)):
                jitt.append(start_time + np.random.uniform(0, window_size))

    # Append the sorted jittered data for this row to the list
        all_jittered.append(np.sort(jitt))

    # Convert the list of jittered spike times back to a 2D NumPy array
    return all_jittered

