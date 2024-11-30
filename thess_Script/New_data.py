# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 20:19:11 2024

This data set contains 87 sessions which have left ALM & left thalamus simultaneously recorded and 1 meta file. 

Meta file contains the general information of the session (mouse, data, units num etc.). To have a quick idea of number of recorded units for each probe, you may look at the field "neuron_stats". In each session, "neuron_stats" contains a 2X4 matrix, where the first numbers of each row are the number of recorded neurons.

Actual data of each session is saved individually. Every session contains 4 variables: behavior, unit_info, spk_times, matrix.
1.behavior    (n_trl  X 1 struct)
The task is standard delayed-response task in our lab. Each trial has sample (1.3s), delay (1.3s), and response.A subset of trials has photostimulation on cerebellum, which you can identify based on the field "stim_pos"----0 means control trial

2.unit_info   (n_unit X 1 struct)
Each unit has CCF coordinates, CCF_x represents LR (left has small value), CCF_y represents DV (dorsal has small value), CCF_z represents AP (anterior has small value).
If you want to quickly identify units of interests--------
For units from the ALM-recording probe, "in_target" == 1 means they are inside of ALM mask.
For units from the TH-reocrding probe, "in_target" is not useful. Instead, you can use "brain_region_ibl" to identify untis located in thalamical nuclei.

3.spk_times   (n_trl X n_unit  cell) 
Spike times are aligned to sample onset


4.matrix      (n_trl X n_unit array)
1 means the correpsonding neuron is present in that trial, 0 means not).

@author: Smit3
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import plotly.graph_objects as go
import scipy.signal as sig
import scipy.integrate as integrate
import scipy.stats as sc_stats
import seaborn as sns
import os
from scipy.optimize import curve_fit
import get_CCG_functions as gcg
from jitter import jitter
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat
import plotly.io as pio
pio.renderers.default = 'browser'
regions = ['left ALM','left Thalamus']

path = "C:\\Users\\Smit3\\Downloads\\data_for_jorg 1\\data_for_jorg/"
os.chdir(path)
directories = os.listdir(path)

# path = 'D:/Mesoscale-Activity-Analysis/NWBdata/'
# os.chdir(path)
# alldirectories = 'no'
# if alldirectories =='yes':
#     directories = os.listdir(path)
# else:
#     directories = ['sub-480135']
# for dir1 in directories:
#     if not dir1.startswith('.'):
#         sessions = gcg.get_sessions(path, dir1)
#         sub_id = dir1[4:]
#         for session in sessions: ### loops over sessions within a subdirectory 
#             os.chdir(path+'/'+dir1+'/analysis')
#             spikes_seg = {}
#             file0 = regions[0] +'_withtrials'+'sub-'+str(sub_id)+'_'+str(session)+'_allunits.pkl'
#             file1 = regions[1] +'_withtrials'+'sub-'+str(sub_id)+'_'+str(session)+'_allunits.pkl'
#             hemi = regions[0][0:4]
#             if os.path.isfile(file0)==True and os.path.isfile(file1)==True :
#                 print('session '+ (sub_id+session)+' crosscorrelation between ' +regions[0]+ ' and '+ regions[1])
#                 with open(file0, 'rb') as f:  # open a text file
#                     spikes_seg[regions[0]] = pickle.load(f) # # 
#                 with open(file1, 'rb') as f:  # open a text file
#                     spikes_seg[regions[1]] = pickle.load(f) # # 
#                 with open('session'+str(session)[4:]+'_sub'+str(sub_id)+'_stats.pkl', 'rb') as f: 
#                     stats= pickle.load(f) 
                
#                 with open(regions[0]+'_'+'sub-'+str(sub_id)+'_'+str(session)+'_CCF_Allunits.pkl', 'rb') as f:
#                     reg0_ccf = pickle.load(f)
#                 with open(regions[1]+'_'+'sub-'+str(sub_id)+'_'+str(session)+'_CCF_Allunits.pkl', 'rb') as f:
#                     reg1_ccf = pickle.load(f)
#                 print("load CCG sub"+str(sub_id)+" ses "+str(session)+"complete")
                
                
#                 timevec, spikevec_ALM, spikevec_Thal, sel_vec_cx, sel_vec_th, ALM_FR, Thal_FR  = gcg.spikestosel(spikes_seg[regions[0]], spikes_seg[regions[1]], 'all', stats, hemi, 0.05)
                
#             break
def left_right_seg(TrialData):
    l_trials = []  # List to store indices of left trials
    r_trials = []  # List to store indices of right trials
    
    # Iterate through the trials in the behavior dictionary
    for i, trial in enumerate(TrialData):
        if trial['L_hit'] == 1:  # Left trial condition
            l_trials.append(i)
        elif trial['R_hit'] == 1:  # Right trial condition
            r_trials.append(i)

    print(f"Number of Left Trials: {len(l_trials)}")
    print(f"Number of Right Trials: {len(r_trials)}")
    
    return l_trials,r_trials

def Region_seg(data):
    alm_units = [i for i, unit in enumerate(data) if unit['in_target'] == 1]
    thalamic_regions_a = {'VM','VAL'}
    thalamic_regions_b = {'MOs2/3', 'MOs6', 'MOs5'}
    thalamic_unit_a_indices = [i for i, unit in enumerate(data) if unit['brain_region_ibl'] in thalamic_regions_a]
    thalamic_unit_b_indices = [i for i, unit in enumerate(data) if unit['brain_region_ibl'] in thalamic_regions_b]

    print(f"Number of Thalamic units(Group A): {len(thalamic_unit_a_indices)}")
    print(f"Number of Thalamic units(Group B): {len(thalamic_unit_b_indices)}")
    print(f"Number of ALM units: {len(alm_units)}")
    return alm_units,thalamic_unit_a_indices, thalamic_unit_b_indices

def reformat_spikes(spikes):
    """
    Converts (n_train, n_neuron) spikes to a list of neurons' spike times across trials.
    
    Parameters:
        spikes (array-like): Array of shape (n_train, n_neurons), where each value is a list of spike times.
    
    Returns:
        List of spike times for each neuron.
    """
    n_train, n_neurons = spikes.shape
    neuron_spike_list = []
    for neuron_idx in range(n_neurons):
        aggregated_spikes = []
        for trial_idx in range(n_train):
            trial_spike_times = spikes[trial_idx, neuron_idx]
            if isinstance(trial_spike_times, (list, np.ndarray)):  # Ensure it's iterable
                aggregated_spikes.extend(trial_spike_times)
            elif isinstance(trial_spike_times, (int, float)):  # Handle single spike time
                aggregated_spikes.append(trial_spike_times)
            else:
                raise ValueError(f"Unexpected data type: {type(trial_spike_times)}")
        neuron_spike_list.append(np.array(aggregated_spikes))  # Convert to NumPy array
    return neuron_spike_list

    
def ccg_filt(binsize,filt_vec,std_th,time_th):
    numcx = np.shape(filt_vec)[1]
    numth = np.shape(filt_vec)[2]

    ccgwithpeak = []
    peakindices = []

    FR_th = 1
    peakwidth = int(50) #in units of binsize
    

    for i in range(0, numcx):#range(0,nrows):
        for j in range(0, numth):#range(0,ncolumns):
            #filt = filt_vec[i,j,:]
            CCG = filt_vec[:, i,j]-np.mean(filt_vec[:, i,j])
            filt = CCG#
            #filt = butter_bandpass_filter(CCG, 30, 700, 1/binsize, order=3)
            #filt = butter_highpass_filter(CCG, 100,1/binsize)

            filtstd = np.std(filt[(filt_time>-0.02)*(filt_time<0.02)])            
            peakCCG = np.max(filt-np.mean(filt[(filt_time>-0.1)*(filt_time<0.1)]))
            minCCG = np.min(filt-np.mean(filt[(filt_time>-0.1)*(filt_time<0.1)]))
            rangeCCG = peakCCG-minCCG
            argpeak = np.argmax(filt-np.mean(filt))
            if ALM_FR[i]>FR_th and Thal_FR[j]>FR_th:
                argmaxtest = np.argmax(filt)
                argument = range(argmaxtest-10, argmaxtest+10)
                if np.abs(filt_time[argmaxtest])<0.01:
                    integral = integrate.trapz(filt[argument], dx = binsize)
                else:
                    integral=0
                #print(np.abs(filt_time[argmaxtest]))   
                #print('integral', integral)
                #print('rangeCCG', rangeCCG)
                if integral>0.007 and rangeCCG>5 and peakCCG>std_th*filtstd:#peakCCG>std_th*filtstd:                  
                   if np.abs(filt_time[argpeak])<time_th and np.abs(filt_time[argpeak])>0.0005:
                        print('went in') 
                       #print(argpeak)
                        #range_int = range(int(argpeak-peakwidth/2), int(argpeak+peakwidth/2))
                        #print(range_int)
                        #integral = integrate.trapz(filt[range_int]-np.mean(filt), dx = binsize)
                        #print('peaksunit i,j', i,j)
                        ccgwithpeak.append(CCG)
                        peakindices.append([i,j])
                        
    return peakindices

def plotccf(ccf_coords):
    x, y, z = zip(*ccf_coords)

    fig = go.Figure(data=[go.Scatter3d(
                x=x,  # LR
                y=y,  # DV
                z=z,  # AP
                mode='markers',
                marker=dict(
                    size=5,
                    color=z,  # Use z-coordinates for color
                    colorscale='Viridis',  # Color scale
                    opacity=0.8
                ))])
    fig.update_layout(
        scene=dict(
            xaxis_title='Left-Right (LR)',
            yaxis_title='Dorsal-Ventral (DV)',
            zaxis_title='Anterior-Posterior (AP)'
        ),
        title='Interactive 3D Map of Neuron Locations',
        margin=dict(l=0, r=0, b=0, t=40))
    fig.show()
    
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, c=z, cmap='viridis', alpha=0.7)
    plt.colorbar(label='Anterior-Posterior (AP) Coordinate')
    plt.xlabel('Left-Right (LR) Coordinate')
    plt.ylabel('Dorsal-Ventral (DV) Coordinate')
    plt.title('Neuron Locations in CCF Space')
    plt.show()

dt = 0.0005

for dir in directories:
    if dir.startswith('RGT'):
        data = loadmat(path+dir, simplify_cells=True)
        
        spk = data['spk_times']
        units = data['unit_info']  #How to identify thalamus --> brain_region_ibl has codenames
        matrr = data['matrix']
        bh = data['behavior']
        
        n_trials, n_units = len(spk), len(spk[0])
        ccf_coords = [(unit['CCF_x'], unit['CCF_y'], unit['CCF_z']) for unit in units]
        
        print()
        print("the File "+str(dir)+" has :")
        print("Number of total Trials "+str(n_trials))
        
        unique_brain_regions = np.unique([unit['brain_region_ibl'] for unit in data['unit_info']])
        
        l_trial_idx, r_trial_idx = left_right_seg(bh)
        alm_idx,thal_idx_a,thal_idx_b = Region_seg(units)
        alm_spikes,thal_spikes_a,thal_spikes_b = spk[:,alm_idx],spk[:,thal_idx_a],spk[:,thal_idx_b]
        # if ( (len(thal_idx_a) != 0) and (len(thal_idx_b) != 0)):
        #     ls.append(dir)
        
        #plotccf(ccf_coords)
        
        
        
        # for i in range(10):
        #     plt.hist(spk[i][0],bins = 20)
        #     plt.show()

        # Create raster plot
        # fig, ax = plt.subplots(figsize=(12, 8))
        
        # for trial_idx, spikes in enumerate(spk):
        #     for neuron_idx, spike_train in enumerate(spikes):
        #         ax.vlines(spike_train, neuron_idx + trial_idx, neuron_idx + trial_idx + 1, color='black')
        
        # # Format plot
        # ax.set_xlabel('Time (s)')
        # ax.set_ylabel('Neurons / Trials')
        # ax.set_title('Raster Plot of Spike Times')
        # plt.show()

        alm_spikes_reformatted = reformat_spikes(alm_spikes)
        thal_spikes_reformatted = reformat_spikes(thal_spikes_b)
        if (len(alm_spikes) != 0 and len(thal_spikes_b) != 0):
            sparse1, sparse2 = gcg.getsparsematrix([spike + 2 for spike in alm_spikes_reformatted], [spike + 2 for spike in thal_spikes_reformatted])
            corr_vec, filt_time, ALM_FR, Thal_FR = gcg.cross_corr_sam(sparse1, sparse2 )
            
            peakindices_alltrials = ccg_filt(dt,corr_vec,std_th=6,time_th=0.015)
            
            peak_cx_idx = [subarray[0] for subarray in peakindices_alltrials]
            peak_thal_idx = [subarray[1] for subarray in peakindices_alltrials]
            
            for i in range(len(alm_spikes_reformatted)):
                for j in range(len(thal_spikes_reformatted)):
                    # First bar plot
                    plt.plot(filt_time,corr_vec[:,i,j], color='blue')
                    plt.title(str(dir)+'_'+str('ALM')+" "+str(i)+" -"+str('Thalamus')+" "+str(j))
                    plt.xlabel("Time (ms)")
                    plt.xlim(-0.020,0.020)
                    plt.show()
            break
