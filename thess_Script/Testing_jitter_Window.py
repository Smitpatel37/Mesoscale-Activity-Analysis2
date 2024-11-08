# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 22:49:23 2024

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

mat = scipy.io.loadmat('D:\\Mesoscale-Activity-Analysis/MAT/medialALM_mask_150um3Dgauss_Bilateral.mat')  #FOR CHECKING OVERLAP WITH THALAMUS
vox = mat['F_smooth']
r, c, v = np.where(vox >= 0.25)
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

def get_overlapidx(locs,r,c,v):
                
    ccf_y,ccf_x,ccf_z = zip(*locs)
    voxel_rd = 100      #Flexibility Criteria(Radius) for overlapping
   #vox: First axis: dorsal-ventral, second axis: medial-lateral, third axis: anterior-posterior
    r = r * 20
    c = c * 20
    v = v * 20
    
    frontal = np.where(np.array(ccf_z) < 8000)[0]
    
    Thalidx = []
    #Condition for overlapping
    for u in range(len(frontal)):
        for thal in range(len(c)):
            idx = frontal[u]
            if ccf_x[idx] >= r[thal] - voxel_rd and ccf_x[idx] < r[thal] + voxel_rd and ccf_y[idx] >= c[thal] - voxel_rd and ccf_y[idx] < c[thal] + voxel_rd and ccf_z[idx] >= v[thal] - voxel_rd and ccf_z[idx] < v[thal] + voxel_rd:
                Thalidx.append(idx)
    print(np.unique(Thalidx))
    return np.unique(Thalidx)
# with open('jitter_data_peak.pkl', 'rb') as f: 
#     ccg = pickle.load(f)
    
# corr_vec = ccg[0]
# corr_vec_25 = ccg[1]
# regions = ['left ALM','left Thalamus']
# spikes_pooled_or = {}
# spikes_pooled = {}
# file0 = 'left ALM_notrialssub-480135_ses-20210226T161028_alloverlappedunits.pkl'
# file1 = 'left Thalamus_notrialssub-480135_ses-20210226T161028_alloverlappedunits.pkl'
# if os.path.isfile(file0)==True and os.path.isfile(file1)==True:
#     #print('\nsession '+ (sub_id+session)+' crosscorrelation between ' +regions[0]+ ' and '+ regions[1])
#     with open(file0, 'rb') as f:  # open a text file
#         spikes_pooled_or[regions[0]] = pickle.load(f) # # 
#     with open(file1, 'rb') as f:  # open a text file
#         spikes_pooled_or[regions[1]] = pickle.load(f)
a=[]
b=[]

# spikes_pooled_or[regions[0]] = np.expand_dims(spikes_pooled_or[regions[0]][52],axis=0)
# spikes_pooled_or[regions[1]] = np.expand_dims(spikes_pooled_or[regions[1]][123],axis=0)

dt = 0.5  # Time step (in milliseconds)

# # Generate spikes
# spike_times_cx = inhomogeneous_poisson(sinusoidal_rate, int(spikes_pooled_or[regions[0]][0][-1])*100, dt)
# spike_times_th = inhomogeneous_poisson(sinusoidal_rate, int(spikes_pooled_or[regions[1]][0][-1])*100, dt)
# # Print the list of spike times
# # print(spike_times)

dt = 0.0005
maxlag = 100e-3
Nlag = int(maxlag/dt)
filt_time = dt*np.arange(-Nlag-1, Nlag)
regions = ['left ALM','left Thalamus']

path = 'D:/Mesoscale-Activity-Analysis/NWBdata/'
os.chdir(path)
alldirectories = 'no'
if alldirectories =='yes':
   directories = os.listdir(path)
else:
    directories = ['sub-480135']
for dir1 in directories:
    if not dir1.startswith('.'):
        sessions = gcg.get_sessions(path, dir1)
        sub_id = dir1[4:]
        for session in sessions: ### loops over sessions within a subdirectory 
            os.chdir(path+'/'+dir1+'/analysis')
            spikes_seg = {}
            file0 = regions[0] +'_withtrials'+'sub-'+str(sub_id)+'_'+str(session)+'_allunits.pkl'
            file1 = regions[1] +'_withtrials'+'sub-'+str(sub_id)+'_'+str(session)+'_allunits.pkl'
            hemi = regions[0][0:4]
            if os.path.isfile(file0)==True and os.path.isfile(file1)==True :
                
                # with open(regions[0]+'_'+'sub-'+str(sub_id)+'_'+str(session)+'_CCF_Allunits.pkl', 'rb') as f:
                #     reg0_ccf = pickle.load(f)
                with open(regions[1]+'_'+'sub-'+str(sub_id)+'_'+str(session)+'_CCF_Allunits.pkl', 'rb') as f:
                    reg1_ccf = pickle.load(f)
                print("load CCG sub"+str(sub_id)+" ses "+str(session)+"complete")
                ccfloc1,unit1 = zip(*reg1_ccf)
                # print(len(unit1))
                    
                
                with open(regions[1]+'sub-'+str(sub_id)+'_'+str(session)+'CCF_overlappedunits100radius_0.25.pkl', 'rb') as file:
                      idx = pickle.load(file)
                     
                idx = np.unique(idx)
                
                ccf_y,ccf_x,ccf_z = zip(*ccfloc1)
                ccf_y , ccf_x, ccf_z = np.array(ccf_y),np.array(ccf_x),np.array(ccf_z)
                for i in list(np.sort(ccf_x)):
                    plt.title('sub-'+str(sub_id)+'_'+str(session)+'total - '+str(len(unit1))+'Idx - '+str(len(idx)))
                    plt.imshow(vox[int(i/20),:,:])
                    plt.plot(ccf_y/20,ccf_z/20,marker='*')
                    plt.show()
                
                
                # if os.path.isfile(regions[1]+'sub-'+str(sub_id)+'_'+str(session)+'CCF_overlappedunits100radius_0.25.pkl')==False: 
                #     reg1_index = get_overlapidx(ccfloc1,r,c,v)
                #     print('sub-'+str(sub_id)+'_'+str(session))
                #     print(len(unit1),len(reg1_index),100*(len(reg1_index)/len(unit1)))
                #     print('....')
                    
                #     with open(regions[1]+'sub-'+str(sub_id)+'_'+str(session)+'CCF_overlappedunits100radius_0.25.pkl', 'wb') as file:
                #         pickle.dump(reg1_index, file)
                # else:
                #     print("analysis already complete!! Skipping session.")
