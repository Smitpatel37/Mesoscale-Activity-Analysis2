# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 17:29:29 2024

@author: smit3
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
##### IMPORT files from jupyter notebook/NWB

os.chdir('D:/Mesoscale-Activity-Analysis/NWBdata')
def load_onesession(sub_index, session_index): 
    """
    Parameters
    ----------
    sub_index : index for subject
    session_index : index for session
    Returns
    -------
    allspikes, allunits, stats
    """
    sub_id = 455219
    session_id = [20190806143015, 20190807134913, 20190808140448,20190805152117]
    with open('sub-'+str(sub_id)+'/unitsALM_notrials'+'sub-'+str(sub_id)+'_ses-'+str(session_id[session_index])+'_alloverlappedunits.pkl', 'rb') as f:  # open a text file
        allspikes_ALM = pickle.load(f) # 
    with open('sub-'+str(sub_id)+'/unitsALM_withtrials'+'sub-'+str(sub_id)+'_ses-'+str(session_id[session_index])+'_alloverlappedunits.pkl', 'rb') as f:  # open a text file
        allunits_ALM = pickle.load(f) # 
    with open('sub-'+str(sub_id)+'/session'+str(session_id[session_index])+'_sub'+str(sub_id)+'_stats.pkl', 'rb') as f: 
        stats = pickle.load(f)    
        
    return allspikes_ALM, allunits_ALM, stats

def exponential_decay_with_offset(t, A, tau, B):
    """Exponential decay function with an offset."""
    return A * np.exp(-t / tau) + B

def calculate_tau(lags, autocorr):
    """Fit the exponential decay with offset to autocorrelation data and return tau (intrinsic timescale)."""

    # Fit exponential decay with offset to the autocorrelation data
    popt, _ = curve_fit(exponential_decay_with_offset, lags, autocorr, p0=(1000,0.1,20))

    # Extract the intrinsic timescale (tau) and offset (B)
    A, tau_c, B = popt
    return A, tau_c, B

                   
def Cal_jitter(regions):
    path = 'D:/Mesoscale-Activity-Analysis/NWBdata/'
    os.chdir(path)
    for dir1 in os.listdir():
        if not dir1.startswith('.'):
            sessions = gcg.get_sessions(path, dir1)
            sub_id = dir1[4:]
            for session in sessions: ### loops over sessions within a subdirectory 
                os.chdir(path+'/'+dir1+'/analysis')
                print('.')
                spikes_pooled = {}
                file0 = regions[0] +'_notrials'+'sub-'+str(sub_id)+'_'+str(session)+'_alloverlappedunits.pkl'
                file1 = regions[1] +'_notrials'+'sub-'+str(sub_id)+'_'+str(session)+'_alloverlappedunits.pkl'
                if os.path.isfile(file0)==True and os.path.isfile(file1)==True:
                    print('\nsession '+ (sub_id+session)+' crosscorrelation between ' +regions[0]+ ' and '+ regions[1])
                    with open(file0, 'rb') as f:  # open a text file
                        spikes_pooled[regions[0]] = pickle.load(f) # # 
                    with open(file1, 'rb') as f:  # open a text file
                        spikes_pooled[regions[1]] = pickle.load(f)
                    
                    spikes_pooled[regions[0]] = jitter(spikes_pooled[regions[0]])
                    spikes_pooled[regions[1]] = jitter(spikes_pooled[regions[1]])
                    sparse1, sparse2 = gcg.getsparsematrix(spikes_pooled[regions[0]], spikes_pooled[regions[1]])
                    corr_vec, filt_time, ALM_FR, Thal_FR = gcg.cross_corr_sam(sparse1, sparse2)
                    with open('CCG_25msjitter'+regions[0]+'-'+regions[1]+'sub-'+str(sub_id)+'_'+str(session)+'overlapped.pkl', 'wb') as f:  # open a text file
                        pickle.dump(corr_vec, f) # 
                    print('File saved : CCG jitter overlap '+sub_id+session)
                    del corr_vec
                else:
                    print('session '+ (sub_id+session)+' does not have'+regions[0]+ ' and '+ regions[1])
                    
                    
def get_allpeaks(regions,*params):
    '''
    Calculates the peaks of the crosscorrelation function

    Parameters
    ----------
    regions : list of strings, defines the regions to import crosscorrelation data from
        
    *params : list.     peak_th, norm, peakstrength, strict_contraipsi = params


    Returns
    -------
    all_contrapeaks : accumulates all contrapeaks as defined by the params
    all_ipsipeaks : accumulates all ipsipeaks as defined by the params
    all_nonselpeaks : ccumulates all nonselective peaks as defined by the params

    '''
    ###load parameters
    peak_th, norm, peakstrength, strict_contraipsi = params
    ### crosscorrelation parameters
    dt = 0.0005
    maxlag = 100e-3
    Nlag = int(maxlag/dt)
    filt_time = dt*np.arange(-Nlag-1, Nlag)
    ### variable initialization
    all_contrapeaks = []
    all_ipsipeaks = []
    all_nonselpeaks = []
    all_sel_pre = []
    all_sel_post = []
    all_efficacy = []
    global all_tau,all_A,all_B
    all_tau = []
    all_A = []
    all_B = []

    #### load data
    path = 'D:/Mesoscale-Activity-Analysis/NWBdata/'
    os.chdir(path)
    alldirectories = 'no'
    if alldirectories =='yes':
       directories = os.listdir(path)
    else:
        directories = ['sub-480928']
    for dir1 in directories:
        if not dir1.startswith('.'):
            sessions = gcg.get_sessions(path, dir1)
            sub_id = dir1[4:]
            for session in sessions: ### loops over sessions within a subdirectory 
                os.chdir(path+'/'+dir1+'/analysis')
                spikes_seg = {}
                file0 = regions[0] +'_withtrials'+'sub-'+str(sub_id)+'_'+str(session)+'_alloverlappedunits.pkl'
                file1 = regions[1] +'_withtrials'+'sub-'+str(sub_id)+'_'+str(session)+'_alloverlappedunits.pkl'
                hemi = regions[0][0:4]
                if os.path.isfile(file0)==True and os.path.isfile(file1)==True :
                    print('session '+ (sub_id+session)+' crosscorrelation between ' +regions[0]+ ' and '+ regions[1])
                    with open(file0, 'rb') as f:  # open a text file
                        spikes_seg[regions[0]] = pickle.load(f) # # 
                    with open(file1, 'rb') as f:  # open a text file
                        spikes_seg[regions[1]] = pickle.load(f) # # 
                    with open('session'+str(session)[4:]+'_sub'+str(sub_id)+'_stats.pkl', 'rb') as f: 
                        stats= pickle.load(f) 
                    with open('CCG'+regions[0]+'-'+regions[1]+'sub-'+str(sub_id)+'_'+str(session)+'overlapped.pkl', 'rb') as f:  # open a text file
                        corr_vec = pickle.load(f)
                    with open('CCG_25msjitter'+regions[0]+'-'+regions[1]+'sub-'+str(sub_id)+'_'+str(session)+'overlapped.pkl', 'rb') as f:  # open a text file
                        corr_vec_jitter = pickle.load(f)
                    print("load CCG sub"+str(sub_id)+" ses "+str(session)+"complete")
                    
                    
                    timevec, spikevec_ALM, spikevec_Thal, sel_vec_cx, sel_vec_th, ALM_FR, Thal_FR  = gcg.spikestosel(spikes_seg[regions[0]], spikes_seg[regions[1]], 'all', stats, hemi, 0.05)
                    
                    ccgs_norm = gcg.CCG_norm(corr_vec, filt_time, ALM_FR, Thal_FR, norm)
                    peakindices_alltrials, ccgwithpeak_alltrials, allpeaks_alltrials, peak_sel_alltrials, allcounters_alltrials = gcg.peak_filt(dt, filt_time, ccgs_norm, sel_vec_cx, sel_vec_th,ALM_FR, Thal_FR,peak_th, 0.015, peakstrength)        

                    peak_cx_idx = [subarray[0] for subarray in peakindices_alltrials]
                    peak_thal_idx = [subarray[1] for subarray in peakindices_alltrials]
                    
                    for i in range(len(spikes_seg[regions[0]])):
                        for j in range(len(spikes_seg[regions[1]])):
                            if(max(corr_vec[:,i,j])>9):
                                plt.plot(filt_time,corr_vec[:,i,j])
                                plt.plot(filt_time,corr_vec_jitter[:,i,j])
                                plt.xlim(-0.015,0.015)
                                plt.title(str(sub_id)+'_'+str(session)+" "+str(regions[0])+" "+str(i)+" -"+str(regions[1])+" "+str(j))
                                plt.figure()
                                    
                
                    # sel_pre, sel_post, efficacy = np.transpose(peak_sel_alltrials)
                    # all_sel_pre = np.hstack([all_sel_pre, sel_pre])
                    # all_sel_post = np.hstack([all_sel_post, sel_post])
                    # all_efficacy= np.hstack([all_efficacy, efficacy])
                    # contra_peaks = allpeaks_alltrials['peaks_contracontra'] + strict_contraipsi*allpeaks_alltrials['peaks_contranon']
                    # ipsi_peaks = allpeaks_alltrials['peaks_ipsiipsi'] + strict_contraipsi*allpeaks_alltrials['peaks_ipsinon']
                    # all_contrapeaks = np.hstack([all_contrapeaks, contra_peaks])
                    # all_ipsipeaks = np.hstack([all_ipsipeaks, ipsi_peaks])
    # plt.figure()
    # sns.kdeplot(all_contrapeaks, fill = 'True', color = 'steelblue').set(xlim=0)     
    # sns.kdeplot(all_ipsipeaks, fill = 'True', color = 'red').set(xlim=0)
    # plt.figure()
    # binwidth = 0.006
    # plt.hist(all_contrapeaks, bins=np.arange(min(all_contrapeaks), max(all_contrapeaks) + binwidth, binwidth), color = 'steelblue')    
    # plt.hist(all_ipsipeaks, bins=np.arange(min(all_ipsipeaks), max(all_ipsipeaks) + binwidth, binwidth), color='red')
    # print('mean_contra', np.mean(all_contrapeaks))    
    # print('mean_ipsi', np.mean(all_ipsipeaks))
    # plt.figure()
    # plt.scatter(all_sel_pre, all_sel_post, s = 2000*all_efficacy)
    # plt.axvline(x = 0, color = 'k')
    # plt.axhline(y = 0, color = 'k')
    # return all_contrapeaks, all_ipsipeaks, all_nonselpeaks, all_sel_pre, all_sel_post, all_efficacy         
                
 
params = [6,"both", "integral", 0 ]#peak_th, norm, peak_strength, strict_contraipsi = params
#all_contrapeaks, all_ipsipeaks, all_nonselpeaks = get_allpeaks(*params)

#Cal_jitter(['left ALM', 'left Thalamus'])
#get_allCCG(['left Thalamus', 'left Thalamus'])
# all_contrapeaks_L, all_ipsipeaks_L, all_nonselpeaks_L, all_sel_pre, all_sel_post, all_efficacy = get_allpeaks(['left ALM', 'left ALM'], *params)
'''all_contrapeaks_R, all_ipsipeaks_R, all_nonselpeaks_R = get_allpeaks(['right ALM', 'right ALM'], *params)
all_contrapeaks = np.hstack([all_contrapeaks_R,all_contrapeaks_L])
all_ipsipeaks = np.hstack([all_ipsipeaks_R,all_ipsipeaks_L])
sns.kdeplot(all_contrapeaks, fill = 'True', color = 'steelblue').set(xlim=0)     
sns.kdeplot(all_ipsipeaks, fill = 'True', color = 'red').set(xlim=0)
plt.figure()
binwidth = 5
plt.hist(all_contrapeaks, bins=np.arange(min(all_contrapeaks), max(all_contrapeaks) + binwidth, binwidth), color = 'steelblue')    
plt.hist(all_ipsipeaks, bins=np.arange(min(all_ipsipeaks), max(all_ipsipeaks) + binwidth, binwidth), color='red')
'''
get_allpeaks(['left ALM', 'left Thalamus'], *params)

# plt.hist(all_tau,bins=30)
# plt.hist(all_A,bins=30)
# plt.hist(all_B,bins=30)
'''
==>  Change by jorge for Autocorrelograms

--> Change in Peakwidth,
--> dt,binsize in three function "getsparseformat", "getsparsematris" and "crosscorrsam"
--> and changing maxLag in crosscorrsam to get more points
'''

###### allspikes contains all spike times for all units. allunits is segmented into trials, as defined by the export file
    
    
### criteria/parameters
## before export: hit vs miss, all trial types? 
## after export
### selectivity: epoch where selectivity is calculated, p value
### CCG calculation: timebin, lag. 
### peakindices and CCG analysis:    peak to std, height vs integral, normalization (e.g., pre vs post)  
### definition of contra/ipsi peak? 
    
    
#### obtain selectivity and firing rate from spike trains segmented into trials
#timevec, spikevec_ALM, spikevec_Thal, sel_vec_cx, sel_vec_th, ALM_FR, Thal_FR  = spikestosel(ALM_units, Thal_units, 'all', stats, 'left', 0.05)

#### obtain spiketrain matrix (time, units), from spike trains that are not segmented intro trials
#sparse, sparse_th = getsparsematrix(allspikes_ALM, allspikes_Thal)

### obtain cross correlogram from sparse matrices for cortex and/or thalamus. ccgs_norm applies a normalization, pre or postsynaptic
#ccgs_allunits, filt_time, ALM_FR, Thal_FR = cross_corr_sam(sparse, sparse)  
#ccgs_norm = CCG_norm(ccgs, filt_time, ALM_FR, Thal_FR, 's')

### obtain different peaks
#peakindices_alltrials, ccgwithpeak_alltrials, allpeaks_alltrials, peak_sel_alltrials, allcounters_alltrials = peak_filt(0.0001, filt_time, ccgs_norm_allunits, sel_vec_cx, sel_vec_cx,ALM_FR, ALM_FR,5.5, 0.015, 'cxandcx')#allpeaksipsi = allpeaks_alltrials['peaks_ipsiipsi'] + allpeaks_alltrials['peaks_ipsinon']
#allpeakscontra = allpeaks_alltrials['peaks_contracontra'] + allpeaks_alltrials['peaks_contranon']


### plots