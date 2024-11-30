# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 19:53:08 2024

@author: Smit3
"""
# PLOTS FOR THESIS (FINAL VERSION)

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
    global all_tau,all_A,all_B
    all_tau = []
    all_A = []
    all_B = []

    #### load data
    path = 'D:/Mesoscale-Activity-Analysis/NWBdata/'
    os.chdir(path)
    alldirectories = 'yes'
    if alldirectories =='yes':
        directories = os.listdir(path)
    else:
        directories = ['sub-456772']
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
                    print('session '+ (sub_id+session)+' crosscorrelation between ' +regions[0]+ ' and '+ regions[1])
                    with open(file0, 'rb') as f:  # open a text file
                        spikes_seg[regions[0]] = pickle.load(f) # # 
                    with open(file1, 'rb') as f:  # open a text file
                        spikes_seg[regions[1]] = pickle.load(f) # # 
                    with open('session'+str(session)[4:]+'_sub'+str(sub_id)+'_stats.pkl', 'rb') as f: 
                        stats= pickle.load(f) 
                    with open('CCG_Bin_0.5ms_'+regions[0]+'-'+regions[1]+'sub-'+str(sub_id)+'_'+str(session)+'allunits.pkl', 'rb') as f:  # open a text file
                        corr_vec = pickle.load(f)
                    with open('CCG_20msjitter_'+regions[0]+'-'+regions[1]+'sub-'+str(sub_id)+'_'+str(session)+'allunits.pkl', 'rb') as f:  # open a text file
                        corr_vec_jitter = pickle.load(f)
                    with open(regions[0]+'_'+'sub-'+str(sub_id)+'_'+str(session)+'_CCF_Allunits.pkl', 'rb') as f:
                        reg0_ccf = pickle.load(f)
                    with open(regions[1]+'_'+'sub-'+str(sub_id)+'_'+str(session)+'_CCF_Allunits.pkl', 'rb') as f:
                        reg1_ccf = pickle.load(f)
                    print("load CCG sub"+str(sub_id)+" ses "+str(session)+"complete")
                    
                    CCF_allregions0 = gcg.unzip_CCF(reg0_ccf)
                    CCF_allregions1 = gcg.unzip_CCF(reg1_ccf)
                    timevec, spikevec_ALM, spikevec_Thal, sel_vec_cx, sel_vec_th, ALM_FR, Thal_FR  = gcg.spikestosel(spikes_seg[regions[0]], spikes_seg[regions[1]], 'all', stats, hemi, 0.05)
                    
                    ccgs_norm = gcg.CCG_norm(corr_vec, filt_time, ALM_FR, Thal_FR, norm)
                    peakindices_alltrials, CCF, ccgwithpeak_alltrials, allpeaks_alltrials, peak_sel_alltrials, allcounters_alltrials,allsession = gcg.peak_filt(dt, filt_time, ccgs_norm, sel_vec_cx, sel_vec_th, CCF_allregions0, CCF_allregions1, ALM_FR, Thal_FR,peak_th,session, 0.015, peakstrength)        

                    # print(peakindices_alltrials)
                    peak_cx_idx = [subarray[0] for subarray in peakindices_alltrials]
                    peak_thal_idx = [subarray[1] for subarray in peakindices_alltrials]
                    cx_unit = [subarray[2] for subarray in peakindices_alltrials]
                    thal_unit = [subarray[3] for subarray in peakindices_alltrials]
                    unit_reg0 = {peak_cx_idx[i]: cx_unit[i] for i in range(len(peak_cx_idx))}
                    unit_reg1 = {peak_thal_idx[i]: thal_unit[i] for i in range(len(peak_thal_idx))}
                    
                    for i in peak_cx_idx:
                        for j in peak_thal_idx:
                            if(max(corr_vec[:,i,j])>15):
                                
                                plt.plot(filt_time,corr_vec[:,i,j])
                                plt.title(str(sub_id)+'_'+str(session)+" "+str(regions[0])+" "+str(unit_reg0[i])+" -"+str(regions[1])+" "+str(unit_reg1[j]))
                                plt.xlim(-0.02,0.02)
                                plt.xlabel("Time (ms)")
                                plt.ylabel("Raw CCG")
                                plt.show()
                                # fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                                # fig.suptitle(str(sub_id)+'_'+str(session)+" "+str(regions[0])+" "+str(unit_reg0[i])+" -"+str(regions[1])+" "+str(unit_reg1[j]))
                                # # First bar plot
                                # ax[0].plot(filt_time,corr_vec[:,i,j], color='blue')
                                # ax[0].set_title("Raw CCG")
                                # ax[0].set_xlabel("Time (ms)")
                                # ax[0].set_xlim(-0.02,0.02)
                                
                                # # Second bar plot
                                # ax[1].plot(filt_time,(corr_vec[:,i,j]-corr_vec_jitter[:,i,j]), color='orange')
                                # ax[1].set_title("JitterCorrected CCG")
                                # ax[1].set_xlabel("Time (ms)")
                                # ax[1].set_xlim(-0.02,0.02)
                                # # Show the plots
                                # plt.show()
                                
                        
params = [6,"both", "integral", 0 ]
get_allpeaks(['left ALM', 'left Thalamus'], *params)
