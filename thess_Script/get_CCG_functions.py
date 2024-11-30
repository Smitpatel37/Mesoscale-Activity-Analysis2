 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 17:13:46 2024

@author: jaramillo
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.signal as sig
import scipy.integrate as integrate
import scipy.stats as sc_stats
import seaborn as sns
import os
 


def get_sessions(path, dir1):
    
    '''
    obtains the sessions from a given subdirectory
    '''
    sessions = []
    os.chdir((path+'/'+dir1))
    for file in (os.listdir()):
        if  file.endswith(".nwb"):
            indexfile = file.find('ses-')
            session_id= file[indexfile:int(indexfile+19)]
            sessions +=[session_id]
    return sessions

def unzip_CCF(rg_ccf):
    ccfloc1,unit1 = zip(*rg_ccf)
    x,y,z = zip(*ccfloc1)
    result = []
    for i in range(len(unit1)):  # Loop over indices
        combined_list = [x[i], y[i], z[i]]  # Combine elements from x, y, and z
        result.append([ unit1[i], combined_list ]) 
    return result

def peak_filt(binsize, filt_time, filt_vec, sel_vec_cx, sel_vec_th,CCF_allregions0, CCF_allregions1, ALM_FR, Thal_FR,std_th, session, time_th, peakstrength):
    '''
    

    Parameters
    ----------
    binsize : self explanatory
    filt_time : time vector to plot CCG
    filt_vec : cross correlogram CCG
    sel_vec_cx : vector of selectivities for cortical neurons
    sel_vec_th : vector of selectivities for thalamic neurons
    ALM_FR : ALM units firing rate
    Thal_FR : Thal units firing rate
    std_th : Standard deviation threshold in CCG to capture a peak 
    time_th : time below which peaks are considered
    case : way to interpret thalamocortical connectionscontra-contra, etc. (thal_only, cx_only, both, either)

    Returns
    -------
    peakindices : [i,j,sel_vec_cx[i], sel_vec_th[j], peakfilt,filt_time[argpeak] , ALM_FR[i], Thal_FR[j]])
    peak_sel: [sel_vec_cx[i][0], sel_vec_th[j][0], peakfilt])
  
    ccgwithpeak : 
    contra_peaks : synaptic weights for contra connections
    ipsi_peaks : synaptic weights for ipsi connections
    nonsel_peaks : synaptic weights for non-selective connections

    '''
    
    
    numcx = np.shape(filt_vec)[1]
    numth = np.shape(filt_vec)[2]
    listcx = CCF_allregions0
    listth = CCF_allregions1
    #print('listcxshape', np.shape(listcx))
    #print(listcx)
    ccgwithpeak = []
    peakindices = []
    CCF = []

    FR_th = 1
    peakwidth = int(50) #in units of binsize
    peak_sel = []
    contra_peaks = []
    ipsi_peaks =  []
    nonsel_peaks = []
    counter_contracontra =0
    peaks_contracontra = []
    counter_ipsiipsi = 0
    peaks_ipsiipsi = []
    counter_mixed= 0
    peaks_mixed = []
    counter_ipsinon =0 
    peaks_ipsinon = []
    counter_contranon =0
    peaks_contranon = []
    counter_nonnon = 0
    peaks_nonnon= []
    all_sessions = []

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
                        if peakstrength=="amp":
                            peakfilt = peakCCG
                        elif peakstrength=="integral":
                            peakfilt = integral#integral#peakfilt#[peakfilt,filt_time[argpeak]]
                        #peakindices.append([i,j,sel_vec_cx[i], sel_vec_th[j], listcx[i], listth[j], peakfilt,filt_time[argpeak] , ALM_FR[i], Thal_FR[j]])
                        peakindices.append([i,j,listcx[i][0], listth[j][0]])
                        CCF.append([listcx[i][1], listth[j][1]])
                        all_sessions.append(session)

                        peak_sel.append([sel_vec_cx[i], sel_vec_th[j], peakfilt])
                        if sel_vec_cx[i][0]=='contra' and sel_vec_th[j][0]=='contra':
                            counter_contracontra+=1
                            peaks_contracontra.append(peakfilt)
                        if sel_vec_cx[i][0]=='ipsi' and sel_vec_th[j][0]=='ipsi':
                            counter_ipsiipsi+=1
                            peaks_ipsiipsi.append(peakfilt)

                        if (sel_vec_cx[i][0]=='ipsi' and sel_vec_th[j][0]=='contra') or (sel_vec_cx[i][0]=='contra' and sel_vec_th[j][0]=='ipsi') :
                            counter_mixed+=1
                            peaks_mixed.append(peakfilt)

                        if (sel_vec_cx[i][0]=='ipsi' and sel_vec_th[j][0]=='non-sel') or (sel_vec_cx[i][0]=='non-sel' and sel_vec_th[j][0]=='ipsi') :
                              counter_ipsinon+=1
                              peaks_ipsinon.append(peakfilt)

                        if (sel_vec_cx[i][0]=='contra' and sel_vec_th[j][0]=='non-sel') or (sel_vec_cx[i][0]=='non-sel' and sel_vec_th[j][0]=='contra') :
                              counter_contranon+=1
                              peaks_contranon.append(peakfilt)

                        if (sel_vec_cx[i][0]=='non-sel' and sel_vec_th[j][0]=='non-sel'):# or ([sel_vec_cx[i][0]=='non-sel' and sel_vec_th[j][0]]=='contra') :
                              counter_nonnon+=1   
                              peaks_nonnon.append(peakfilt)

              
              
    contra_peaks = np.array(contra_peaks)
    ipsi_peaks = np.array(ipsi_peaks)    
    nonsel_peaks = np.array(nonsel_peaks)  
    allcounters = [counter_contracontra, counter_ipsiipsi, counter_mixed, counter_contranon, counter_ipsinon, counter_nonnon]
    allpeaks =  {'peaks_contracontra':peaks_contracontra, 'peaks_ipsiipsi':peaks_ipsiipsi, 'peaks_mixed':peaks_mixed,'peaks_contranon': peaks_contranon, 'peaks_ipsinon':peaks_ipsinon, 'peaks_nonnon':peaks_nonnon}
    return peakindices, CCF, ccgwithpeak, allpeaks, peak_sel, allcounters, all_sessions


# def peak_filt(binsize, filt_time, filt_vec, sel_vec_cx, sel_vec_th,ALM_FR, Thal_FR,std_th, time_th, peakstrength):
#     '''
    

#     Parameters
#     ----------
#     binsize : self explanatory
#     filt_time : time vector to plot CCG
#     filt_vec : cross correlogram CCG
#     sel_vec_cx : vector of selectivities for cortical neurons
#     sel_vec_th : vector of selectivities for thalamic neurons
#     ALM_FR : ALM units firing rate
#     Thal_FR : Thal units firing rate
#     std_th : Standard deviation threshold in CCG to capture a peak 
#     time_th : time below which peaks are considered
#     case : way to interpret thalamocortical connectionscontra-contra, etc. (thal_only, cx_only, both, either)

#     Returns
#     -------
#     peakindices : [i,j,sel_vec_cx[i], sel_vec_th[j], peakfilt,filt_time[argpeak] , ALM_FR[i], Thal_FR[j]])
#     peak_sel: [sel_vec_cx[i][0], sel_vec_th[j][0], peakfilt])
  
#     ccgwithpeak : 
#     contra_peaks : synaptic weights for contra connections
#     ipsi_peaks : synaptic weights for ipsi connections
#     nonsel_peaks : synaptic weights for non-selective connections

#     '''
    
    
#     numcx = np.shape(filt_vec)[1]
#     numth = np.shape(filt_vec)[2]
#     #print(numcx)
#     #print(numth)
#     ccgwithpeak = []
#     peakindices = []
#     FR_th = 1.5
#     peakwidth = 0.025 #in units of binsize
#     peak_sel = []
#     contra_peaks = []
#     ipsi_peaks =  []
#     nonsel_peaks = []
#     counter_contracontra =0
#     peaks_contracontra = []
#     counter_ipsiipsi = 0
#     peaks_ipsiipsi = []
#     counter_mixed= 0
#     peaks_mixed = []
#     counter_ipsinon =0 
#     peaks_ipsinon = []
#     counter_contranon =0
#     peaks_contranon = []
#     counter_nonnon = 0
#     peaks_nonnon= []

#     for i in range(0, numcx):#range(0,nrows):
#         for j in range(0, numth):#range(0,ncolumns):
#             #filt = filt_vec[i,j,:]
#             CCG = filt_vec[:, i,j]-np.mean(filt_vec[:, i,j])
#             filt = CCG#
#             filtstd = np.std(filt[(filt_time>-0.02)*(filt_time<0.02)])            
#             peakCCG = np.max(filt-np.mean(filt[(filt_time>-0.1)*(filt_time<0.1)]))
#             minCCG = np.min(filt-np.mean(filt[(filt_time>-0.1)*(filt_time<0.1)]))
#             rangeCCG = peakCCG-minCCG
#             argpeak = np.argmax(filt-np.mean(filt))
#             if ALM_FR[i]>FR_th and Thal_FR[j]>FR_th:
#                 argmaxtest = np.argmax(filt)
#                 argument = range(argmaxtest-10, argmaxtest+10)
#                 if np.abs(filt_time[argmaxtest])<0.01:
#                     integral = integrate.trapz(filt[argument], dx = binsize)
#                 else:
#                     integral=0
#                 #print(np.abs(filt_time[argmaxtest]))   
#                 #print('integral', integral)
#                 #print('rangeCCG', rangeCCG)
#                 if integral>0.007 and rangeCCG>5 and peakCCG>std_th*filtstd:#peakCCG>std_th*filtstd:                  
#                    if np.abs(filt_time[argpeak])<time_th and np.abs(filt_time[argpeak])>0.0005:
#                         #print(argpeak)
#                         range_int = range(int(argpeak-peakwidth/2), int(argpeak+peakwidth/2))
#                         #print(range_int)
#                         integral = integrate.trapz(filt[range_int]-np.mean(filt), dx = binsize)
#                         #print('peaksunit i,j', i,j)
#                         ccgwithpeak.append(filt)
#                         if peakstrength=="amp":
#                             peakfilt = peakCCG
#                         elif peakstrength=="integral":
#                             peakfilt = integral#integral#peakfilt#[peakfilt,filt_time[argpeak]]
#                         peakindices.append([i,j,sel_vec_cx[i], sel_vec_th[j], peakfilt,filt_time[argpeak] , ALM_FR[i], Thal_FR[j]])
#                         peak_sel.append([sel_vec_cx[i][1], sel_vec_th[j][1], peakfilt])
#                         if sel_vec_cx[i][0]=='contra' and sel_vec_th[j][0]=='contra':
#                             counter_contracontra+=1
#                             peaks_contracontra.append(peakfilt)
#                         if sel_vec_cx[i][0]=='ipsi' and sel_vec_th[j][0]=='ipsi':
#                             counter_ipsiipsi+=1
#                             peaks_ipsiipsi.append(peakfilt)

#                         if (sel_vec_cx[i][0]=='ipsi' and sel_vec_th[j][0]=='contra') or (sel_vec_cx[i][0]=='contra' and sel_vec_th[j][0]=='ipsi') :
#                             counter_mixed+=1
#                             peaks_mixed.append(peakfilt)

#                         if (sel_vec_cx[i][0]=='ipsi' and sel_vec_th[j][0]=='non-sel') or (sel_vec_cx[i][0]=='non-sel' and sel_vec_th[j][0]=='ipsi') :
#                               counter_ipsinon+=1
#                               peaks_ipsinon.append(peakfilt)

#                         if (sel_vec_cx[i][0]=='contra' and sel_vec_th[j][0]=='non-sel') or (sel_vec_cx[i][0]=='non-sel' and sel_vec_th[j][0]=='contra') :
#                               counter_contranon+=1
#                               peaks_contranon.append(peakfilt)

#                         if (sel_vec_cx[i][0]=='non-sel' and sel_vec_th[j][0]=='non-sel'):# or ([sel_vec_cx[i][0]=='non-sel' and sel_vec_th[j][0]]=='contra') :
#                               counter_nonnon+=1   
#                               peaks_nonnon.append(peakfilt)

              
              
#     contra_peaks = np.array(contra_peaks)
#     ipsi_peaks = np.array(ipsi_peaks)    
#     nonsel_peaks = np.array(nonsel_peaks)  
#     allcounters = [counter_contracontra, counter_ipsiipsi, counter_mixed, counter_contranon, counter_ipsinon, counter_nonnon]
#     allpeaks =  {'peaks_contracontra':peaks_contracontra, 'peaks_ipsiipsi':peaks_ipsiipsi, 'peaks_mixed':peaks_mixed,'peaks_contranon': peaks_contranon, 'peaks_ipsinon':peaks_ipsinon, 'peaks_nonnon':peaks_nonnon}
#     return peakindices, ccgwithpeak, allpeaks, peak_sel, allcounters
     



def getsparseformat(allspikes):
    '''
    returns binarized vectors, to be transformed to a sparse matrix with csr_matrix
    '''
    
    binsize = 0.0005  # 1/2000 , 0.5 ms
    rows_vec=[]
    col_vec=[]
    array_vec=[]
    for i_index, i in enumerate(allspikes):
        rows_vec = np.hstack([rows_vec,(allspikes[i_index]/binsize).astype(int)]) ### calculates the time locationsx  of '1' in the timevector
        col_vec =  np.hstack([col_vec, i_index*np.ones(len(allspikes[i_index]))])
        array_vec = np.hstack([array_vec, allspikes[i_index]])
    return np.array(rows_vec), np.array(col_vec), np.array(array_vec)
        
def getsparsematrix(spikes1, spikes2):
    from scipy.sparse import csr_matrix
    binsize = 0.0005    
    rows_vec, col_vec, array_vec = getsparseformat(spikes1)
    rows_vec_th, col_vec_th, array_vec_th = getsparseformat(spikes2)
    timevec = np.arange(0,max(np.max(array_vec), np.max(array_vec_th)),binsize)

    #print(len(rows_vec))
    sparse1 = csr_matrix((np.ones(len(rows_vec)), (rows_vec.astype(int), col_vec.astype(int))), shape = (len(timevec), len(spikes1)))
    sparse2 = csr_matrix((np.ones(len(rows_vec_th)), (rows_vec_th.astype(int), col_vec_th.astype(int))), shape = (len(timevec), len(spikes2)))
    return sparse1, sparse2

def cross_corr_sam(sparse, sparse_th):
    datamatrix_cx = sparse
    datamatrix_th = sparse_th

    N_cx = np.shape(datamatrix_cx)[1]
    N_th = np.shape(datamatrix_th)[1]
    T = np.shape(datamatrix_cx)[0]

    dt = 0.0005  #0.5ms
    maxlag = 100e-3
    Nlag = int(maxlag/dt)
    #lags = 2*Nlag+1#dt*range(-Nlag, Nlag)
    Ntotlag = 2*Nlag+1# len(lags)
    ivalid_start = Nlag+1###??
    ivalid_end = T-Nlag ##??
    Mvalid = datamatrix_cx[ivalid_start:ivalid_end, :]
    #print('Mvalid', np.shape(Mvalid), N_cx, N_th)
    ccgs = np.zeros((Ntotlag, N_cx, N_th))
    ALM_FR = (np.array(np.mean(datamatrix_cx, axis = 0)))[0]/dt
    Thal_FR = (np.array(np.mean(datamatrix_th, axis = 0)))[0]/dt      
   
    #print('ccgshape', np.shape(ccgs))
    #ccgs = []
    import time
    start = time.time()
    for ilag in range(0, Ntotlag):
        print(ilag)
        #print(type(ilag))
        shift = ilag-Nlag -1 ####??
        ishift_start = ivalid_start + shift
        ishift_end = ivalid_end + shift
        Mshift = datamatrix_th[ishift_start:ishift_end, :]
        #print('Mshift', np.shape(Mshift))
        #print(np.shape( Mvalid.T*(Mshift) ))
        #ccgs.append(Mvalid.T*(Mshift)) #### sparse?
        product = (Mvalid.T*(Mshift))      
        #print (type(product))
        ccgs[ilag,:, :]  = product.toarray()
        #return ccgs
    filt_time = dt*np.arange(-Nlag-1, Nlag)
    duration = (time.time()-start )
    print('duration in sec', duration)
    print('duration in min', duration/60)
    return ccgs, filt_time, ALM_FR, Thal_FR





def step(t, t_shift):
      return (np.sign(t-t_shift)+1)/2
def pulse(t, t_start, t_duration):
      return step(t,t_start) - step(t,t_start+t_duration)

def rate_fromspikes(times, trials, delta, unit_index, allunits_new):
    '''
    get time-dependent rate by windowing a spike train
    
    input:
        
    delta: window width
    
    unit_index = units in spike train list allunits_new
    allunits_new = list of spike times as a function of units and trials
        
    output
    
    rate: rate as a function of time

    '''
    def step(t, t_shift):
          return (np.sign(t-t_shift)+1)/2
    def pulse(t, t_start, t_duration):
          return step(t,t_start) - step(t,t_start+t_duration)
    window_vec = 0
    window_vec_alltrials = 0
    for trial_index in trials:#range(0,len(allunits_new[0])):
        #print(trial_index)
        spiketimes = allunits_new[unit_index][trial_index]
        for spiketime in spiketimes:
        #print(spiketime)
            window = 1/delta*pulse(times,-delta/2+spiketime,delta)
            #plt.plot(times,window)
            window_vec+=window
        window_vec_alltrials+=window_vec
        window_vec = 0
    rate = window_vec_alltrials/len(trials)
    if type(rate) ==float:
        rate=np.zeros(len(times))
    meanrate = np.mean(rate)
    #print(meanrate)
    return rate


def spikestoCCG(unitsALM, unitsThal, alpha):

    papersamplingfreq = 30e3 #### 30khz from paper
    samplingfreq = 4e3# 
    binsize = 0.01#1/samplingfreq
    timevec, spikevec_ALM = spiketobinary(binsize, unitsALM)
    print('ALM spike to binary done')
    timevec, spikevec_Thal = spiketobinary(binsize, unitsThal)
    print('Thal spike to binary done')
    interval = (timevec>-0.5)*(timevec<0.5)#timevec>-3 #(timevec>-1.2)*(timevec<0)
    spikemeansALM = np.mean(spikevec_ALM[:,:,interval], axis=2)/binsize
    spikemeansThal = np.mean(spikevec_Thal[:,:,interval], axis=2)/binsize
    ALM_FR = np.mean(spikemeansALM, axis=0)#/binsize
    Thal_FR = np.mean(spikemeansThal, axis=0)
    return spikevec_ALM,spikevec_Thal,ALM_FR,Thal_FR
    
def spikestosel (unitsALM, unitsThal, epoch, stats,hemi, alpha):
    papersamplingfreq = 30e3 #### 30khz from paper
    samplingfreq = 4e3# 
    binsize = 0.01#1/samplingfreq
    hit, miss, ignore, early, alltrials, left_hit, right_hit = stats
    left,right = get_left_righttrials (alltrials, left_hit, right_hit)
    timevec, spikevec_ALM = spiketobinary(binsize, unitsALM)
    print('ALM spike to binary done')
    timevec, spikevec_Thal = spiketobinary(binsize, unitsThal)
    print('Thal spike to binary done')
    
    sel_vec_cx, sel_vec_th, ALM_FR, Thal_FR = get_sel(binsize, timevec, spikevec_ALM, spikevec_Thal, stats, hemi,alpha)
    return timevec, spikevec_ALM, spikevec_Thal, sel_vec_cx, sel_vec_th, ALM_FR, Thal_FR 


def spiketobinary(binsize, spike_trials):
    '''
    inputs: 
    spike_trials = list of spiketimes, units, and trials: shape(units,trials, time)
    units = list of units
    
    outputs:
    binsize (self explanatory)
    timevec = time vector of duration trialtime with binsize
    spike_vec
    '''
    
    spike_trials = np.array(spike_trials, dtype=object)
    trialtime = 6 #### duration of trial
    #lenvec = int(trialtime/binsize)
    timevec = np.arange(-trialtime/2,trialtime/2,binsize)
    timevec_ = np.arange(-trialtime/2,trialtime/2+binsize,binsize)
    Ntrials = np.shape(spike_trials)[1]
    numunits = np.shape(spike_trials)[0]
    spike_vec = np.zeros((numunits, Ntrials,len(timevec)))
    #spike_vec2 = np.zeros((numunits, Ntrials,len(timevec)))

    for i_unit in range(0,numunits):
        #print(i_unit)
        for trialindex in range(0,Ntrials):
            spike_vec[i_unit,trialindex, :], bins = np.histogram(spike_trials[i_unit][trialindex], bins = timevec_)
            #spike_vec2[i_unit,trialindex, :] = (spike_trials[i_unit][trialindex]/binsize).astype(int)
    return timevec, spike_vec


def get_sel (binsize, timevec, spikevec_cx, spikevec_Thal, stats, hemi, alpha):
    hit, miss, ignore, early, alltrials, left_hit, right_hit = stats
    left,right = get_left_righttrials (alltrials, left_hit, right_hit)
    interval = (timevec>-0.5)*(timevec<0.5)#timevec>-3 #(timevec>-1.2)*(timevec<0)
    spikemeansALM = np.mean(spikevec_cx[:,:,interval], axis=2)/binsize
    spikemeansThal = np.mean(spikevec_Thal[:,:,interval], axis=2)/binsize
    ALM_FR = np.mean(spikemeansALM, axis=1)#/binsize
    Thal_FR = np.mean(spikemeansThal, axis=1)#/binsize
    if hemi=='left':
        contra = right
        ipsi = left
    else:
        contra = left
        ipsi = right
    sel_vec_cx = []
    sel_vec_th = []
    #print(np.shape(spikemeansALM))
    for i in range(0, np.shape(spikemeansALM)[0]):
        freq_left = spikemeansALM[i,left]#np.sum(spikevec_cx[i,left,:], axis =1)#spikemeansALM[i,left]
        freq_right = spikemeansALM[i,right]#np.sum(spikevec_cx[i,right,:], axis =1)#spikemeansALM[i,right]
        t_stat, pval = sc_stats.ttest_ind(freq_left, freq_right, equal_var=True)
        diff_sel_cx = (np.mean(spikemeansALM[i,contra])-np.mean(spikemeansALM[i,ipsi]))/ALM_FR[i]

        if pval<alpha:
            if diff_sel_cx<0:
                sel_vec_cx.append(['ipsi', diff_sel_cx])
            else:
                sel_vec_cx.append(['contra', diff_sel_cx])
        else:
            sel_vec_cx.append(['non-sel', diff_sel_cx])
    for i in range(0, np.shape(spikemeansThal)[0]):
        freq_left = spikemeansThal[i,left]
        freq_right = spikemeansThal[i,right]
        t_stat, pval = sc_stats.ttest_ind(freq_left, freq_right, equal_var=True)
        diff_sel_th = (np.mean(spikemeansThal[i,contra])-np.mean(spikemeansThal[i,ipsi]))/Thal_FR[i]

        if pval<alpha:
            if diff_sel_th<0:
                sel_vec_th.append(['ipsi', diff_sel_th])
            else:
                sel_vec_th.append(['contra',diff_sel_th])
        else:
            sel_vec_th.append(['non-sel',diff_sel_th])
    contra_cx = np.where(np.array(sel_vec_cx)=='contra')[0]
    ipsi_cx = np.where(np.array(sel_vec_cx)=='ipsi')[0]
    non_sel_cx = np.where(np.array(sel_vec_cx)=='non-sel')[0]
    frac_selective_cx = (len(contra_cx) +  len(ipsi_cx))/len(sel_vec_cx)
    contra_th = np.where(np.array(sel_vec_th)=='contra')[0]
    ipsi_th = np.where(np.array(sel_vec_th)=='ipsi')[0]
    non_sel_th = np.where(np.array(sel_vec_th)=='non-sel')[0]
    frac_selective_th = (len(contra_th) +  len(ipsi_th))/len(sel_vec_th)
    print('contra_cx', len(contra_cx))
    print('ipsi_cx', len(ipsi_cx))
    print('fracsel_cx',frac_selective_cx)
    print('contra_th', len(contra_th))
    print('ipsi_th', len(ipsi_th))
    print('fracsel_th',frac_selective_th)
    return sel_vec_cx, sel_vec_th, ALM_FR, Thal_FR

def get_left_righttrials (alltrials, left_hit, right_hit):
    left_ = []
    right_= []
    counter = 0
    for i, trial in enumerate(alltrials):
        if trial in left_hit:
            left_.append(i-counter)
        elif trial in right_hit:
            right_.append(i-counter)
        else:
            counter+=1
    return left_, right_

def CCG_norm(ccg, filt_time, ALM_FR, Thal_FR, norm_type):
    '''
    

    Parameters
    ----------
    ccg : crosscorrelogram time series
    peakindices : list of information related to each ccg peak, including time of peak
    norm_type : 
        pre: normalized by presynaptic neuron, defined by latency
        post: normalized by postsynaptic neuron, defined by latency
        both: self-explanatory
        none: self-explanatory

    Returns
    -------
    None.

    '''
    ccgshape = np.shape(ccg)
    ccgs_norm = np.zeros((ccgshape[0], ccgshape[1], ccgshape[2]))
    filt = ccg-np.mean(ccg,axis =0)
    print('ALMFRshape', np.shape(ALM_FR))
    print('ALMCCG', ccgshape[1])
    print('ThalFRshape', np.shape(Thal_FR))
    print('ThalCCG', ccgshape[2])


    
    for i in range(0, ccgshape[1]):
        for j in range(0,ccgshape[2]):
            argpeak = np.argmax(filt[:,i,j])
            time_peak = filt_time[argpeak]
            if norm_type=="pre":
                if time_peak<0:#thalamus is shifted with tau, tau negative means thalamus is postsynaptic, cortex is presynaptic
                    ccgs_norm[:,i,j] = filt[:,i,j]/ALM_FR[i]
                elif time_peak>0:
                    ccgs_norm[:,i,j] = filt[:,i,j]/Thal_FR[j]
            elif norm_type=="both":
                 ccgs_norm[:,i,j] = filt[:,i,j]/(ALM_FR[i]*Thal_FR[j])**0.5
            elif norm_type =="none":
                 ccgs_norm[:,i,j] = filt[:,i,j]
    return ccgs_norm

def plot_sel(allunits, timevec, sel_vec, stats):
    """
    plots a subset of selective neurons
    
    inputs:
        
    outputs: 
        
    
    """
    hit, miss, ignore, early, alltrials, left_hit, right_hit = stats
    left,right = get_left_righttrials (alltrials, left_hit, right_hit)
    contra_index = np.where(np.array(sel_vec)=='contra')[0]
    ipsi_index = np.where(np.array(sel_vec)=='ipsi')[0]
    non_sel_index = np.where(np.array(sel_vec)=='non-sel')[0]
    Nplots = 20

    fig, ax = plt.subplots(Nplots,3, figsize = (8,30))
    for index, unit in enumerate(contra_index[0:Nplots]):
        ax[index,0].plot(timevec, rate_fromspikes(timevec, right, 0.12, unit, allunits), 'blue')
        ax[index,0].plot(timevec, rate_fromspikes(timevec, left, 0.12, unit, allunits), 'red')
        ax[index,0].set_ylim(0,30)
        ax[index,0].set_xlim(-0.5,0.5)

        ax[0,0].set_title('contra')

        #print('contra', unit)
    for index, unit in enumerate(ipsi_index[0:Nplots]):
        ax[index,1].plot(timevec, rate_fromspikes(timevec, right, 0.12, unit, allunits), 'blue')
        ax[index,1].plot(timevec, rate_fromspikes(timevec, left, 0.12, unit, allunits), 'red')
        ax[index,1].set_ylim(0,50)
        ax[index,1].set_xlim(-0.5,0.5)

        ax[0,1].set_title('ipsi')


        #print('ipsi', unit)
    for index, unit in enumerate(non_sel_index[0:Nplots]):
        ax[index,2].plot(timevec, rate_fromspikes(timevec, right, 0.12, unit, allunits), 'blue')
        ax[index,2].plot(timevec, rate_fromspikes(timevec, left, 0.12, unit, allunits), 'red')
        #print('non-sel', unit)
        ax[index,2].set_ylim(0,30)
        ax[index,2].set_xlim(-0.5,0.5)
        print(unit)

        ax[0,2].set_title('non-sel')

