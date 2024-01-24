# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 16:25:03 2017

@author: Xiaoxuan Jia
"""
###

import numpy as np
from scipy import stats
import scipy
#https://github.com/jiaxx/jitter
from jitter import jitter

def xcorrfft(a,b,NFFT):
    CCG = np.fft.fftshift(np.fft.ifft(np.multiply(np.fft.fft(a,NFFT), np.conj(np.fft.fft(b,NFFT)))))
    return CCG

def nextpow2(n):
    """get the next power of 2 that's greater than n"""
    m_f = np.log2(n)
    m_i = np.ceil(m_f)
    return 2**m_i

def get_ccgjitter(spikes, FR, jitterwindow=25):
    # spikes: neuron*ori*trial*time
    assert np.shape(spikes)[0]==len(FR)

    n_unit=np.shape(spikes)[0]
    n_t = np.shape(spikes)[3]
    # triangle function
    t = np.arange(-(n_t-1),(n_t-1))
    theta = n_t-np.abs(t)
    del t
    NFFT = int(nextpow2(2*n_t))
    target = np.array([int(i) for i in NFFT/2+np.arange((-n_t+2),n_t)])

    ccgjitter = []
    pair=0
    temp1 = np.rollaxis(np.rollaxis(spikes[0],2,0), 2,1)
    temp2 = np.rollaxis(np.rollaxis(spikes[1],2,0), 2,1)
    ttemp1 = jitter(temp1,jitterwindow);  
    ttemp2 = jitter(temp2,jitterwindow);
    tempjitter = xcorrfft(np.rollaxis(np.rollaxis(ttemp1,2,0), 2,1),np.rollaxis(np.rollaxis(ttemp2,2,0), 2,1),NFFT);  
    tempjitter = np.squeeze(np.nanmean(tempjitter[:,:,target],axis=1))
    ccgjitter.append((tempccg - tempjitter).T/np.multiply(np.tile(np.sqrt(FR[0]*FR[1]), (len(target), 1)), 
        np.tile(theta.T.reshape(len(theta),1),(1,len(FR1)))))

    ccgjitter = np.array(ccgjitter)
    return ccgjitter

