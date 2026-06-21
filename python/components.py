"""
RF components
- Code generator (WH, Gold, and ZC) 
- Spreader
- Despreader
- Combiner
- Filters
- Matched Filter 
"""

from scipy.signal import max_len_seq, welch
import numpy as np  
from enum import Enum 
import commpy.filters as filter
from commpy.sequences import pnsequence

import komm as km
import matplotlib.pyplot as plt 
from math import log2

class CDMx:
    def generate_code(self, no_arrays, Lc, codeids, type):
        #L: code length used to generate code sequences 
        if type == "WH":
            #construct n-by-n Hadamard matrix using Sylvesters construction
            #n: must be a power of 2 and int 
            codes = []
            for i in range(no_arrays):
                WHcode = km.WalshHadamardSequence(
                    length=Lc,
                    index=codeids[i],
                    ordering='sequency')

                codes.append(WHcode.polar_sequence)
        else: 
            raise Exception("Unrecognised code family! Choose from [WH, ]")

        return codes 
        

    def spreader(self, code, Lc, signal):
        """
        direct time multiplication
        """
        ss = [] #spread symbols    
        #repeat the code to equal length of the signal that reflects the spreading facto]

        for i in range(len(signal)):
            ss.append(signal[i]*code[i%Lc])
        return ss     

    def despreader(self, Lc, Ts, code, spreadsum):   
        #Depends on Decorrelator being run first. ASK ALIX
        despread_symbols = [] 
        for i in range(len(spreadsum)):
            #restarts the code indexing at the end of sequence 
            despread_symbols.append((spreadsum[i]*code[i%Lc]/ Lc))

        #matched filter / correlator too! 
        sps = Lc #samples per symbol sample 
        Fs = sps/Ts #sample per second, take Lc samples per second 
        alpha = 0.5 #does this need to match?
        span = 16 #consider 8 symbols, i believe this is normal number 
        Ntaps = span*sps

        dt_filter, h_rrc = filter.rrcosfilter(Ntaps, alpha, Ts, Fs)
        h_delay = (Ntaps-1)/2
        conv_symbols = np.convolve(despread_symbols, h_rrc, mode='full')
        h_delay = int(np.round(h_delay))

        Nsamples = len(spreadsum) // Lc   # number of samples representing symbols that was spread
        idxs = h_delay  + np.arange(Nsamples)*sps
        samples = conv_symbols[idxs]

        return [samples, idxs]


class MatchedFilter:
    def txrrcfilter(self, symbols, Ts, ups):
        """
        rrc filter for transmit
        independent of Lc / Rx settings as its suppose to simulate real rxd signals
        what it does:
        - upsamples
        - applies filter 
        - is called by mysignal.py 
        """
        Fs = ups/Ts #Hz; sampling rate, sample per second 
        alpha = 0.5 #roll off factor 
        span =  10 #dictates how far a symbols influence spreads across time
        Ntaps = span*ups + 1  #must be odd, total length of the filter 

        upsampled = np.zeros(len(symbols)*ups) 
        upsampled[::ups] = symbols
  
        #apply filter
        dt_filter, h_rrc = filter.rrcosfilter(Ntaps, alpha, Ts, Fs)
        txsignal = np.convolve(upsampled, h_rrc, mode='full')
        group_delay = int((Ntaps-1)/2)

        return txsignal[group_delay:-group_delay], upsampled


class SpecAn: 
    def calculate_psd(self,fs, signal):
        f,Pxx = welch(signal, fs, return_onesided=False)
        idx = np.argsort(f)
        f = f[idx]
        Pxx = Pxx[idx]
        df = f[1] - f[0]
        cumulative = np.cumsum(Pxx) * df
        total_power = cumulative[-1]
        low_idx = np.where(cumulative >= 0.005 * total_power)[0][0]
        high_idx = np.where(cumulative >= 0.995 * total_power)[0][0]
        bw = f[high_idx] - f[low_idx]
        return f, Pxx, bw
