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
    def __init__(self, conf):
        self.mf = Filter()
        self.conf = conf 


    def generate_code(self, time):
        self.time = time
        if self.conf.codefam == "WH":
            #construct n-by-n Hadamard matrix using Sylvesters construction
            #n: must be a power of 2 and int 
            codes = []
            for i in range(self.conf.no_elem):
                WHcode = km.WalshHadamardSequence(
                    length=self.conf.Lc,
                    index=self.conf.codeids[i],
                    ordering='sequency')

                codes.append(WHcode.polar_sequence)
        else: 
            raise Exception("Unrecognised code family! Choose from [WH, ]")

        return codes 

    def spreader(self, code, asignal):
        """
        direct time multiplication
        """
        ss = [] 
        for i in range(len(asignal)):
            
            val = asignal[i]*code[i%self.conf.Lc]
            ss.append(val)
    
        return ss     

    def despreader(self, code, spreadsum):   
        #Despread signal by multiplying with code block
        despread = spreadsum * code 

        self.recovered, self.rxsymbols = self.mf.rrcfilter(
            despread,
            self.conf.Rb,
            self.conf.Fs,
            self.conf.sps_symbol, 
            self.conf.alpha,
            mode="rxmode"
            )
    
        return self.recovered/np.max(self.recovered)

class Filter:
    def rrcfilter(self, symbols, Rb, Fs, sps_symbol, alpha, span=8, mode="txmode"):
        """
        rrc filter for transmit
        independent of Lc / Rx settings as its suppose to simulate real rxd signals
        what it does:
        - upsamples
        - applies filter 
        - is called by mysignal.py 
        """

        Ts = 1/Rb
        Ntaps = span*sps_symbol + 1  #must be odd, total length of the filter 

        if mode == "txmode":
            upsampled = np.zeros(len(symbols)*sps_symbol)
            upsampled[::sps_symbol] = symbols
        else:
            upsampled = symbols 


        #apply filter
        dt_filter, h_rrc = filter.rrcosfilter(Ntaps, alpha, Ts, Fs)
        txsignal = np.convolve(upsampled, h_rrc, mode='full')
        h_delay = (Ntaps-1)//2

        return txsignal[h_delay:-h_delay], upsampled


class SpecAn: 
    def calculate_psd(self ,fs, signal,threshold=-40):
        fft_data = np.fft.fft(signal)
        fft_data = np.fft.fftshift(fft_data)

        power = np.abs(fft_data)**2
        Pxx = power / np.sum(power) 
        f = np.fft.fftshift(np.fft.fftfreq(len(signal), 1/fs))


        Pxx_dB = 10*np.log10(Pxx + 1e-12)

        peak = np.max(Pxx_dB)
        mask = Pxx_dB > (peak + threshold)

        idx = np.where(mask)[0]

        f_low = f[idx[0]]
        f_high = f[idx[-1]]

        bw = f_high - f_low
        return f, Pxx_dB, bw


