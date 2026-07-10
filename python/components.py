"""
RF components
- Code generator (WH, Gold, and ZC) 
- Spreader
- Despreader
- Combiner
- Filters
- Matched Filter 
"""

from enum import Enum
from math import ceil, log2

import commpy.filters as filter  # type:ignore[import-untyped]
import komm as km  # type:ignore[import-untyped]
import matplotlib.pyplot as plt
import numpy as np
from commpy.sequences import pnsequence  # type:ignore[import-untyped]
from scipy.signal import max_len_seq, resample_poly

from models import DecorrOutput, Specan, SpreadOutput


class CDMx:
    def __init__(self, conf):
        self.mf = Filter(conf)
        self.conf = conf 

    def generate_codes(self):
        if self.conf.codefam == "WH":
            #construct n-by-n Hadamard matrix using Sylvesters construction
            #n: must be a power of 2 and int 
            codes = []
            for i in self.conf.codeids:
                WHcode = km.WalshHadamardSequence(
                    length=self.conf.lc,
                    index=i,
                    ordering='sequency')

                codes.append(WHcode.polar_sequence)
        else: 
            raise Exception("Unrecognised code family! Choose from [WH, ]")

        return codes 

    def resample_to_spschip(self,targetfs, oldfs, signal):
        if oldfs < targetfs:
            ups = int(round(targetfs/oldfs))
            ds = 1 
        elif oldfs > targetfs:
            ups = 1 
            ds = int(round(oldfs/targetfs))
        else:
            ups = 1
            ds = 1
        
        newfs = oldfs * ups / ds #rarely a perfect match
        rs = resample_poly(signal, ups, ds)

        return rs, newfs

    def spreader(self, signals, codes):
        targetfs = self.conf.sps_chip * self.conf.rc
        oldfs = self.conf.fs
        ss = []
        exp_codes = []
        for i in range(self.conf.no_elem):
            resampled, fs = self.resample_to_spschip(targetfs, oldfs, signals[i])

            code_sequence = np.repeat(codes[i], self.conf.sps_chip)
            tile_repeats = int(np.ceil(len(resampled) / len(code_sequence)))
            expanded_code = np.tile(code_sequence, tile_repeats)[:len(resampled)]
            exp_codes.append(expanded_code)

            #FORTEST
            spread_signal = resampled*expanded_code 
            ss.append(spread_signal)
            #ss.append(resampled)

        time = np.arange(len(ss[0])) * (1/fs)
        return SpreadOutput(ss, time, exp_codes, fs)
 

    def despreader(self, sumsignal, codes, fs):
        #Despread signal by multiplying with code block
        dss = [] #despread
        decorr = [] #decorrelated with rrc filter

        for i in range(self.conf.no_elem):
            despread_signal = sumsignal*codes[i]
            dss.append(despread_signal)
            
            #FORTEST:
            #despread_signal = sumsignal[i]

            signal, symbols = self.mf.rrcfilter(
                despread_signal,
                fs,
                mode="rxmode"
                )
            
            decorr.append(signal)
        #removes delay 
        time = np.arange(len(decorr[0]))*(1/fs)
        return DecorrOutput(decorr, time)



class Filter:
    def __init__(self, conf):
        self.alpha = conf.alpha
        self.rb = conf.rb
        self.span = conf.span 

    def rrcfilter(self, signal, fs, mode="txmode"):
        sps_symbol = int(round(fs / self.rb)) #also in config 
        ts = 1 / self.rb 
        ntaps = self.span*sps_symbol + 1  #must be odd, total length of the filter 

        if mode == "txmode":
            symbols = np.zeros(len(signal)*sps_symbol)
            symbols[::sps_symbol] = signal
            samples = symbols
        else:
            samples = signal

        #apply filter
        dt_filter, h_rrc = filter.rrcosfilter(ntaps, self.alpha, ts, fs)
        h_rrc /= np.sqrt(np.sum(h_rrc**2))

        filtered = np.convolve(samples, h_rrc, mode='full')
        h_delay = (ntaps-1)//2
        filtered_nodelay = filtered[h_delay:-h_delay]

        if mode == "rxmode":
            symbols = filtered_nodelay[::sps_symbol]

        return filtered_nodelay, symbols  


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
        return Specan(f, Pxx_dB, bw)


