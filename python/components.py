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
from scipy.signal import max_len_seq, resample_poly, welch, butter, filtfilt
import matplotlib.pyplot as plt 

from models import DecorrOutput, Specan, SpreadOutput


class DiscreteSP:
    #for discrete mixing you need to have equivalent sampling frequency 

    def resample_signal(self,targetfs, oldfs, signal):
        if oldfs < targetfs:
            ups = int(round(targetfs/oldfs))
            ds = 1 
        elif oldfs > targetfs:
            ups = 1 
            ds = int(round(oldfs/targetfs))
        else:
            ups = 1
            ds = 1
        
        newfs = oldfs * ups / ds #for imperfect match
        rs = resample_poly(signal, ups, ds)

        return rs, newfs

class CDMx:
    def __init__(self, conf):
        self.mf = Filter(conf)
        self.dsp = DiscreteSP()
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
        elif self.conf.codefam == "Gold": 
            #for length N=31
            #use preferred polynomial from Matlab
            degree = 5 #2^5-1=31 
            seq_a, _ = max_len_seq(degree, taps=[2]) 
            seq_b, _ = max_len_seq(degree, taps=[4,3,2])

            #degree = 3 #2^3 - 1 = 7 
            #seq_a, _ = max_len_seq(degree, taps=[1])
            #seq_b, _ = max_len_seq(degree, taps=[2])

            gold_codes = []
            for shift in range(len(seq_a)):
                code = np.bitwise_xor(
                    seq_a,
                    np.roll(seq_b, shift)
                )
                gold_codes.append(code)
            codes = [2*gold_codes[idx]-1 for idx in self.conf.codeids]
        else: 
            raise Exception("Unrecognised code family! Choose from [WH, Gold]")

        return codes 


    def spreader(self, signals, codes):
        targetfs = self.conf.sps_chip * self.conf.rc
        oldfs = self.conf.fs
        ss = []
        exp_codes = []
        for i in range(self.conf.no_elem):
            resampled, fs = self.dsp.resample_signal(targetfs, oldfs, signals[i])
            
            code_sequence = np.repeat(codes[i], self.conf.sps_chip)
            tile_repeats = int(np.ceil(len(resampled) / len(code_sequence)))
            expanded_code = np.tile(code_sequence, tile_repeats)[:len(resampled)]
            exp_codes.append(expanded_code)
            """
            #PLOT FOR RESAMPLED SIGNALxCODES
            plt.figure()
            plt.plot(expanded_code, '--o', label=f"Code: {codes[i]}")
            plt.plot(resampled, '-o', label="Resampled Rx Signal")
            plt.legend(loc='upper right')
            plt.title("Discrete Signal Mixing: Code x Signal", fontsize=14)
            plt.ylabel("Amplitude")
            plt.xlabel("Sample index")
            plt.grid()
            plt.show()
            """
            #FORTEST
            spread_signal = resampled*expanded_code 
            ss.append(spread_signal)
            #ss.append(resampled)

        #check = np.dot(exp_codes[0], exp_codes[1])

        time = np.arange(len(ss[0])) * (1/fs)
        """
        #PLOT FOR CODE SERIES ONLY
        for i in range(len(exp_codes)):
            print("code=", codes[i])
            plt.rcParams['font.size'] = 15
            plt.figure()
            plt.title(f"Code sequence index={i+1}")
            plt.plot(time, exp_codes[i], '-g', label="WH Code Sequence")
            plt.legend(loc='upper right')
            plt.grid()
            plt.xlabel("Time (seconds)")
            plt.ylabel("Amplitude")
            plt.show()
        """
        return SpreadOutput(ss, time, exp_codes, fs)

    def recover(self, sumsignal, codes, fs):
        if not self.conf.CALIBRATE:
            dss = self.despread(sumsignal, codes)
        else:
            dss = sumsignal 

        decorrelated = self.decorrelate(dss, fs)

        #removes delay and fs unchanged 
        time = np.arange(len(decorrelated[0]))*(1/fs)
        return DecorrOutput(decorrelated, time, fs)

    def despread(self, sumsignal, codes):
        #Despread signal by multiplying with code block
        dss = [] #despread
        for i in range(self.conf.no_elem):
            despread_signal = sumsignal*codes[i]
            dss.append(despread_signal)
        return dss

    def decorrelate(self, despread_signal, fs):
        decorr = [] 
        for i in range(self.conf.no_elem):
            decorrelated_signal, symbols = self.mf.rrcfilter(
                despread_signal[i],
                fs,
                mode="rxmode"
                )
            decorr.append(decorrelated_signal)
        return decorr


class Filter:
    def __init__(self, conf):
        self.alpha = conf.alpha
        self.rb = conf.rb
        self.span = conf.span 
        self.fs = conf.fs
        self.flo = conf.flo
        self.sps_symbol = conf.sps_symbol 
        self.fs_carrier = conf.CARRIER

    def rrcfilter(self, signal, fs, mode="txmode"):
        sps_symbol  = int(round(fs / self.rb)) #changes from config due to resampling
        ts = 1 / self.rb 
        ntaps = self.span*sps_symbol + 1  #must be odd, total length of the filter 
        
        #upsampling
        if mode == "txmode":
            prepad = sps_symbol
            symbols = np.zeros(
                prepad + len(signal)*sps_symbol,
                dtype=signal.dtype)
            symbols[prepad::sps_symbol] = signal
            samples = symbols
        else:
            samples = signal
        
        #apply filter
        dt_filter, h_rrc = filter.rrcosfilter(ntaps, self.alpha, ts, fs)
        h_rrc /= np.sqrt(np.sum(h_rrc**2))
        filtered = np.convolve(samples, h_rrc, mode='full')
        h_delay = (ntaps-1)//2
        
        filtered_delay = filtered[h_delay:-h_delay]

        if mode == "rxmode":
            symbols = filtered_delay[::sps_symbol]
            filtered_delay = filtered[h_delay+1:-h_delay+1]
            #combined gain is not affected by this since its from adding multiple signal paths
            #not related to chip gain so keep it 
            #filtered_nodelay /= np.max(np.abs(filtered_nodelay))

        return filtered_delay, [0]

    def butterworth(self, cutoff, fs, signal, filter_type="low"):
        #fs is sampling frequency! 
        nyquist = fs/2
        #cutoff = 1.5*self.rb
        b, a = butter(
            N=4, 
            Wn = cutoff/nyquist, 
            btype=filter_type
        )
        return filtfilt(b,a,signal)


class Beamformer:    
    def align_combine_signals(self,psi,sig):
        #output aligned signals and combined

        beamformed = sig[0].copy()
        #align rest of signals 
        for i in range(1,len(sig)):
            #applying the phase in opposite sign  - ONLY KEEPS REAL PART FOR BPSK FOR NOW 
            beamformed += np.real(sig[i]*np.exp(+1j*psi[i]))
    
        return beamformed


class Mixer:
    def __init__(self,conf):
        self.conf = conf 
        self.dsp = DiscreteSP()

    def superhet(self, signal):
        #match sampling rates for discrete time mixing 
        
        signal_resampled, fs_resampled = self.dsp.resample_signal(
            fs_mixer,
            self.conf.fs,
            signal)

        if fs_resampled != fs_mixer:
            raise ValueError("Incorrect sampling rates at Mixer and Signal because resampler only takes integers")
        
        lo_time = np.arange(len(signal_resampled))*(1/fs_mixer)
        lo_signal = np.cos(2*np.pi*self.conf.flo*lo_time)

        #mix signal by multiplication 
        mixer_output = signal_resampled * lo_signal 

        return fs_mixer, lo_time, mixer_output 

    def quadrature(self,signal, signal_fs):
        #mixer sampling frequency is 3x the carrier frequency 
        fs_mixer = 3*self.conf.flo
        #match sampling rates for discrete time mixing 
        signal_resampled, fs_resampled = self.dsp.resample_signal(
            fs_mixer,
            signal_fs,
            signal)

        if fs_resampled != fs_mixer:
            raise ValueError("Incorrect sampling rates at Mixer and Signal because resampler only takes integers")
        
        #local oscillators for I and Q component
        lo_time = np.arange(len(signal_resampled))*(1/fs_mixer)
        lo_signal = np.exp(1j*2*np.pi*self.conf.flo*lo_time) 

        #mix signal by multiplication 
        mixer_output = signal_resampled * lo_signal

        return fs_mixer, lo_time, mixer_output 

class SpecAn: 
    def calculate_psd(self ,fs, signal,psdtype="normal"):
        f, Pxx = welch(
            signal,
            fs=fs,
            window='hann',
            nperseg=4096,
            scaling='density',
            return_onesided=False,
            noverlap=0
        )

        f = np.fft.fftshift(f)
        Pxx = np.fft.fftshift(Pxx)

        Pxx_dB = 10*np.log10(Pxx + 1e-12) 
        Pxx_dB = Pxx_dB - np.max(Pxx_dB) #normalise

        if psdtype =="normal":
            df = np.abs(f[1]-f[0])
            power = Pxx* df
            power = power / np.sum(power)

            cdf = np.cumsum(power)

            f_low  = f[np.searchsorted(cdf, 0.00005)]
            f_high = f[np.searchsorted(cdf, 0.99995)]

            bw = f_high - f_low
        else: 
            peak = np.max(Pxx_dB)

            mask = Pxx_dB > (peak + psdtype)

            idx = np.where(mask)[0]

            f_low = f[idx[0]]
            f_high = f[idx[-1]]

            bw = f_high - f_low
        return Specan(f, Pxx_dB, bw)


