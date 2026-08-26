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
from math import ceil, log2, gcd

import commpy.filters as filter  # type:ignore[import-untyped]
import komm as km  # type:ignore[import-untyped]
import matplotlib.pyplot as plt
import numpy as np
from commpy.sequences import pnsequence  # type:ignore[import-untyped]
from scipy.signal import periodogram, max_len_seq, resample_poly, welch, butter, filtfilt
import matplotlib.pyplot as plt 

from models import Signals_Time, ResampledSignals, Specan, SpreadSignals


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
        rsignal = resample_poly(signal, ups, ds)

        return rsignal, newfs

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


    def spreader(self, signals, fs_signal, codes):
        fs_code = self.conf.sps_chip * self.conf.lc * fs_signal
        #fs_code = self.conf.sps_chip * self.conf.rc #atleast double the chip frequency
        fs_signal = fs_signal
        ss = []
        exp_codes = []
        for i in range(self.conf.no_elem):
            #resample signal to match sample frequency of chip 
            resampled, fs_resampled = self.dsp.resample_signal(
                targetfs=fs_code,
                oldfs=fs_signal,
                signal=signals[i]
            )
            if (fs_resampled > 1.1*fs_code) | (fs_resampled<0.1*fs_code):
                raise ValueError("Signal sampling frequency not equal to code sampling frequency!")
            #extend code sequence to the same length as the resampled signal 
            code_sequence = np.repeat(codes[i], self.conf.sps_chip)
            tile_repeats = int(np.ceil(len(resampled) / len(code_sequence)))
            expanded_code = np.tile(code_sequence, tile_repeats)[:len(resampled)]
            exp_codes.append(expanded_code)
            
            """
            #PLOT FOR RESAMPLED SIGNALxCODES BEFORE SPREADING
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
            
            #spreading operation
            spread_signal = resampled*expanded_code 
            ss.append(resampled)
        
        time = np.arange(len(ss[0])) * (1/fs_resampled)

        if max(time)>self.conf.simtime*1.2:
            print(max(time))
            raise ValueError("Something went wrong with upsampling of the time series! Exceeds sim time")
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
        return SpreadSignals(ss, time, exp_codes, fs_resampled)

    def recover(self, sumsignal, codes, fs):
        if not self.conf.CALIBRATE:
            dss = self.despread(sumsignal, codes)
        else:
            dss = sumsignal 

        decorrelated = self.decorrelate(dss, fs)

        #removes delay and fs unchanged 
        new_time = np.arange(len(decorrelated[0]))*(1/fs)
        return Signals_Time(new_time, decorrelated)

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
        self.sps_symbol = conf.sps_symbol 

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

        return filtered_delay, symbols

    def butterworth(self, cutoff, fs, signals, filter_type="low"):
        #fs is sampling frequency! 
        nyquist = fs/2
        #cutoff = 1.5*self.rb
        #3dB point at cutoff frequency 
        a,b = butter(
            N=4, 
            Wn = cutoff/nyquist, 
            btype=filter_type
        )
        #filter can be applied to an array type which signal is 
        filteredsignals = filtfilt(a,b, signals)
        return filteredsignals


class Beamformer:    
    def align_combine_signals(self,psi,signals):
        #output aligned signals and combined

        #first element assumed to have zero spatial delay  
        beamformed = signals[0]
        #align rest of signals 
        for i in range(1,len(signals)):
            #applying the phase in opposite sign  - ONLY KEEPS REAL PART FOR BPSK FOR NOW 
            beamformed += np.real(signals[i]*np.exp(+1j*psi[i]))
    
        return beamformed

class Mixer:
    def __init__(self,conf):
        self.conf = conf 
        self.dsp = DiscreteSP()

    def superhet(self, signal, signal_fs):
        #assumes signal is array of element signals!
        fs_mixer = 3*self.conf.fcar

        #for superhet, flo != fcar, just above zero-IF
        flo_mixer = self.conf.fcar - self.conf.fbb
        
        mixer_output = [] 
        for i in range(self.conf.no_elem): 
            lo_time = np.arange(len(signal[i]))*(1/fs_mixer)
            lo_signal = np.cos(2*np.pi*flo_mixer*lo_time)

            #mix signal by multiplication 
            mixed_signal = signal[i] * lo_signal 
            mixer_output.append(mixed_signal)

        return mixer_output

    def quadrature(self,signal,fs_signal):
        #mixer sampling frequency is 3x the carrier frequency 
        fs_mixer = 3*self.conf.fcar 

        #flo  = fcarrier 
        flo_mixer = self.conf.fcar

        #match sampling rates for discrete time mixing 
        if fs_signal != fs_mixer:
            signal_resampled, fs_resampled = self.dsp.resample_signal(
                targetfs=fs_mixer,
                oldfs=fs_signal,
                signal=signal)
            if fs_resampled != fs_mixer:
                raise ValueError("Resampling did not work!")
        else:
            fs_resampled = fs_signal
            signal_resampled = signal
        
        #local oscillators for I and Q component
        lo_time = np.arange(len(signal_resampled))*(1/fs_mixer)
        lo_signal = np.exp(1j*2*np.pi*flo_mixer*lo_time) 

        #mix signal by multiplication 
        mixer_output = signal_resampled * lo_signal

        return ResampledSignals(mixer_output, lo_time, fs_mixer)

class SpecAn: 
    def calculate_psd(self ,fs, signal):
        """
        #Most basic calculation for spectrum 
        X = np.fft.fft(signal)
        N = len(X)
        f = np.fft.fftfreq(N,1/fs)
        
        #Using scipy periodogram
        f, X = periodogram(signal, fs)
        """
        f, X = welch(
                signal, 
                fs=fs, 
                window='hann',             # Window function applied to each segment
                nperseg=len(signal)/8,
                return_onesided=False,
                scaling='density',       # Returns V^2/Hz (use 'spectrum' for V^2)
                noverlap=len(signal)/(8*2)
            )

        f = np.fft.fftshift(f)
        X = np.fft.fftshift(X)
        return Specan(f,X/np.max(X))
