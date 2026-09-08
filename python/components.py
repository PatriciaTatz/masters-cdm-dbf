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
from scipy.signal import periodogram, max_len_seq, resample_poly, welch, butter, freqz, filtfilt
import matplotlib.pyplot as plt 
from fractions import Fraction

from models import Signals_Time, ResampledSignals, Specan, SpreadSignals


class DiscreteSP:
    #for discrete mixing you need to have equivalent sampling frequency 

    def resample_signal(self,targetfs, oldfs, signal):
        ratio = Fraction(targetfs / oldfs).limit_denominator()

        up = ratio.numerator
        down = ratio.denominator

        rsignal = resample_poly(signal, up, down)

        newfs = oldfs * up / down

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


    def spreader(self, signals, fs_signal, f_signal, codes):
        f_code = self.conf.lc * f_signal       #Transition xLc times over one period of signal
        fs_code = self.conf.sps_chip*f_code  
        ss = []
        os = []
        exp_codes = []
        #create code sequence 

        print("fs_code=", fs_code)
        print("fs_signal=", fs_signal)
        for i in range(self.conf.no_elem):
            #upsample is required to do discrete multiplication 
            if fs_code != fs_signal:
                resampled_signal, fs_resampled = self.dsp.resample_signal(
                    targetfs=fs_code,
                    oldfs=fs_signal,
                    signal=signals[i]
                )
            os.append(resampled_signal)
            print("resampled=", fs_resampled)

            #generate code sequencec 
            code_sequence = np.repeat(codes[i], self.conf.sps_chip)                       #upsamples to f[+1 +1] -> [+1 +1 +1 +1] if sps_chip=2
            tile_repeats = int(np.ceil(len(resampled_signal) / len(code_sequence)))       #calculate number of repeats required 
            expanded_code = np.tile(code_sequence, tile_repeats)[:len(resampled_signal)]  #expand then trim to signal length  
            exp_codes.append(expanded_code)             

            #spreading operation
            spread_signal = resampled_signal*expanded_code
            ss.append(spread_signal)
    
        time_before = np.arange(len(signals[i]))/fs_signal
        time = np.arange(len(ss[0]))/fs_resampled

        if max(time)>max(time_before)*1.2:
            raise ValueError("Something went wrong with upsampling of the time series! Exceeds sim time")

        """
        #PLOT FOR CODE SERIES ONLY
        for i in range(len(exp_codes)):
            print("code=", codes[i])
            plt.rcParams['font.size'] = 15
            plt.figure()
            plt.title(f"Code sequence index={i+1}")
            plt.plot(time, exp_codes[i], '-og', label="WH Code Sequence")
            plt.legend(loc='upper right')
            plt.grid()
            plt.xlabel("Time (seconds)")
            plt.ylabel("Amplitude")
            plt.show()
        """
        return SpreadSignals(os, ss, time, exp_codes, fs_resampled)

    def recover(self, signal, codes, fs_signal, f_signal):

        if not self.conf.CALIBRATE:
            #despread returns a Spread Signal object
            dss = self.despread(
                signal,
                codes,
                fs_signal=fs_signal,
                f_signal=f_signal)

            #returns N arrays of signals and symbols!
            decorrelated, symbols = self.decorrelate(
                dss.signals,
                fs_signal=dss.fs)

            dss.signals = decorrelated
            dss.time = np.arange(len(dss.signals[0]))*(1/dss.fs)
            return dss
        else:
            dss = signal 
            decorrelated, symbols = self.decorrelate(
                dss,
                fs_signal=fs_signal)
            return decorrelated
        

    def despread(self, sumsignal, codes,fs_signal,f_signal):
        #Despread signal by multiplying with code block
        dss = [] #despread

        f_code = self.conf.lc * f_signal       #Transition xLc times over one period of signal
        fs_code = self.conf.sps_chip * f_code  
        dss = []
        symbol_elements = []
        os = []
        exp_codes = []

        #create code sequence 
        print("fs_code=",fs_code)
        print("fs_signal=",fs_signal)

        for i in range(self.conf.no_elem):
            #upsample is required to do discrete multiplication 
            if fs_code != fs_signal:
                resampled_signal, fs_resampled = self.dsp.resample_signal(
                    targetfs=fs_code,
                    oldfs=fs_signal,
                    signal=sumsignal
                )
            else:
                #No resampling required
                resampled_signal = sumsignal
                fs_resampled = fs_signal
            
            os.append(resampled_signal)
            code_sequence = np.repeat(codes[i], self.conf.sps_chip)                       #upsamples to f[+1 +1] -> [+1 +1 +1 +1] if sps_chip=2
            tile_repeats = int(np.ceil(len(resampled_signal) / len(code_sequence)))       #calculate number of repeats required 
            expanded_code = np.tile(code_sequence, tile_repeats)[:len(resampled_signal)]  #expand then trim to signal length  
            exp_codes.append(expanded_code)             

            #spreading operation
            despread_signal  = resampled_signal*expanded_code
            dss.append(despread_signal)
            
        time = np.arange(len(dss[0]))/fs_resampled

        return SpreadSignals(os, dss, time, exp_codes, fs_resampled) 

    def decorrelate(self, despread_signal, fs_signal):
        decorr = [] 
        syms = []
        for i in range(self.conf.no_elem):
            decorrelated_signal, symbols = self.mf.rrcfilter(
                despread_signal[i],
                fs_signal=fs_signal,
                mode="rxmode"
                )
            decorr.append(decorrelated_signal)
            syms.append(symbols)

        return decorr, syms


class Filter:
    def __init__(self, conf):
        self.conf = conf

    def rrcfilter(self, signal, fs_signal, mode="txmode"):
        #upsampling
        if mode == "txmode":
            sps_symbol  = self.conf.sps_symbol              #number of samples per symbol
            ts = 1 / (self.conf.fb)                         #symbol period in seconds 
            symbols = np.zeros(
                len(signal)*sps_symbol,
                dtype=signal.dtype)
            symbols[::sps_symbol] = signal
            samples = symbols            
        else:
            sps_symbol = int(round(fs_signal/self.conf.fb)) #now we are filtering to resolve the fIF 
            ts = 1/self.conf.fb                             #baseband symbol rate 
            samples = signal                                #this is the full rx signal 
        

        #evaluate filter parameters
        ntaps = int(self.conf.span*sps_symbol) #must be odd, total length of the filter 
        delay = ntaps//2 - sps_symbol//2 #remove the odd component 

        #generate filter coefficients
        dt_filter, h_rrc = filter.rrcosfilter(
            ntaps,
            self.conf.alpha,
            ts,
            fs_signal
            )
        
        #filter normalisation 
        h_rrc /= np.sqrt(np.sum(h_rrc**2))

        #apply filter to the samples
        filtered = np.convolve(samples, h_rrc, mode='full')

        #remove delay caused by the filter
        if mode == "txmode":
            filtered_no_delay = filtered[delay+2*sps_symbol:-delay]
        else:
            filtered_no_delay = filtered[delay+sps_symbol//2:-delay]

        #sample at the symbol
        symbols = filtered_no_delay[::sps_symbol]

        """
        #Plot for pulse-shaping and constellation
        if self.conf.modType=="BPSK":

            if mode == 'txmode':
                #Figure 3.4
                offset = sps_symbol//2
                time_sample_right_shift = np.arange(len(samples[:sps_symbol*10])) + offset
                time = np.arange(0, len(filtered))
                plt.figure(figsize=(15,6))
                plt.rcParams['font.size'] = 15
                plt.stem(time_sample_right_shift, samples[:sps_symbol*10], label="Upsampled Symbols")
                plt.plot(filtered_no_delay[:sps_symbol*10], '-o', c='orange', label="RRC filtered symbols w/o group delay")
                plt.grid()
                plt.title("BPSK with RRC Pulse Shaping")
                plt.ylabel("Symbol")
                plt.xlabel("Sample Index\n")
                plt.legend(loc='upper right')
                #Figure 3.5, constellation q
                plt.figure(figsize=(10,8))
                plt.rcParams['font.size'] = 15
                plt.axhline(y=0, color='orange', linestyle='-', linewidth=1.5)
                plt.axvline(x=0, color='orange', linestyle='-', linewidth=1.5)
                plt.title(f"Constellation Diagram of BPSK Signal")
                plt.ylabel("Quadrature (Q)")
                plt.xlabel("In-Phase (I)")
                plt.scatter(
                    np.real(symbols),
                    np.imag(symbols),
                    color="blue",
                    alpha=0.5,
                    s=30,
                )
                plt.grid()
            else:
                #Figure 3.4
                offset = sps_symbol
                idx = np.arange(0, len(filtered_no_delay))
                plt.figure(figsize=(15,6))
                plt.rcParams['font.size'] = 15
                plt.stem(
                    idx[::sps_symbol][1:-1],
                    filtered_no_delay[::sps_symbol][1:-1],
                    '-o', label="Symbols at Receiver"
                    )
                plt.plot(filtered_no_delay, '-o', c='orange', label="Rx RRC Matched Filter")
                plt.grid()
                plt.title("BPSK with RRC Pulse Shaping")
                plt.ylabel("Symbol")
                plt.xlabel("Sample Index\n")
                plt.legend(loc='upper right')


                #Figure 3.5, constellation 
                plt.figure(figsize=(8,8))
                theta = np.linspace(0, 2*np.pi, 1000)
                plt.plot(np.cos(theta),np.sin(theta),'k--', alpha=0.5)

                plt.rcParams['font.size'] = 15
                plt.axhline(y=0, color='orange', linestyle='-', linewidth=1.5)
                plt.axvline(x=0, color='orange', linestyle='-', linewidth=1.5)
                plt.title(f"Constellation Diagram of BPSK Signal")
                plt.ylabel("Quadrature (Q)")
                plt.xlabel("In-Phase (I)")
                plt.scatter(
                    np.real(filtered_no_delay[::sps_symbol][1:-1]),
                    np.imag(filtered_no_delay[::sps_symbol][1:-1]),
                    color="blue",
                    alpha=0.5,
                    s=30,
                )
                plt.tight_layout()
                plt.grid()


        else:
            plt.figure(figsize=(15,10))
            plt.rcParams['font.size'] = 15
            plt.suptitle("QPSK with RRC Pulse Shaping")
            plt.subplot(2, 1, 1)
            plt.stem(np.real(samples[:sps_symbol*10]),  label="Upsampled Symbols")
            plt.plot(np.real(filtered_no_delay[:sps_symbol*10]),  '-o', c='orange', label="RRC filtered symbols w/o group delay")
            plt.ylabel("Real Part (In-Phase / I)")
            plt.grid()
            plt.legend(loc='upper right')

            plt.subplot(2, 1, 2)
            plt.stem(np.imag(samples[:sps_symbol*10]),  label="Upsampled Symbols")
            plt.plot(np.imag(filtered_no_delay[delay:sps_symbol*10:sps_symbol]),  '-o', c='orange', label="RRC filtered symbols w/o group delay")
            plt.ylabel("Imaginary Part (Quadrature / Q)")
            plt.xlabel("Sample Index")
            plt.grid()

            plt.legend(loc='upper right')
        plt.show()
        """
        return filtered_no_delay, symbols

    def butterworth(self, cutoff, fs, signals, filter_type="low"):
        print(cutoff)
        #fs is sampling frequency! 
        nyquist = fs/2
        #cutoff = 1.5*self.fb
        b,a= butter(
            N=4, 
            Wn = cutoff/nyquist, 
            btype=filter_type
        )
        
        """
        #computer frequency response
        #w in radians/sample, h the complex frequency response of filter
        w, h = freqz(b,a, fs=fs)
        #convert rad/sample to Hz
        plt.figure(figsize=(10,7))
        plt.rcParams['font.size'] = 15

        plt.semilogx(w/1e6, 20 * np.log10(abs(h)), label="LPF Response")
        plt.title(f'4th order Butterworth LPF frequency response')
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Amplitude (dB)')
        plt.grid(which='both', axis='both')
        plt.axvline(cutoff/1e6, color='green', linewidth=2, label=f"Cuttoff Frequency = {cutoff/1e6} MHz") # cutoff frequency
        plt.legend()
        plt.show()
        """

        #filter can be applied to an array type which signal is 
        filteredsignals = filtfilt(b,a,signals)
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

    def superhet(self, signal, fs_signal, flo):
        #assumes signal is array of element signals!
        fs_mixer = 3*(self.conf.fc + flo) 

        #for superhet, flo != fc, just above zero-IF
        flo_mixer = flo
        print(flo)
        
        mixer_output = [] 
        for i in range(self.conf.no_elem): 
            if fs_signal != fs_mixer:
                signal_resampled, fs_resampled = self.dsp.resample_signal(
                    targetfs=fs_mixer,
                    oldfs=fs_signal,
                    signal=signal[i])
                if fs_resampled != fs_mixer:
                    raise ValueError("Resampling did not work!")
            lo_time = np.arange(len(signal_resampled))*(1/fs_mixer)
            lo_signal = np.cos(2*np.pi*flo_mixer*lo_time)

            #mix signal by multiplication 
            mixed_signal = signal_resampled * lo_signal 
            mixer_output.append(mixed_signal)

        return ResampledSignals(mixer_output, lo_time, fs_resampled)

    def quadrature(self,signal,fs_signal, flo):
        #mixer sampling frequency is 3x the carrier frequency 

        if fs_signal <= flo: 
            #upconverting to carrier 
            #fs_signal is just symbol sample frequency at this stage 
            #recall fb is symbol frequency exactly but now has bandwidth of b=(1+alpha)fb
            fs_mixer = 3*self.conf.fc 
        elif fs_signal > flo:
            #downconverting to zero-IF 
            #fs_signal is now the spread signal sample frequency 
            #recall fcode is the spread signal frequency 
            fs_mixer = 3*fs_signal 
        #flo  = fcr
        flo_mixer = flo

        #match sampling rates for discrete time mixing 
        if fs_signal != fs_mixer:
            signal_resampled, fs_resampled = self.dsp.resample_signal(
                targetfs=fs_mixer,
                oldfs=fs_signal,
                signal=signal)
            if fs_resampled != fs_mixer:
                raise ValueError("Resampling did not work!")

        #local oscillators for I and Q component
        lo_time = np.arange(len(signal_resampled))*(1/fs_mixer)

        if fs_signal < flo:
            #quadrature upconversion
            lo_signal = np.exp(+1j*2*np.pi*flo_mixer*lo_time) 
        else:
            #quadrature downconversion
            lo_signal = np.exp(-1j*2*np.pi*flo_mixer*lo_time) 

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
        q
        #Using scipy periodogram
        f, X = periodogram(signal, fs)
        
        f, X = welch(
                signal, 
                fs=fs, detrend=False,
                window='hann',              # Window function applied to each segment
                nperseg=len(signal)/40,#4096,      #frequency bins averaged over segments
                return_onesided=False,      #double sided to see rrc filtered effect
                scaling='density',        # Returns V^2/Hz (use 'spectrum' for V^2)
                noverlap=len(signal)/(40*2)   #overlap of frequency bins 
            )
        #this shifts a twosided spectrum from 0 +f -f into -f 0 +f 
        f = np.fft.fftshift(f)
        X = np.fft.fftshift(X)
        
        """
        x = signal
        X = 10*np.log10(np.fft.fftshift(np.abs(np.fft.fft(x))**2)/len(x))
        f = np.linspace(-fs/2, fs/2, len(X))
        
        #Xnorm
        X -= np.max(X)
        return Specan(f,X)
