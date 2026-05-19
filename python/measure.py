"""
Used to probe signal and make plots 
- SpecAn 
    - fourtier transform 
    - find peaks & distance 
    - find peak distance 

"""
from scipy.fft import fft, fftfreq
import numpy as np
from receiver import Receiver
from scipy.signal import find_peaks, peak_widths, correlate, correlation_lags
import matplotlib.pyplot as plt 

class SpecAn:
    def __init__(self):
        pass

    def fft(self, signal, samplerate, duration):
        Nwindow = samplerate*duration
        freq = fft(signal)
        fbins = fftfreq(Nwindow, 1/samplerate)

        return freq, fbins

    def find_peaks_distance(self, signal): 
        peaks, _ = find_peaks(signal, height=0)
        pdist_first =  signal[peaks] 
        return peaks, pdist_first
        
class Correlation:
    def __init__(self):
        pass

    def acf(self, code, corr_len):
        lags = correlation_lags(corr_len, corr_len)
        corr = correlate(code, code, mode="full")

        peaks, _ = find_peaks(corr)
        peak_full_width = peak_widths(corr, peaks, rel_height=1)
        
        return corr, lags, peak_full_width

    def ccf(self, signal1, signal2):
        lags = correlation_lags(len(signal1), len(signal2))
        xcorr = correlate(signal1, signal2, mode="same")

        return xcorr 
        

