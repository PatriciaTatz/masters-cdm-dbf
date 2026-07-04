"""
Used to probe signal and make plots 
- SpecAn 
    - fourtier transform 
    - find peaks & distance 
    - find peak distance 

"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq

from receiver import Receiver


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
        
