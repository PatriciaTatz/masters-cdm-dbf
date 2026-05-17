"""
Used to probe signal 
- SpecAn 
    - fourtier transform 
    - find peaks & distance 
    - find peak distance 

"""
from scipy.fft import fft, fftfreq
import numpy as np
import mysignal as s
from receiver import *
from scipy.special import erfc
from scipy.signal import find_peaks, peak_widths, correlate, correlation_lags
import matplotlib.pyplot as plt 
from matplotlib.ticker import LogLocator, LogFormatterMathtext

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
        

class BER:
    def __init__(self, signalconfig, codeconfig, arrpath):
        self.sigcfg = signalconfig
        self.codecfg = codeconfig
        self.arrpath = arrpath


    def curve(self, maxSims, maxSNR):
        self.snr_range = np.arange(0,maxSNR, 1)
        self.ber_theory = 0.5 * erfc(np.sqrt(self.snr_range))
        self.ber_curve = [] 
        for snr in self.snr_range:
            ber_sum = 0
            print("snr=",snr)
            #generate signal 
            inputSignal = s.DiscreteSignal(self.sigcfg, channel="withNoise", SNR=snr)     

            for _ in range(maxSims):
                #generate modulation 
                rx = Receiver(self.codecfg)

                self.txbits = np.sign(inputSignal.arrSignals)
                self.rxbits = np.sign(rx.signals.arrayDecisionSymbols[self.arrpath])    

                ber_sum += np.mean(self.txbits!=self.rxbits)
        
            self.ber_curve.append(ber_sum/maxSims)

        return self.snr_range, self.ber_curve, self.ber_theory

    def plot_curve(self,show_plot):
    # Plot BER curve
        fig, ax = plt.subplots(figsize=(7,5))
        ax.semilogy(
            self.snr_range,
            self.ber_curve,
            marker='o',
            linewidth=2,
            markersize=6,
            label='Simulation'
        )
        ax.semilogy(
            self.snr_range,
            self.ber_theory,
            '--',
            linewidth=2,
            label='Theoretical BPSK BER'
        )

        # Log-scale tick formatting
        ax = plt.gca()

        # Major ticks at powers of 10
        ax.yaxis.set_major_locator(LogLocator(base=10.0))

        # Format as 10^-6 instead of 1e-6
        ax.yaxis.set_major_formatter(LogFormatterMathtext())

        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        if show_plot=="on": plt.show()
