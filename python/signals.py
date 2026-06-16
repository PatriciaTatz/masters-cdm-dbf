"""
Simulate effects of channel
- AWGN
- Doppler shift 
"""
import commpy.channels as ch
from math import pi 
import random 
import numpy as np  
from components import MatchedFilter
from config import CXBF
import matplotlib.pyplot as plt 
from scipy.signal import welch

class DiscreteSignal:
    #Method to generate discrete signal on the phase array 
    #Assumed output after the RF-front end. That is after carrier downconversion and LPF. 

    def __init__(self, sigconfig):
        self.conf = sigconfig

        #generate signal/s
        self.mf = MatchedFilter()
        self.create_symbols() #create symbols, rrc filtered
        self.timeseries()#create time series based on input symbol rate
        self.elementsignals()

    def create_symbols(self):
        #modType, Ns, Ts: required
        if self.conf.modType == "BPSK":
            values = [1, -1]
        elif self.conf.modType == "QPSK":
            values = [pi/4, 3*pi/4, 5*pi/4, 7*pi/4]

        #randomized symbols 
        self.bpsksymbols = np.random.choice(values, self.conf.no_symbols)

        #add rrcfilter
        self.rrcsymbols, self.bpsksymbols = self.mf.txrrcfilter(self.bpsksymbols, self.conf.Ts, self.conf.sps)

    def timeseries(self):
        #simtime: required
        dt = self.conf.Ts/self.conf.sps
        self.timeidx = np.arange(0, self.conf.simtime, dt)
        repeatf = int(self.conf.simtime/(self.conf.no_symbols*self.conf.Ts))
        self.signalts = np.tile(self.rrcsymbols, repeatf)

        
    def add_awgn(self, signal, SNR):
        return ch.awgn(signal,SNR)

    def elementsignals(self):
        #apply phase rotation 
        psi = pi
        #boresight element has no shift 
        self.element = self.signalts
        for i in range(1,self.conf.no_arrays):
            signalNs = self.signalts*np.exp(1j*i*psi) 
            self.element = np.vstack((self.element, signalNs))


if __name__ == "__main__":
    #show how to initiate the signal for whatever use 

    """
    Ts: symbol period = e.g. 10 seconds
    sps: samples per symbol 
    Ns: no of symbols = e.g. 5 unique symbols which lasts 10 seconds each post filter 
    modType: modulation type = BPSK 
    no_arrays: number of array signals t 
    """
    config = CXBF(
        Ts = 1/6.25e6,
        sps = 10,
        no_symbols= 500, 
        simtime=500*1/6.25e6, 
        modType="BPSK",
        no_arrays=2
    )
    
    #repeating the signal does not behave well in the PSD so leave that for now and just translate this one signal 

    signal = DiscreteSignal(config)
    fs = config.sps/config.Ts #sampling rate is how much ive upsampled (samples per symbol) / symbol period
    f,Pxx = welch(signal.signalts, fs, return_onesided=False)
    idx = np.argsort(f)
    f = f[idx]
    Pxx = Pxx[idx]
    df = f[1] - f[0]
    cumulative = np.cumsum(Pxx) * df
    total_power = cumulative[-1]
    low_idx = np.where(cumulative >= 0.0005 * total_power)[0][0]
    high_idx = np.where(cumulative >= 0.9995 * total_power)[0][0]
    bw = f[high_idx] - f[low_idx]

    #plot symbol + simulated timeseries
    plt.figure()
    plt.plot(signal.timeidx, signal.signalts, '--o')
    plt.title(f"Simulated Rx Signal Tsim={config.simtime}")
    plt.stem(signal.timeidx[:len(signal.bpsksymbols)],signal.bpsksymbols, 'orange')
    plt.grid()
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")

    plt.figure()
    plt.semilogy(f, Pxx, label="PSD")
    plt.text(
        0.02, 0.95,
        f"99.9% BW={bw/1e6:.2f}MHz",
        transform=plt.gca().transAxes,
        ha='left',
        va='top',
        bbox=dict(facecolor='white', edgecolor='black')
    )
    plt.text(
        0.02, 0.85,
        f"Theoretical BW={1/config.Ts*(1+0.5)/1e6}MHz",
        transform=plt.gca().transAxes,
        ha='left',
        va='top',
        bbox=dict(facecolor='white', edgecolor='black')
    )

    #plot all array elements 
    plt.figure()
    plt.plot(signal.timeidx, signal.element[0], '-', label="Element 1")
    plt.plot(signal.timeidx, signal.element[1], '-', label="Element 2")
    plt.legend()
    plt.show()