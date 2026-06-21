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
        self.create_timeseries()#create time series based on input symbol rate
        self.create_elementsignals()

    def create_symbols(self):
        #modType, Ns, Ts: required
        if self.conf.modType == "BPSK":
            values = [1, -1]
        elif self.conf.modType == "QPSK":
            values = [pi/4, 3*pi/4, 5*pi/4, 7*pi/4]

        #randomized symbols 
        no_symbols = int(self.conf.simtime // self.conf.Ts)
        self.bpsksymbols = np.random.choice(values, no_symbols) 

        #add rrcfilter
        self.transmit, self.bpsksymbols = self.mf.txrrcfilter(self.bpsksymbols, self.conf.Ts, self.conf.samples_symbol)

    def add_awgn(self, signal, SNR):
        return ch.awgn(signal,SNR)

    def create_timeseries(self):
        #the timestep is the symbol period divided by how much each sample was upsampled by the RRC filter
        self.dt = self.conf.Ts/self.conf.samples_symbol
        self.time =  np.arange(len(self.transmit)) * self.dt
        

    def create_elementsignals(self):
        #apply phase rotation as delay is too small to be seen as lagged samples. Why? 
        psi = pi 
        #boresight element has no shift create_elementsignals
        self.elements = [self.transmit]
        for i in range(1,self.conf.no_arrays):
            signalNs = self.transmit*np.exp(1j*i*psi) 
            self.elements.append(signalNs)

    def calculate_psd(self):
        fs = self.conf.samples_symbol/self.conf.Ts  #sampling rate is how much ive upsampled (samples per symbol) / symbol period ==  sampling frequency
        f,Pxx = welch(self.transmit, fs, return_onesided=False, detrend=False)
        idx = np.argsort(f)
        f = f[idx]
        Pxx = Pxx[idx]
        df = f[1] - f[0]
        cumulative = np.cumsum(Pxx) * df
        total_power = cumulative[-1]
        low_idx = np.where(cumulative >= 0.0005 * total_power)[0][0]
        high_idx = np.where(cumulative >= 0.9995 * total_power)[0][0]
        self.bw = f[high_idx] - f[low_idx]
        return f, Pxx


if __name__ == "__main__":
    #show how to initiate the signal for whatever use 

    """
    Ts: symbol period = e.g. 10 seconds
    samples_symbol: samples per symbol 
    Ns: no of symbols = e.g. 5 unique symbols which lasts 10 seconds each post filter 
    modType: modulation type = BPSK 
    no_arrays: number of array signals t 
    """
    config = CXBF(
        Ts = 1/6.25e6,
        samples_symbol = 20,
        simtime=50/6.25e6, 
        modType="BPSK",
        no_arrays=2
    )
    
    #repeating the signal does not behave well in the PSD so leave that for now and just translate this one signal 

    signal = DiscreteSignal(config)
    f, Pxx = signal.calculate_psd()
    #plot symbol + simulated timeseries
    plt.figure()
    plt.plot(signal.time*1e9, signal.transmit, '--o')
    plt.title(f"Simulated Rx Signal Tsim={config.simtime}")
    plt.stem(signal.time[:len(signal.bpsksymbols)]*1e9,signal.bpsksymbols, 'orange')
    plt.grid()
    plt.xlabel("Time (nsec)")
    plt.ylabel("Amplitude")

    plt.figure()
    plt.semilogy(f, Pxx, label="PSD")
    plt.text(
        0.02, 0.95,
        f"99.9% BW={signal.bw/1e6:.2f}MHz",
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

    plt.figure()
    plt.plot(signal.timeidx, signal.element[0], '-', label="Element 1")
    plt.plot(signal.timeidx, signal.element[1], '-', label="Element 2")
    plt.legend()
    plt.show()