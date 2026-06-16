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

    def create_symbols(self):
        #modType, Ns, Ts: required
        if self.conf.modType == "BPSK":
            values = [1, -1]
        elif self.conf.modType == "QPSK":
            values = [pi/4, 3*pi/4, 5*pi/4, 7*pi/4]

        #randomized symbols 
        self.bpsksymbols = np.random.choice(values, self.conf.no_symbols)

        #add rrcfilter
        self.rrcsymbols, self.bpsksymbols = self.mf.txrrcfilter(self.bpsksymbols, self.conf.symbol_period, self.conf.sps)

    def timeseries(self):
        #simtime: required
        dt = self.conf.symbol_period/self.conf.sps
        self.timeidx = np.arange(0, self.conf.simtime, dt)
        repeatf = int(self.conf.simtime/(self.conf.no_symbols*self.conf.symbol_period))
        self.signalts = np.tile(self.rrcsymbols, repeatf)

        
    def add_awgn(self, signal, SNR):
        return ch.awgn(signal,SNR)

    def elementsignals(self, no_arrays):
        #creates time series array for each array element 
        timedelay = 1 #time delay between array elements due to propagation distance
        indexshift = int(timedelay/self.conf.Ts)
        self.inputSignals = np.empty((0,len(self.signalts)))

        for i in range(no_arrays):
            nSignal =  np.roll(self.signalts, shift=i*indexshift)
            nSignal[:i*indexshift] = 0
            self.inputSignals = np.vstack((self.inputSignals, nSignal))


            
if __name__ == "__main__":
    #show how to initiate the signal for whatever use 

    """
    *Fs = Sample rate is always 1 sample per second 
    Ts: symbol period = e.g. 10 seconds
    Ns: no of symbols = e.g. 5 unique symbols which lasts 10 seconds each post filter 
    modType: modulation type = BPSK 
    no_arrays: number of array signals t 
    """

    Tb = 1/6.25e6
    config = CXBF(
        symbol_period= Tb,
        sps = 10,
        no_symbols= 1000, 
        simtime=1*1000*Tb, 
        modType="BPSK",
    )

    signal = DiscreteSignal(config)
    fs = 10/config.symbol_period #sampling rate is how much ive upsampled (samples per symbol) / symbol period
    f,Pxx = welch(signal.signalts, fs, nperseg=8192, return_onesided=False)
    idx = np.argsort(f)
    f = f[idx]
    Pxx = Pxx[idx]
    df = f[1] - f[0]
    cumulative = np.cumsum(Pxx) * df
    total_power = cumulative[-1]
    low_idx = np.where(cumulative >= 0.00005 * total_power)[0][0]
    high_idx = np.where(cumulative >= 0.99995 * total_power)[0][0]
    bw = f[high_idx] - f[low_idx]


    #plot symbol + simulated timeseries
    plt.figure()
    plt.plot(signal.timeidx, signal.signalts, '--o')
    plt.title(f"Simulated Rx Signal Tsim={config.simtime}")
    plt.stem(signal.timeidx,signal.bpsksymbols, 'red')
    plt.grid()
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")

    plt.figure()
    plt.semilogy(f, Pxx, label="PSD")
    plt.text(
        0.02, 0.95,
        f"99.99% BW={bw/1e6:.2f}MHz",
        transform=plt.gca().transAxes,
        ha='left',
        va='top',
        bbox=dict(facecolor='white', edgecolor='black')
    )
    plt.text(
        0.02, 0.85,
        f"Theoretical Null-Null BW={1/config.symbol_period*(1+0.5)/1e6}MHz",
        transform=plt.gca().transAxes,
        ha='left',
        va='top',
        bbox=dict(facecolor='white', edgecolor='black')
    )
    plt.show()