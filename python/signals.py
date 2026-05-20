"""
Simulate effects of channel
- AWGN
- Doppler shift 
"""
import commpy.channels as ch
from math import pi 
import random 
import numpy as np  
from components import MatchedFilter, SymbolSampler
from config import signalCfg
import matplotlib.pyplot as plt 

class DiscreteSignal:
    #Method to generate discrete signal on the phase array 

    def __init__(self, sigconfig, channel, SNR):
        self.conf = sigconfig
        #generate signal/s
        self.create_symbols()
        self.filtered_signal()

        #add noise 
        if channel=="withNoise":
            self.transmit = self.add_awgn(self.transmit, SNR)

    def create_symbols(self):
        #modType: requires 
        if self.conf.modType == "BPSK":
            values = [1, -1]
        elif self.conf.modType == "QPSK":
            values = [pi/4, 3*pi/4, 5*pi/4, 7*pi/4]

        self.symbols = np.random.choice(values, self.conf.Ns)

    def filtered_signal(self):
        self.mf = MatchedFilter()
        self.transmit, self.group_delay = self.mf.txrrcfilter(self.symbols, self.conf.Ts)

    def add_awgn(self, signal, SNR):
        return ch.awgn(signal,SNR)
    
class ArraySignals: 
    def create_rxsignals(self, Lss, Ns):
        self.Spread = np.empty((0,Lss))
        self.SprSum = np.empty((0,Lss))
        self.Despread= []
        self.Symbols = np.empty((0,Ns))

    def create_inputsignals(self, signal, no_arrays):
        self.signal = signal
        txsignal = self.signal.transmit
        timedelay = 1 #time delay between array elements on antenna 
        indexshift = int(timedelay/self.signal.conf.Ts)

        self.inputSignals = np.empty((0,len(txsignal)))

        for i in range(no_arrays):
            nSignal =  np.roll(txsignal, shift=i*indexshift)
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
    sigCfg = signalCfg(Ts=5, modType="BPSK", Ns=1)  
    signal = DiscreteSignal(sigCfg, channel="noNoise", SNR=20)

    sampler = SymbolSampler()
    sampled_symbols, idx = sampler.symbols(
        signal.transmit,
        signal.group_delay,
        signal.conf.Ts,
        signal.conf.Ns)

    fig, axs = plt.subplots(3, 1)
    axs[0].stem(signal.symbols)
    axs[0].set_title("Random Symbols")
    axs[1].plot(signal.transmit, '--o')
    axs[1].set_title("Transmited Signal")
    axs[2].plot(signal.transmit, '--o')
    axs[2].stem(idx, sampled_symbols,  linefmt='r--')
    axs[2].set_title("Sampled Symbols w/o Matched Filter")

    for ax in axs:
        ax.grid(True)

    plt.show()