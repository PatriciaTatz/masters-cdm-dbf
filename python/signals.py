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
from config import signalCfg
import matplotlib.pyplot as plt 

class DiscreteSignal:
    #Method to generate discrete signal on the phase array 

    def __init__(self, sigconfig):
        self.conf = sigconfig
        #generate signal/s
        self.create_symbols()
        self.filtered_signal()


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

    def symbol_sampler(self, txsignal):
        group_delay = int(np.round(self.group_delay))
        interval = int(np.round(self.mf.ups))

        idxs = group_delay + np.arange(self.conf.Ns)*interval
        samples = txsignal[idxs]
        return samples, idxs

    
class ArraySignals: 
    def create_rxsignals(self, Lss, Ns):
        self.Spread = []
        self.SprSum = []
        self.Despread= []

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
    sigCfg = signalCfg(Ts=5, modType="BPSK", Ns=10)  
    signal = DiscreteSignal(sigCfg)
    #signal.transmit = signal.add_awgn(signal.transmit,SNR=5)
    sampled_symbols, idx = signal.symbol_sampler(signal.transmit)

    fig, axs = plt.subplots(3, 1)
    axs[0].stem(signal.symbols)
    axs[0].set_title("Random Symbols")
    axs[1].plot(signal.transmit, '--o')
    axs[1].set_title("Transmited Signal")
    axs[2].plot(signal.transmit, '--o')
    axs[2].stem(idx, sampled_symbols,  linefmt='r--')
    axs[2].set_title("Sampled Symbols")

    for ax in axs:
        ax.grid(True)

    plt.show()