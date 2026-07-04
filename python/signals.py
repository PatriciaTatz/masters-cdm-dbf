"""
Simulate effects of channel
- AWGN
- Doppler shift 
"""
import random
from math import pi

import commpy.channels as ch  # type:ignore[import-untyped]
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from components import Filter, SpecAn
from config import CDMcfg


class DiscreteSignal:
    #Method to generate discrete signal on the phase array 
    #Assumed output after the RF-front end. That is after carrier downconversion and LPF. 

    def __init__(self, conf):
        self.conf = conf

        #generate signal/s
        self.mf = Filter(conf)
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
        no_symbols = int(self.conf.simtime*self.conf.rb)
        self.symbols = np.random.choice(values, no_symbols) 
        #self.bpsksymbols = [1, -1, 1, -1, 1, 1, -1, -1, 1, 1]

        #add rrcfilter
        self.transmit, self.symbols = self.mf.rrcfilter(self.symbols, self.conf.fs)

    def add_awgn_to_elems(self, clean_signals, SNR):
        txdirty = [] 
        for i, element_signal in enumerate(clean_signals):
            #this will add noise, unique AWGN, to each element 
            #overwrite original signal 
            #do not return anything 
            txdirty.append(ch.awgn(element_signal, SNR))
        return txdirty 
        

    def create_timeseries(self):
        #the timestep is the symbol period divided by how much each sample was upsampled by the RRC filter
        self.dt = 1 / self.conf.fs 
        self.time =  np.arange(len(self.transmit)) * self.dt
        
    def create_elementsignals(self):
        #apply phase rotation as delay is too small to be seen as lagged samples. Why? 
        psi = pi 
        #boresight element has no shift create_elementsignals
        self.elements = [self.transmit]
        for i in range(1,self.conf.no_elem):
            signalNs = self.transmit*np.exp(1j*i*psi) 
            self.elements.append(signalNs)




if __name__ == "__main__":
    #show how to initiate the signal for whatever use 

    """
    Ts: symbol period = e.g. 10 seconds
    samples_symbol: samples per symbol 
    Ns: no of symbols = e.g. 5 unique symbols which lasts 10 seconds each post filter 
    modType: modulation type = BPSK 
    no_elem: number of array signals t 
    """
    #==================================
    #RUNSIMULATOR
    #==================================

    Rb = 6.25e6
    Tb = 1/Rb
    symbols = 6
    totaltime = symbols*Tb

    config = CDMcfg(
        rb = Rb,
        lc = 4,
        sps_chip= 1,
        alpha = 0.5,
        simtime = totaltime,
        no_elem = 2
    )
    
    signal = DiscreteSignal(config)


    #==================================
    #PLOTRESULTS
    #==================================
    plt.figure()
    plt.plot(signal.time*1e9, signal.transmit, '--o', label="RRC Filtered Tx Signal")
    plt.title(f"Signal Generator, Tsim={config.simtime*1e9:g}ns")
    plt.stem(signal.time[:len(signal.symbols)]*1e9,signal.symbols, 'red', label="Upsampled Symbols")
    plt.grid()
    plt.xlabel("Time (nsec)")
    plt.ylabel("Amplitude")
    plt.legend()

    plt.figure()
    plt.title(f"PSD of Tx Signal")
    specan = SpecAn().calculate_psd(
        config.fs,
        signal.elements[0],
        threshold=-3
        )
    plt.plot(specan.f/1e6, specan.Pxx_dB, label="PSD")
    plt.text(
        0.02, 0.95,
        f"99.9% BW={specan.bw/1e6:.2f}MHz",
        transform=plt.gca().transAxes,
        ha='left',
        va='top',
        bbox=dict(facecolor='white', edgecolor='black')
    )
    plt.grid()

    plt.figure()
    plt.grid()
    plt.title("Two element signals with phase delay")
    plt.plot(signal.time*1e9, signal.elements[0], '-', label="Element 1")
    plt.plot(signal.time*1e9, signal.elements[1], '-', label="Element 2")
    plt.xlabel("Time (nsecs)")
    plt.legend()
    plt.show()