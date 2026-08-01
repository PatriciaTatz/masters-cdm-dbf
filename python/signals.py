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
from math import sin 

from components import Filter, SpecAn
from config import CDMcfg


class DiscreteSignal:
    #Method to generate discrete signal on the phase array 
    #Assumed output after the RF-front end. That is after carrier downconversion and LPF. 

    def __init__(self, conf):
        self.conf = conf

        #generate signal/s
        self.mf = Filter(conf)
        txsignal = self.create_symbols() #create symbols, rrc filtered
        self.create_timeseries(txsignal) #create time series based on input symbol rate

        """
        #ADD INTERFERENCE - removed by despreading process & MF 
        N = len(txsignal)
        BW_int = 1e6
        Rs = BW_int
        sps = int(self.conf.fs / Rs)
        bits = np.random.choice([-1, 1], int(np.ceil(N/sps)))
        symbols = np.repeat(bits, sps)[:N]
        interference =  symbols * np.exp(1j*2*np.pi*7.5e6*self.time)
        
        txsignal += np.real(interference)
        """      
        self.create_elementsignals(txsignal)

    def create_symbols(self):
        #modType, Ns, Ts: required
        if self.conf.modType == "BPSK":
            values = [1, -1]
        elif self.conf.modType == "QPSK":
            values = [pi/4, 3*pi/4, 5*pi/4, 7*pi/4]

        #randomized symbols 
        no_symbols = int(self.conf.simtime*self.conf.rb)
        #self.txsymbols = np.random.choice(values, no_symbols) 

        #FOR REPORT_REPEATABLE RESULTS
        self.txsymbols = np.load("input.npy")

        #add rrcfilter
        self.transmit, self.rrcsymbols = self.mf.rrcfilter(self.txsymbols, self.conf.fs)

        np.save("transmit_clean", self.transmit)
        return self.transmit

    def create_timeseries(self, signal):
        #the timestep is the symbol period divided by how much each sample was upsampled by the RRC filter
        self.dt = 1 / self.conf.fs 
        self.time =  np.arange(len(signal)) * self.dt
        
    def create_elementsignals(self, signal):
        #apply phase rotation as delay is too small to be seen as lagged samples. Why? 
        #psi = pi 
        #applying d <= lambda/2 and lambda is for 10GHz (lambda=30mm) for Xband operations
        d = 0.015
        c = 3*10**8
        theta = pi/2
        lam = 0.03
        fd = 0
        #each time this is recreated
        self.elements = [signal]
        self.psi = [0]
        for i in range(1,self.conf.no_elem):
            tau_i = d*sin(theta)
            psi_i = 2*pi*i*tau_i/lam
            signalNs = signal*np.exp(-1j*psi_i)*np.exp(+1j*2*pi*fd*self.time) 
            self.elements.append(signalNs)
            self.psi.append(psi_i)
        return self.elements 

    def add_element_noise(self, signal, noise):
        for i in range(self.conf.no_elem):
            self.elements[i] = signal + noise[i]


    def show_signal_plot(self):
        plt.figure()
        for i in range(len(self.elements)):
            plt.plot(self.time, self.elements[i], '-', label=f"Element {i} Signal")

        #plt.plot(self.time, self.transmit, '-o', label="RRC-Shaped Tx Signal")
        plt.title(f"Tx Signal on Element=1")
        #plt.stem(self.time[:len(self.rrcsymbols)],self.rrcsymbols, 'orange', label="Symbols")
        plt.legend()
        plt.grid()
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")

        plt.figure()
        plt.title(f"Power Spectrum Density of Tx Signal on Element=1")
        specan = SpecAn().calculate_psd(
            self.conf.fs,
            self.elements[0],
            psdtype="normal"
            )
        plt.plot(specan.f/1e6, specan.Pxx_dB, label="PSD")
        """
        plt.text(
            0.02, 0.95,
            f"99.9% BW={specan.bw/1e6:.2f}MHz",
            transform=plt.gca().transAxes,
            ha='left',
            va='top',
            bbox=dict(facecolor='white', edgecolor='black')
        )
        """
        plt.ylabel("Power (dB)")
        plt.xlabel("Frequency (MHz)")
        plt.grid()
        plt.show()  


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
    symbols = 2000
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
    #=================================
    signal.show_signal_plot()