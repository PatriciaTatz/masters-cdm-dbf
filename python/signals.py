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

from components import Filter, SpecAn, Mixer
from config import CDMcfg
from models import ResampledSignals


class DiscreteSignal:
    #Method to generate discrete signal on the phase array 
    #Assumed output after the RF-front end. That is after carrier downconversion and LPF. 

    def __init__(self, conf):
        self.conf = conf

        #generate signal/s
        self.filters = Filter(conf)
        txsignal = self.create_symbols() #create symbols, rrc filtered
        self.create_timeseries(txsignal) #create time series based on input symbol rate

        if self.conf.INTERFERENCE: 
            #ADD INTERFERENCE - removed by despreading process & MF 
            N = len(txsignal)
            BW_int = 1e6
            Rs = BW_int
            sps = int(self.conf.fs / Rs)
            bits = np.random.choice([-1, 1], int(np.ceil(N/sps)))
            symbols = np.repeat(bits, sps)[:N]
            interference =  symbols * np.exp(1j*2*np.pi*7.5e6*self.time)

            txsignal += np.real(interference)

        if self.conf.CARRIER:
            self.mixer = Mixer(conf)
            #modulation will change time array anfs_mixer, carrier_time, carrier_signald sampling frequency
            #and over txsignal which now represents a QPSK modded carrier 
            modulated = self.modulate_to_carrier(txsignal)
            self.fs = modulated.fs
            self.time = modulated.time
               
        #create elements that will be received 
        self.create_elementsignals(modulated.signals)

    def create_symbols(self):
        #This does not worry about bit generation and modulation 
        no_symbols = int(self.conf.simtime*self.conf.rb)

        #modType, Ns, Ts: required
        if self.conf.modType == "BPSK":
            constellation = np.array([1, -1])
            
        elif self.conf.modType == "QPSK":
            #normalised QPSK symbols 
            constellation = np.array([
                1 + 1j,
                1 - 1j,
                -1 + 1j,
                -1 -1j
            ])/ np.sqrt(2)
        else:
            raise ValueError("Unsupported modulation scheme. Choose [BPSK, QPSK]")
        #Random symbol generator    
        self.txsymbols = np.random.choice(constellation, no_symbols) 
    
        #OVERWRITE WARNING!! FOR REPEATABLE RESULTS SAVE DATA
        #np.save("input", self.txsymbols)

        #FOR REPEATABLE RESULTS LOAD DATA
        #self.txsymbols = np.load("input.npy")

        #add rrcfilter
        self.transmit, self.rrcsymbols = self.filters.rrcfilter(self.txsymbols, self.conf.fs)
        self.symbol_time = np.arange(len(self.rrcsymbols))/self.conf.fs
        #np.save("transmit_clean", self.transmit)
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


    def modulate_to_carrier(self, signal):
        #carrier frequency = flo in config 
        #cutoff frequency is equal to pulse shaped bandwidth 

        
        #PLOT FOR BEFORE THE MODULATION
        plt.figure()
        plt.title(f"Before U/C: Power Spectrum Density of Tx Signal on Element=1")
        specan = SpecAn().calculate_psd(
            self.conf.fs, 
            signal,
            )
        plt.plot(specan.f, specan.Pxx_dB, '-b')
        plt.grid()
        plt.ylabel("Frequency kHz")
        
        #modulate to carrier
        modsignal = self.mixer.quadrature(
            signal,
            self.conf.fs
        )
        #bandpass filter to remove harmonics caused by resampling 
        flow = self.conf.fcar-self.conf.rb
        fhigh = self.conf.fcar+self.conf.rb
        filtered_modsignal = self.filters.butterworth(
            cutoff = np.array([flow,fhigh]),
            fs = modsignal.fs,
            signals = modsignal.signals,
            filter_type="bandpass",
        )
        modsignal.signals = filtered_modsignal

        """
        #PLOT FOR AFTER THE MODULATION
        plt.figure()
        plt.title(f"After U/C: Power Spectrum Density of Tx Signal on Element=1")
        specan = SpecAn().calculate_psd(
            modsignal.fs,
            modsignal.signals,
            )
        plt.plot(specan.f, specan.Pxx_dB, label="PSD")
        plt.xlabel("Frequency MHz")
        plt.xlim(0,fhigh)
        plt.grid()
        """
        return ResampledSignals(modsignal.signals, modsignal.time,modsignal.fs)


    def show_signal_plot(self):
        plt.figure()
        elements_to_plot = range(len(self.elements))
        for i in elements_to_plot:
            if self.conf.modType=="BPSK":
                plt.stem(signal.symbol_time, signal.rrcsymbols , 'orange')
                plt.plot(self.time, self.elements[i], '-', label=f"Element {i} Signal")
                plt.grid()

            elif self.conf.modType=="QPSK":
                plt.subplot(2, 1, 1)
                plt.stem(self.symbol_time, np.real(self.rrcsymbols), 'orange')
                plt.plot(self.time, np.real(self.elements[i]), '-b')
                plt.title("Real Part (In-Phase / I)")
                plt.grid()

                plt.subplot(2, 1, 2)
                plt.stem(self.symbol_time, np.imag(self.rrcsymbols), 'orange')
                plt.plot(self.time, np.imag(self.elements[i]), '-g')
                plt.title("Imaginary Part (Quadrature / Q)")
                plt.xlabel("Sample Index")
                plt.tight_layout()
                plt.grid()

            else:
                raise ValueError("Unknown modulation scheme. Cannot plot!")
            
        plt.title(f"Tx Signals on all elements")
        plt.legend()
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")

        plt.figure()
        plt.title(f"Power Spectrum Density of Tx Signal on Element=1")
        specan = SpecAn().calculate_psd(
            self.conf.fs,
            self.elements[0],
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

        plt.figure()
        plt.title(f"Constellation Diagram of Tx Signal on Element=1")
        plt.scatter(
            np.real(self.txsymbols),
            np.imag(self.txsymbols),
            color="blue",
            alpha=0.5,
            s=10,
        )
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
    Rb = 1e3
    Tb = 1/Rb
    symbols = 1000
    totaltime = symbols*Tb

    config = CDMcfg(
        rb = Rb,
        lc = 4,
        sps_chip= 1,
        alpha = 0.5,
        simtime = totaltime,
        no_elem = 2,
        modType="BPSK"
    )
    
    signal = DiscreteSignal(config)

    #==================================
    #PLOTRESULTS
    #=================================
    signal.show_signal_plot()