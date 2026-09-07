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
        txsignal = self.create_symbols().copy() #create symbols, rrc filtered, copy of transmit to avoid overwriting original waveform 
        self.create_timeseries(txsignal) #create self.time series based on input symbol rate

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

        else:
            self.create_elementsignals(txsignal)
            self.fs = self.conf.fs


    def create_symbols(self):
        #This does not worry about bit generation and modulation 
        no_symbols = int(self.conf.simtime*self.conf.rb)

        #modType, Ns, Ts: requiredq
        if self.conf.modType == "BPSK":
            constellation = np.array([1, -1])
            
        elif self.conf.modType == "QPSK":
            #normalised QPSKqq symbols 
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
        #np.save("input_thesis_QPSK_pulse_shaping", self.txsymbols)

        #FOR REPEATABLE RESULTS LOAD DATA
        self.txsymbols = np.load("input_thesis_pulse_shaping.npy")

        #add rrcfilter
        self.symbol_waveform, self.symbols = self.filters.rrcfilter(
            self.txsymbols,
            self.conf.fs
            )

        #np.save("transmit_clean", self.transmit)
        return self.symbol_waveform

    def create_timeseries(self, signal):
        #the timestep is the symbol period divided by how much each sample was upsampled by the RRC filter
        self.dt = 1 / self.conf.fs 
        self.time =  np.arange(len(signal)) * self.dt
        self.symbol_time = self.time[::self.conf.sps_symbol] #np.arange(len(self.symbols))/self.conf.fs

        
    def create_elementsignals(self, signal):
        #apply phase rotation as delay is too small to be seen as lagged samples. Why? 
        #psi = pi 
        #applying d <= lambda/2 and lambda is for 10GHz (lambda=30mm) for Xband operations
        c = 3*10**8
        theta = pi/4
        fd = 0
        #each time this is recreated
        self.elements = [signal]
        self.psi = [0]
        for i in range(1,self.conf.no_elem):
            psi_i = i*pi*sin(theta)
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

        """
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
        """
        #modulate to carrier
        modsignal = self.mixer.quadrature(
            signal,
            fs_signal = self.conf.fs,
            flo = self.conf.fcar
        )
        """
        #bandpass filter to remove harmonics caused by resampling 
        flow = self.conf.fcar-self.conf.rb
        fhigh = self.conf.fcar+self.conf.rb
        filtered_modsignal = self.filters.butterworth(
            cutoff = np.array([flow,fhigh]),
            fs = modsignal.fs,
            signals = modsignal.signals,
            filter_type="bandpass",
        )
        
        modsignal.signals = np.real(filtered_modsignal)
        """
        modsignal.signals = np.real(modsignal.signals)
        
    
        
        #PLOT FOR AFTER THE MODULATION
        plt.figure(figsize=(15,6))
        plt.rcParams['font.size'] = 15

        plt.title(f"Carrier Modulated {self.conf.modType} Signal")
        specan = SpecAn().calculate_psd(
            modsignal.fs,
            modsignal.signals,
            )
        plt.plot(specan.f/1e6, specan.Pxx_dB, label="PSD")
        plt.axvline(x=self.conf.fcar/1e6, color='red', linestyle='--', linewidth=1.5, label="Carrier Frequency")
        plt.xlim(-2*self.conf.fcar/1e6,2*self.conf.fcar/1e6)
        plt.ylabel("Normalised PSD (dB)")
        plt.xlabel("Frequency (MHz)")
        #plt.ylim(-9, 1)
        plt.legend(loc="upper right")
        plt.grid()

        plt.figure(figsize=(15,6))
        plt.rcParams['font.size'] = 15

        plt.title(f"Carrier Modulated {self.conf.modType} Signal")
        plt.stem(self.symbol_time, self.symbols, 'orange', label="BPSK Symbols")
        plt.plot(modsignal.time, modsignal.signals, '-', label="Carrier modulated signal")
        plt.legend(loc='upper right')
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.grid()

        plt.show()
        

        return modsignal

    def show_signal_plot(self):
        plt.figure(figsize=(15,6))
        elements_to_plot = range(len(self.elements))
        plt.rcParams['font.size'] = 15
        for i in elements_to_plot:
            if self.conf.modType=="BPSK":
                print(len(self.symbols))
                plt.plot(self.time, self.elements[i], '-', label=f"Element {i+1} Signal")

            elif self.conf.modType=="QPSK":
                plt.subplot(2, 1, 1)
                plt.stem(self.symbol_time, np.real(self.symbols), 'orange')
                plt.plot(self.time, np.real(self.elements[i]), '-b')
                plt.title("Real Part (In-Phase / I)")
                plt.grid()

                plt.subplot(2, 1, 2)
                plt.stem(self.symbol_time, np.imag(self.symbols), 'orange')
                plt.plot(self.time, np.imag(self.elements[i]), '-g')
                plt.title("Imaginary Part (Quadrature / Q)")
                plt.xlabel("Sample Index")
                plt.tight_layout()
                plt.grid()

            else:
                raise ValueError("Unknown modulation scheme. Cannot plot!")
        
        plt.title(f"Received Signals by N=2 Elements")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.legend(loc="upper right")
        plt.grid()


        #Symbol Plot

        #Spectum plot
        plt.figure(figsize=(15,6))
        plt.title(f"{self.conf.modType} with RRC Pulse Shaping")
        specan = SpecAn().calculate_psd(
            self.conf.fs,
            self.symbol_waveform,
            )
    
        plt.plot(specan.f/1e6, specan.Pxx_dB, label="PSD")
        #plt.ylim(-7,1)
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
        plt.ylabel("Normalised PSD (dB)")
        plt.xlabel("Frequency (MHz)")
        plt.axvline(x=-(1+self.conf.alpha)*self.conf.rb/1e6/2, color='red', linestyle='--', linewidth=1.5)
        plt.axvline(x=(1+self.conf.alpha)*self.conf.rb/1e6/2, color='red', linestyle='--', linewidth=1.5)
        plt.grid()
        
        #constellation diagram
        plt.figure()
        plt.title(f"Constellation Diagram of Tx Signal on Element=1")
        plt.scatter(
            np.real(self.symbols),
            np.imag(self.symbols),
            color="blue",
            alpha=0.5,
            s=30,
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
    Rb = 30e6
    Tb = 1/Rb
    symbols = 2000
    totaltime = symbols*Tb

    config = CDMcfg(
        rb = Rb,
        fcar= 200e6,
        lc = 4,
        sps_chip= 2,
        alpha = 0.5,
        simtime = totaltime,
        no_elem = 1,
        modType="BPSK"
    )
    
    signal = DiscreteSignal(config)

    #==================================
    #PLOTRESULTS
    #=================================
    signal.show_signal_plot()