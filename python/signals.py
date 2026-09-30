"""
Simulate effects of channel
- AWGN
- Doppler shift 
"""
import random
from math import pi
from dataclasses import replace
import commpy.channels as ch  # type:ignore[import-untyped]
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
from math import sin 

from components import Filter, SpecAn, Mixer, Doppler
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
            #modulation will change time array, fs_mixer, carrier_time, carrier_signald sampling frequency
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
        no_symbols = int(self.conf.simtime*self.conf.fb)

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
        #self.txsymbols = np.random.choice(constellation, no_symbols) 
    
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
        #time of series which can be overwritte by carrier modulated
        self.time =  np.arange(len(signal))/ self.conf.fs 

        #time of symbol sampler [+/-1]
        self.symbol_time = self.time[::self.conf.sps_symbol][1:]

        #time of full symbol pulses
        self.symbol_wf_time = np.arange(len(self.symbol_waveform))/self.conf.fs

        
    def create_elementsignals(self, signal):
        #apply phase rotation as delay is too small to be seen as lagged samples. Why? 
        #psi = pi 
        #applying d <= lambda/2 and lambda is for 10GHz (lambda=30mm) for Xband operations
        c = 3*10**8
        theta = np.deg2rad(90-self.conf.elevation)
        #each time this is recreated
        self.elements = []
        self.psi = []
        self.symbols_elements = [self.symbols]
        for i in range(0,self.conf.no_elem):
            psi_i = i*pi*sin(theta)
            signalNs = signal*np.exp(-1j*psi_i)
            symbolNs = self.symbol_waveform*np.exp(-1j*psi_i)
            symbolNs = symbolNs[::self.conf.sps_symbol][1:]
            self.symbols_elements.append(symbolNs)
            self.elements.append(signalNs)
            self.psi.append(psi_i)
        return self.elements 

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
            flo = self.conf.fc + self.conf.fd
        )
        """
        #bandpass filter to remove harmonics caused by resampling 
        flow = self.conf.fc-self.conf.fb
        fhigh = self.conf.fc+self.conf.fb
        filtered_modsignal = self.filters.butterworth(
            cutoff = np.array([flow,fhigh]),
            fs = modsignal.fs,
            signals = modsignal.signals,
            filter_type="bandpass",
        )
        
        modsignal.signals = np.real(filtered_modsignal)
        """
        modsignal.signals = np.real(modsignal.signals)
        
        """
        #PLOT FOR AFTER THE MODULATION
        plt.figure(figsize=(15,6))
        plt.rcParams['font.size'] = 15

        plt.title(f"Carrier Modulated {self.conf.modType} Signal")
        specan = SpecAn().calculate_psd(
            modsignal.fs,
            modsignal.signals,
            )
        plt.plot(specan.f/1e6, specan.Pxx_dB, label="PSD")
        plt.axvline(x=self.conf.fc/1e6, color='red', linestyle='--', linewidth=1.5, label="Carrier Frequency")
        plt.xlim(-2*self.conf.fc/1e6,2*self.conf.fc/1e6)
        plt.ylabel("Normalised Power (dB)")
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
        """

        return modsignal

    def show_signal_plot(self):
        plt.figure(figsize=(15,10))
        plt.style.use("classic")
        elements_to_plot = range(len(self.elements))
        plt.rcParams['font.size'] = 15
        for i in elements_to_plot:
            if self.conf.modType=="BPSK":
                plt.plot(self.time[:1000], self.elements[i][:1000], '-o', label=f"Element {i+1}")

            elif self.conf.modType=="QPSK":
                plt.subplot(2, 1, 1)
                plt.stem(self.symbol_time[:11], np.real(self.symbols)[:11], 'orange', label="Sampled Symbols")
                plt.plot(self.time[:11*8], np.real(self.elements[i])[:11*8], '-b', label="RRC filtered symbols w/o group delay")
                plt.grid()
                plt.legend()
                plt.ylabel("Amplitude", fontsize=18)
                plt.title("Real Part (In-Phase / I)")

                plt.subplot(2, 1, 2)
                plt.stem(self.symbol_time[:11], np.imag(self.symbols)[:11], 'orange', label="Sampled Symbols")
                plt.plot(self.time[:11*8], np.imag(self.elements[i])[:11*8], '-g', label="RRC filtered symbols w/o group delay")
                plt.title("Imaginary Part (Quadrature / Q)")
            else:
                raise ValueError("Unknown modulation scheme. Cannot plot!")
        
        #plt.title(f"N=2 Element Received Signal Waveform")
        plt.xlabel("Time (seconds)", fontsize=18)
        plt.ylabel("Amplitude", fontsize=18)
        plt.tight_layout()
        plt.legend(loc="upper right")
        plt.grid()


        #Symbol Plot

        #Spectum plot
        plt.figure(figsize=(20,6))
        plt.rcParams['font.size'] = 15
        plt.style.use("classic")

        plt.title(f"{self.conf.modType} with RRC Pulse Shaping")
        specan = SpecAn().calculate_psd(
            self.conf.fs,
            self.symbol_waveform,
            )
    
        plt.plot(specan.f/1e6, specan.Pxx_dB, label="Spectrum of BPSK waveform")
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylabel("Normalised Power (dB)",fontsize=18)
        plt.xlabel("Frequency (MHz)",fontsize=18)
        plt.axvline(x=-(1+self.conf.alpha)*self.conf.fb/1e6/2, color='red', linestyle='--', linewidth=1.5, label="Bs=(1+α)Rb")
        plt.axvline(x=(1+self.conf.alpha)*self.conf.fb/1e6/2, color='red', linestyle='--', linewidth=1.5)
        plt.legend()
        plt.tight_layout()
        plt.grid()
        
        #constellation diagram
        plt.figure(figsize=(10,8))
        plt.style.use("classic")
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


    def plot_beampattern(self, N_fft=4096):
        Nr = self.conf.no_elem
        w = np.exp(1j*np.array(self.psi)) # conventional beamformer

        w_padded = np.concatenate((w, np.zeros(N_fft - Nr))) # zero pad to N_fft elements to get more resolution in the FFT
        w_fft_dB = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(w_padded)))**2) # magnitude of fft in dB
        w_fft_dB -= np.max(w_fft_dB) # normalize to 0 dB at peak

        # Map the FFT bins to angles in radians
        theta_bins = np.arcsin(np.linspace(-1, 1, N_fft)) # in radians

        # find max so we can add it to plot
        theta_max = theta_bins[np.argmax(w_fft_dB)]

        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.plot(theta_bins, w_fft_dB) # MAKE SURE TO USE RADIAN FOR POLAR
        ax.plot([theta_max], [np.max(w_fft_dB)],'ro')
        ax.text(theta_max - 0.1, np.max(w_fft_dB) - 4, np.round(theta_max * 180 / np.pi))
        ax.set_theta_zero_location('N') # make 0 degrees point up
        ax.set_theta_direction(-1) # increase clockwise
        ax.set_rlabel_position(55)  # Move grid labels away from other labels
        ax.set_thetamin(-90) # only show top half
        ax.set_thetamax(90)
        ax.set_ylim([-30, 1]) # because there's no noise, only go down 30 dB
        plt.show()

    def estimate_fd(self):
        f, P = welch(
            self.elements[0],
            self.fs
        )
        # Select ONLY the positive-frequency signal around fc
        bw = self.conf.b
        guard = 100e6
        mask = (
            (f >= self.conf.fc - bw/2 - guard) &
            (f <= self.conf.fc + bw/2 + guard )
        )

        f_signal = f[mask]
        P_signal = P[mask]


        # Spectral centroid
        f_est = np.sum(f_signal * P_signal) / np.sum(P_signal)
        plt.plot(f_signal, P_signal)


        # Doppler relative to nominal carrier
        fd_est = f_est - self.conf.fc

        return fd_est

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
    symbols = 1000
    totaltime = symbols*Tb

    config = CDMcfg(
        fb = Rb,
        lc = 4,
        sps_chip= 2,
        alpha = 0.5,
        simtime = totaltime,
        no_elem = 4,
        modType="BPSK"
    )
    
    signal = DiscreteSignal(config)

    fref = signal.estimate_fd()

    #Calculation for Doppler 
    fds, els, time = Doppler().calculate_shifts()
    fd_list = []

    for fd, el in zip(fds, els):

        config_fd = replace(
            config,
            fd=fd,
            elevation=el
        )

        signal = DiscreteSignal(config_fd)

        fd_list.append(
            signal.estimate_fd()
        )

    fd_estimated = np.asarray(fd_list) - fref



    #==================================
    #PLOTRESULTS
    #=================================

    plt.figure(figsize=(10, 6))

    plt.plot(
        np.asarray(fds) / 1e3,
        els,
        "k--",
        linewidth=2,
        label=r"Injected $f_D$"
    )

    plt.plot(
        fd_estimated / 1e3,
        els,
        "r.",
        markersize=5,
        label=r"Estimated $\hat{f}_D$"
    )

    plt.xlabel(r"Doppler Shift $f_D$ (kHz)")
    plt.ylabel("Elevation / Beam Angle (deg)")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
    #signal.show_signal_plot()
    #signal.plot_beampattern()