from enum import Enum, auto
from dataclasses import replace
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.signal import welch

import utils as u
from components import CDMx, SpecAn, Beamformer, Filter, Mixer, Doppler
from config import CDMcfg
from signals import DiscreteSignal
from models import ResampledSignals


class PlotType(Enum):
    signal = auto()
    spread = auto()
    summed = auto()
    ddc = auto()
    recovered = auto()
    bf = auto()

    @classmethod
    def all(cls) -> "set[PlotType]":
        return {t for t in cls}


class Receiver():
    def __init__(self, config, plot: set[PlotType] = set()):
        self.conf  = config 
        self.plot = plot
        
    def setup(self, signal):
        self.cdmx = CDMx(self.conf)                     #generate code division miultiplexer
        self.bf = Beamformer()                          #generate beamformer
        self.mixer = Mixer(self.conf)                   #generate mixers, superheterodyne and quadrature
        self.filter = Filter(self.conf)                 #generate filter, butterworth bandpass filter 

        self.codes = self.cdmx.generate_codes()         #spreading codes

        #recieve the signal
        self.rxsignals = copy.deepcopy(signal.elements) #element signals
        self.fs_rx = signal.fs                          #sampling frequency
        self.rx_time = signal.time                      #element signal time axis
        self.element_delays = signal.psi                #element signal spatial delay 
        
        if PlotType.signal in self.plot:
            self.signal_plot(
                "Received signal waveform",
                self.rx_time,
                self.rxsignals[0],
                self.fs_rx
            )

    def assemble(self):
        if not self.conf.CALIBRATE: 
            #===================================
            #   D/C TO IF OR BB PER ELEMENT     |
            #   Fsignal change not Fs_signal!!  |
            #===================================     
            
            #Baseband: baseband [array] Signal for each element 
            #Time = self.rx_time
            #Signal Frequency =  self.conf.fIF
            #Sampling Frequency = fs_rx
            if self.conf.SPREADING_STAGE=="IF":
                #Band-pass filter as Image reject filter
                fband = np.array([self.conf.fc - self.conf.b/2, self.conf.fc + self.conf.b/2])
                self.rxsignals = self.filter.butterworth(
                    cutoff = fband,
                    fs = self.fs_rx,
                    signals =  self.rxsignals,
                    filter_type="bandpass"
                )
                #Carrier enabled implies spreading at IF 
                flo = self.conf.fc - self.conf.fIF 
                self.baseband = self.mixer.superhet(
                    self.rxsignals,
                    fs_signal = self.fs_rx, 
                    flo = flo 
                )
            elif self.conf.SPREADING_STAGE=="BB":
                #No carrier implies spreading at BB
                flo = self.conf.fc 
                self.baseband = self.mixer.superhet(
                    self.rxsignals,
                    fs_signal = self.fs_rx, 
                    flo = flo 
                )

            if PlotType.signal in self.plot:
                self.signal_plot(
                    "After D/C at Element=1",
                    self.baseband.time, self.baseband.signals[0],
                    self.baseband.fs
                )
            
 
            #===============================
            #       LPF PER ELEMENT        |
            #   TO REMOVE SUPERHET IMAGE   |
            #===============================

            #Filtered Baseband: filtered_bb [array], Signal for each element 
            #Time = self.rx_time
            #Signal Frequency =  self.conf.fIF w/o image 
            #Sampling Frequency = fs_rx

            if self.conf.SPREADING_STAGE=="IF":
                flow = self.conf.fIF + self.conf.b/2
                self.filtered_bb = self.filter.butterworth(
                    cutoff = flow,
                    fs = self.fs_rx,
                    signals =  self.baseband.signals,
                    filter_type="lowpass"
                )

            elif self.conf.SPREADING_STAGE=="BB":
                flow = self.conf.b
                self.filtered_bb = self.filter.butterworth(
                    cutoff = flow,
                    fs = self.baseband.fs,
                    signals =  self.baseband.signals,
                    filter_type="lowpass"
                )

            elif self.conf.SPREADING_STAGE=="RF":
                self.filtered_bb = self.baseband.signals

            if PlotType.signal in self.plot:
                self.signal_plot(
                    "After LPF at Element=1",
                    self.baseband.time, self.filtered_bb[0],
                    self.baseband.fs
                )
            

            #============================
            #   SPREADER PER ELEMENT     |
            # CHANGES SAMPLING FREQUENCY |
            #============================

            #Spread Resampled dataclass
            #Spread Baseband: spread_bb.signals, Signal for each element 
            #Spread Time: spread_bb.time, Timestep changed to match timestep of code sequence 
            #Spread Code: spread_bb.codes, Codes equal length to Spread Signal
            #Spread Sampling Frequency: spread_bb.fs, Fs_chip 
            #Signal Frequency = self.conf.fIF * self.conf.lc  , NOT returned 

            if self.conf.SPREADING_STAGE=="IF":
                Bs = self.conf.fIF
            elif self.conf.SPREADING_STAGE=="BB":
                Bs = self.conf.b
            elif self.conf.SPREADING_STAGE=="RF":
                Bs = self.conf.b
            
            self.spread_bb = self.cdmx.spreader(
                signals = self.filtered_bb,
                fs_signal = self.baseband.fs,
                f_signal =  Bs,
                codes=self.codes
            )

            if PlotType.spread in self.plot:
                self.spread_plot(
                    index=0
                )
    
            #====================
            #     COMBINER      |
            #====================

            #Summed Baseband: summed_signal [array]
            #Summed Time: spread_bb.time
            #Summed Sampling Frequency: spread_bb.fs, Fs_chip=Fchip*self.conf.sps_chip  
            #Summed Signal Frequency = Fchip= self.conf.fIF * self.conf.lc  , NOT returned 

            summed_signal = [sum(values) for values in zip(*self.spread_bb.signals)]
            if PlotType.summed in self.plot: 
                self.sum_plot(
                    self.spread_bb.time, summed_signal
                    )

    
            #=====================q================
            #   DIGITAL DESPREAD  PER ELEMENT     |
            #   NOT RESAMPLED BUT MF DELAY TRUNC  |
            #=====================================

            self.recovered = self.cdmx.recover(
                signal=summed_signal,
                fs_signal = self.spread_bb.fs,
                f_signal = Bs,
                codes=self.codes
            )

            if PlotType.recovered in self.plot:
                self.recover_plot(
                    index=0
                )        
            
        else: 
            #WHEN CALIBRATING: ONLY PASSING THROUGH MATCHED FILTER
            self.recovered = self.cdmx.recover(
                self.rxsignals,
                [],
                fs_signal=self.fs_rx,
                f_signal=self.conf.fb
                )

            if PlotType.recovered in self.plot:
                self.recover_plot(
                    0
                )

        #=============================
        #     DIGITAL BEAMFORMING    |
        #=============================
        if self.conf.BEAMFORM:
            angles = self.estimate_angles(self.recovered.signals)
            psi = [
            i * np.pi * np.sin(angles[i])
            for i in range(0,len(self.recovered.signals))
            ]
            amplitudes = np.array([
                np.sqrt(np.mean(np.abs(s)**2))
                for s in self.recovered.signals
                ])
            amplitudes /= amplitudes[0]

            if PlotType.bf in self.plot:
                self.beampattern_plot(psi, amplitudes)    
                #self.array_plot(angles, amplitudes)
            return self.recovered
        else:
            return self.recovered


    def estimate_angles(self, elements):

        angles = [0]

        for i in range(1, len(elements)):

            # Phase difference between adjacent elements
            phase_difference = np.angle(
                np.vdot(elements[i-1], elements[i])
            )

            # Adjacent elements are separated by d = lambda/2
            theta = np.arcsin(
                -phase_difference / np.pi
            )

            angles.append(theta)

        return np.array(angles)

    def signal_plot(self, 
        title,
        time,
        asignal,
        fsample
        ):

        #WAVEFORM
        plt.rcParams['font.size'] = 20
        plt.figure(figsize=(18,6))
        plt.style.use("classic")
        plt.title(f"{title}")
        plt.plot(time[:1000], asignal[:1000], '-o', label="Element=1")
        plt.legend(loc='upper right')
        plt.grid()
        plt.xlabel("Time (seconds)",fontsize=18)
        plt.ylabel("Amplitude",fontsize=18)
        plt.tight_layout()

    
        #SPECTRUM
        plt.figure(figsize=(20,6))
        plt.rcParams['font.size'] = 20
        plt.style.use("classic")

        plt.title(f"{title}", fontsize=21)
        specan = SpecAn().calculate_psd(
            fsample,
            asignal,
            )
        plt.plot(
            specan.f/1e6,
            specan.Pxx_dB,
            '-b',
            label="Element=1"
        )
        #plt.axvline(x=self.conf.fc/1e6, color='red', linestyle='--', linewidth=1.5, label="Carrier Frequency")

        
        plt.ylabel("Normalized Power (dB)",fontsize=18)
        plt.xlabel("Frequency (MHz)",fontsize=18)
        plt.tight_layout()
        plt.legend()
        plt.grid()
        plt.show()

    def spread_plot(self, 
        index=0
        ):
        #WAVEFORM
        plt.rcParams['font.size'] = 21
        plt.figure(figsize=(20,6))
        plt.style.use("classic")

        if not self.conf.codefam == "ZC":
            plt.plot(self.spread_bb.time[:1000], self.spread_bb.original[index][:1000], '-ob', label=f"Rx signal")
            plt.plot(self.spread_bb.time[:1000], self.spread_bb.signals[index][:1000], '-og', label=f"c1={self.codes[index]}")
        else:
            plt.subplot(2, 1, 1)
            plt.plot(self.spread_bb.time, self.spread_bb.original[index], '-ob', label=f"Rx signal")
            plt.plot(self.spread_bb.time, np.real(self.spread_bb.signals[index]), '-og', label=f"Real c1")
            plt.title("Real Part (In-Phase / I)")
            plt.grid()

            plt.subplot(2, 1, 2)
            plt.plot(self.spread_bb.time, self.spread_bb.original[index], '-ob', label=f"Rx signal")
            plt.plot(self.spread_bb.time, np.imag(self.spread_bb.signals[index]), '-og', label=f"Imag c1")
            plt.title("Imaginary Part (Quadrature / Q)")
    
        plt.title(f"Spread Signal of Element={index+1}", fontsize=20)
        plt.xlabel("Time (seconds)",fontsize=18)
        plt.ylabel("Amplitude",fontsize=18)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.grid()

        #SPECTRUM
        specan = SpecAn().calculate_psd(
            self.spread_bb.fs,
            self.spread_bb.signals[index]
            )

        plt.figure(figsize=(20,6))
        plt.plot(specan.f/1e6, specan.Pxx_dB, '-g', label="Spread Spectrum of Element 1")
        if self.conf.SPREADING_STAGE=="BB":
            plt.axvline(x= self.conf.lc*self.conf.b/1e6/2, color='red', linestyle='--', linewidth=1.5)
            plt.axvline(x=-self.conf.lc*self.conf.b/1e6/2, color='red', linestyle='--', linewidth=1.5, label="Bss=LsxBs")
        elif self.conf.SPREADING_STAGE=="IF":
            plt.axvline(x=-self.conf.lc*self.conf.fIF/1e6, color='red', linestyle='--', linewidth=1.5)
            plt.axvline(x=self.conf.lc*self.conf.fIF/1e6, color='red', linestyle='--', linewidth=1.5)

        plt.xlabel("Frequency (MHz)",fontsize=18)
        plt.ylabel("Normalised Power (dB)",fontsize=18)
        plt.title(f"Spectrum of Element={index+1}", fontsize=20)
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()

    def sum_plot(self, time, signal):

        #WAVEFORM
        plt.rcParams['font.size'] = 21
        plt.figure(figsize=(20,6))
        plt.style.use("classic")

        plt.plot(time[:1000], signal[:1000], '-ob')
        plt.title(f"Combined signals from {self.conf.no_elem} Elements", fontsize=21)
        plt.xlabel("Time (seconds)",fontsize=18)
        plt.ylabel("Amplitude",fontsize=18)
        plt.tight_layout()
        plt.grid()

        
        #SPECTRUM
        plt.figure(figsize=(20,6))
        plt.title(f"Spectrum of the combined signals", fontsize=21)
        specan = SpecAn().calculate_psd(
            self.spread_bb.fs,
            np.array(signal),
            )
        
        plt.plot(
            specan.f/1e6,
            specan.Pxx_dB,
            '-b',
            label="Combined signals"
        )
        plt.axvline(x=self.conf.lc*self.conf.b/1e6/2, color='red', linestyle='--', linewidth=1.5)
        plt.axvline(x=-self.conf.lc*self.conf.b/1e6/2, color='red', linestyle='--', linewidth=1.5,label="Bss=LsxBs")

        plt.ylabel("Normalised Power (dB)",fontsize=18)
        plt.xlabel("Frequency (MHz)",fontsize=18)
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()

    def recover_plot(self,
        index=0
        ):
        plt.rcParams['font.size'] = 20
        plt.figure(figsize=(20,10))
        #lt.title(f"Recovered Signal of Element={index+1}",fontsize=21)

        if self.conf.modType=="BPSK":
            plt.plot(signal.symbol_wf_time, signal.symbol_waveform, '-b', label=f"Tx symbol waveform")
            plt.plot(self.recovered.time, np.imag(self.recovered.signals[index]), '-g', linewidth='2', label="Recovered signal")
            #plt.stem(self.recovered.time[::self.conf.sps_symbol][1:], self.recovered.signals[0][::self.conf.sps_symbol][1:], label="symbols")
        
        elif self.conf.modType=="QPSK":
            plt.style.use("classic")
            print(index)
            plt.subplot(2, 1, 1)
            plt.plot(self.recovered.time, np.real(self.recovered.signals[index]), '-g', linewidth='3', label="Recovered waveform")
            plt.plot(signal.symbol_wf_time, np.real(signal.symbol_waveform), '-b', label="Tx symbol waveform")
            plt.ylabel("Amplitude",fontsize=18)
            plt.legend(loc='upper right')
            plt.title("Real Part (In-Phase / I)", fontsize=20)
            plt.grid()

            plt.subplot(2, 1, 2)
            plt.plot(self.recovered.time, np.imag(self.recovered.signals[index]), '-g', linewidth='3', label="Recovered waveform")
            plt.plot(signal.symbol_wf_time, np.imag(signal.symbol_waveform), '-b', label="Tx symbol waveform ")
            plt.ylabel("Amplitude",fontsize=18)
            plt.title("Imaginary Part (Quadrature / Q)", fontsize=20)
            plt.legend(loc='upper right')

        plt.grid()
        plt.xlabel("Time (seconds)", fontsize=18)
        plt.tight_layout()

        #SPECTRUM
        plt.figure(figsize=(20,6))
        specan = SpecAn().calculate_psd(
            self.conf.fs,
            signal.symbol_waveform
                        )
        plt.plot(specan.f/1e6, specan.Pxx_dB, '-r', label="Original spectrum")
        specan = SpecAn().calculate_psd(
            self.recovered.fs,
            self.recovered.signals[index]
                        )
        plt.plot(specan.f/1e6, specan.Pxx_dB, '-b', label="Recovered spectrum")
        plt.axvline(x=self.conf.b/1e6/2, color='red', linestyle='--', linewidth=1.5)
        plt.axvline(x=-self.conf.b/1e6/2, color='red', linestyle='--', linewidth=1.5, label="Original Rx bandwidth")
        plt.legend()
        plt.xlabel("Frequency (MHz)", fontsize=18)
        plt.ylabel("Normalised Power (dB)",fontsize=18)
        plt.title(f"Spectrum of Recovered Element{index+1}",fontsize=21)
        plt.grid()
        plt.show()

    def bf_plot(self,
        tbefore,
        before_signal,
        tafter,
        after_signal,
        fs,
        index=0
        ):
        plt.rcParams['font.size'] = 15
        plt.figure(figsize=(15,7))
        plt.title(f"Combined and Phase aligned Signal")
        plt.plot(tbefore, before_signal[index], '--b',alpha=0.5, label=f"Rx signal + Interference")
        plt.plot(tafter, after_signal, '-g', linewidth=3, label="Beamformed Signal")

        #INTERFERENCE PLOT ONLY
        if self.conf.INTERFERENCE:
            tx_clean = np.load("transmit_clean.npy")
            plt.plot(tbefore, tx_clean, '-b', linewidth=2, label=f"Rx Signal")

        plt.legend(loc='upper right')
        plt.grid()
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.show()

    def beampattern_plot(self,psi_angles, A, N_fft=4096):


        fig, ax  = plt.subplots(
            subplot_kw={'projection': 'polar'},
            figsize=(18, 18)
        )
        plt.rcParams['font.size'] = 21
        plt.style.use("classic")
        # -------------------------
        # Original
        # -------------------------
        Nr = self.conf.no_elem
        w = np.exp(1j*np.array(signal.psi)) # conventional beamformer weights
        w_padded = np.concatenate((w, np.zeros(N_fft - Nr))) # zero pad to N_fft elements to get more resolution in the FFT
        w_fft_dB = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(w_padded)))**2) # magnitude of fft in dB
        w_fft_dB -= np.max(w_fft_dB) # normalize to 0 dB at peak

        # Map the FFT bins to angles in radians
        theta_bins = np.arcsin(np.linspace(-1, 1, N_fft)) # in radians

        #Full circle
        theta_bins = np.concatenate([
            theta_bins,
            np.pi - theta_bins[::-1]
            ])
        w_fft_dB  = np.concatenate([
            w_fft_dB,
            w_fft_dB[::-1]
            ])

        # find max so we can add it to plot
        theta_max = theta_bins[np.argmax(w_fft_dB)]

        ax.plot(theta_bins, w_fft_dB, 'g', linewidth=2, label='Original') # MAKE SURE TO USE RADIAN FOR POLAR
        AoA = np.round(theta_max * 180 / np.pi)
        ax.plot([theta_max], [np.max(w_fft_dB)],'go', markersize=10, label=f"AoA={AoA}")
        ax.set_theta_zero_location('N') # make 0 degrees point up
        ax.set_theta_direction(-1) # increase clockwise
        ax.set_rlabel_position(55)  # Move grid labels away from other labels
        #ax.set_thetamin(-90) # only show top half
        #ax.set_thetamax(90)
        ax.set_ylim([-30, 1]) # because there's no noise, only go down 30 dB
        ax.tick_params(axis="both", labelsize=21)
        # -------------------------
        # Recovered
        # -------------------------

        Nr = self.conf.no_elem
        w = A*np.exp(1j*np.array(psi_angles)) # conventional beamformer weights
        w_padded = np.concatenate((w, np.zeros(N_fft - Nr))) # zero pad to N_fft elements to get more resolution in the FFT
        w_fft_dB = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(w_padded)))**2) # magnitude of fft in dB
        w_fft_dB -= np.max(w_fft_dB) # normalize to 0 dB at peak

        # Map the FFT bins to angles in radians
        theta_bins = np.arcsin(np.linspace(-1, 1, N_fft)) # in radians

        #Full circle
        theta_bins = np.concatenate([
            theta_bins,
            np.pi - theta_bins[::-1]
            ])
        w_fft_dB  = np.concatenate([
            w_fft_dB,
            w_fft_dB[::-1]
            ])

        # find max so we can add it to plot
        theta_max = theta_bins[np.argmax(w_fft_dB)]

        ax.plot(theta_bins, w_fft_dB,'r', linewidth=2,label='Recovered') # MAKE SURE TO USE RADIAN FOR POLAR
        AoA = np.round(theta_max * 180 / np.pi)
        ax.plot([theta_max], [np.max(w_fft_dB)],'bo', markersize=10, label=f"AoA={AoA}")

        fig.legend(
        loc='lower center',
        ncol=2,
        fontsize=24
                )
        fig.tight_layout()
        plt.show()
   
    def array_plot(self, angles, A):
        rx_angles = np.rad2deg(angles)
        rx_dB = 20 * np.log10(A)

        tx_psi = signal.psi
        tx_angles = np.array([
            0 if i == 0 else np.arcsin(tx_psi[i] / (i * np.pi))
            for i in range(len(tx_psi))
            ])
        tx_angles = np.rad2deg(tx_angles)
        tx_dB =  np.zeros(len(rx_dB))
        a_error =  rx_dB - tx_dB

        plt.rcParams['font.size'] = 21
        plt.figure(figsize=(8,6))
        plt.style.use("classic")

        plt.plot(rx_dB, rx_angles, '-o', label="Reco")


        plt.xlabel("Angle of Arrival (degrees)",fontsize=18)
        plt.ylabel("Amplitude Loss (dB)",fontsize=18)
        plt.tight_layout()
        plt.grid()
        plt.legend()
        plt.show()

        return fd_est

if __name__ == "__main__":
    """
    Ts: symbol period = e.g. 10 seconds
    samples_symbol: samples per symbol 
    Ns: no of symbols = e.g. 5 unique symbols which lasts 10 seconds each post filter 
    modType: modulation type = BPSK 
    no_elem: number of array signals t 
    """

    # ==============================
    # PARAMETERS 
    # ==============================
    Rb = 30e6
    Tb = 1/Rb
    symbols = 1000
    totaltime = symbols*Tb
    lc = 8
    config = CDMcfg(
        fb = Rb,
        lc = lc, 
        sps_chip= 2,
        alpha = 0.5,
        simtime = totaltime,
        no_elem = 1,
        codefam="WH",
        modType="BPSK",
        codeids=np.array([1,3,4,6]),
        SPREADING_STAGE = "BB"           #Location of the spreading [IF or BB]
        )
    #create signal 
    signal = DiscreteSignal(config)
    signal.show_signal_plot()

    #initiate and assemble receiver 
    rx = Receiver(config, {PlotType.recovered})
    rx.setup(signal)
    rx.assemble()

