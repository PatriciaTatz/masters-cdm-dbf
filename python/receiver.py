from enum import Enum, auto

import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.signal import resample_poly, upfirdn

import utils as u
from components import CDMx, SpecAn, Beamformer, Filter, Mixer
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

        self.codes = self.cdmx.generate_codes()         #spr4eading codes

        #recieve the signal
        self.rxsignals = copy.deepcopy(signal.elements) #element signals
        self.fs_rx = signal.fs                          #sampling frequency
        self.rx_time = signal.time                      #element signal time axis
        self.element_delays = signal.psi                #element signal spatial delay 
        
        if PlotType.signal in self.plot:
            self.signal_plot(
                "Rx Signal at Element 1",
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
                #Carrier enabled implies spreading at IF 
                print("fc=",self.conf.fc)
                flo = self.conf.fc - self.conf.fIF 
                print("flo=",flo)
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
                flow = (self.conf.fIF)
            elif self.conf.SPREADING_STAGE=="BB":
                flow = self.conf.b


            self.filtered_bb = self.filter.butterworth(
                cutoff = flow,
                fs = self.fs_rx,
                signals =  self.baseband.signals,
                filter_type="lowpass"
            )
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
                
                self.spread_bb = self.cdmx.spreader(
                    signals=self.filtered_bb,
                    fs_signal = self.baseband.fs,
                    f_signal = self.conf.b,
                    codes=self.codes
                )
            elif self.conf.SPREADING_STAGE=="BB":
                self.spread_bb = self.cdmx.spreader(
                    signals=self.filtered_bb,
                    fs_signal = self.baseband.fs,
                    f_signal = self.conf.b,
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

            #=====================================
            #        DIGITAL DOWNCONVERTER        |
            # RESAMPLING HAPPENS HERE             |
            #=====================================

            #DDC Resampled dataclass
            #DDC Baseband: ddc.signals, but only ONE combined signal. 
            #DDC Time: ddc.time, Timestep changed to match timestep of code sequence 
            #DDC Sampling Frequency: ddc.fs = 3*flo   to match mixer 
            #DDC Signal Frequency = spread bandwidth (self.conf.b * self.conf.lc) centered at 0Hz

            if self.conf.SPREADING_STAGE=="IF":
                ddc = self.mixer.quadrature(
                            np.array(summed_signal),
                            fs_signal=self.spread_bb.fs,
                            flo=self.conf.fIF
                        )

                if PlotType.ddc in self.plot:
                    self.signal_plot(
                        "After DDC at Element=1",
                        ddc.time, ddc.signals,
                        ddc.fs
                    )
                
            #=====================q================
            #   DIGITAL DESPREAD  PER ELEMENT     |
            #   NOT RESAMPLED BUT MF DELAY TRUNC  |
            #=====================================

            """
            recovered = self.cdmx.recover(
                ddc.signals,
                self.spread_bb.codes,
                ddc.fs
                )
            """
            if self.conf.SPREADING_STAGE=="IF":
                self.recovered = self.cdmx.recover(
                    signal=ddc.signals,
                    fs_signal = ddc.fs,
                    f_signal = self.conf.b,
                    codes=self.codes

                )
                if PlotType.recovered in self.plot:
                    self.recover_plot(
                        index=0
                    )
            elif self.conf.SPREADING_STAGE=="BB":
                self.recovered = self.cdmx.recover(
                    signal=summed_signal,
                    fs_signal = self.spread_bb.fs,
                    f_signal = self.conf.b,
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

            if PlotType.signal in self.plot:
                self.recover_plot(
                    self.rx_time, self.rxsignals, 
                    self.rx_time, self.recovered,
                    self.recovered.fs,
                    0
                )

        #=============================
        #     DIGITAL BEAMFORMING    |
        #=============================
        if self.conf.BEAMFORM:
            bf = self.bf.align_combine_signals(
                self.element_delays,
                self.recovered.signals
            )

            if PlotType.bf in self.plot:
                self.bf_plot(
                    self.rx_time, filtered_bb, 
                    self.recovered.time, bf,
                    0
                )  
            return bf
        else:
            return self.recovered

    def signal_plot(self, 
        title,
        time,
        asignal,
        fsample
        ):
        plt.rcParams['font.size'] = 15
        plt.figure(figsize=(12,7))
        plt.plot(time[:1000], asignal[:1000], '-o', label="RRC Filtered Signal")
        plt.axvline(x=1/self.conf.fb, color='red', linestyle='--', linewidth=1.5)
        plt.axvline(x=2/self.conf.fb, color='red', linestyle='--', linewidth=1.5)
        plt.axvline(x=3/self.conf.fb, color='red', linestyle='--', linewidth=1.5)

        plt.title(f"{title}")

        plt.legend(loc='upper right')
        plt.grid()
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
    
        plt.figure(figsize=(15,6))
        plt.title(f"{title}")
        specan = SpecAn().calculate_psd(
            fsample,
            asignal,
            )
        plt.plot(
            specan.f/1e6,
            specan.Pxx_dB,
            '-b' 
        )
        
        plt.ylabel("Normalized Power (dB)")
        plt.xlabel("Frequency (MHz)")
        plt.grid()
        plt.show()

    def spread_plot(self, 
        index=0
        ):

        plt.rcParams['font.size'] = 15
        plt.figure(figsize=(15,7))
        plt.title(f"Spread Signals of Element={index+1}")
        plt.plot(self.spread_bb.time, self.spread_bb.original[index], '-ob', label=f"Rx signal")
        plt.plot(self.spread_bb.time, self.spread_bb.signals[index], '-og', label=f"c1={self.codes[index]}")

        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.legend(loc='upper right')
        plt.grid()

        specan = SpecAn().calculate_psd(
            self.spread_bb.fs,
            self.spread_bb.signals[index]
            )

        plt.figure(figsize=(15,8))
        plt.plot(specan.f/1e6, specan.Pxx_dB, '-g')
        plt.axvline(x=self.conf.lc*self.conf.b/1e6/2, color='red', linestyle='--', linewidth=1.5)
        plt.axvline(x=-self.conf.lc*self.conf.b/1e6/2, color='red', linestyle='--', linewidth=1.5)

        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Normalised Power (dB)")
        plt.title(f"Spread Signals of Element={index}")
        plt.grid()
        plt.show()

    def sum_plot(self, time, signal):
        plt.rcParams['font.size'] = 15
        plt.figure(figsize=(15,7))
        plt.plot(time, signal, '-ob')
        plt.title(f"Combined Signal from {self.conf.no_elem} Elements")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.grid()

        plt.figure(figsize=(15,8))
        plt.title(f"The Combined Signal")
        specan = SpecAn().calculate_psd(
            self.spread_bb.fs,
            np.array(signal),
            )
        
        """
        plt.semilogy(
            specan.f,
            specan.Pxx_dB,
            color="blue",
            label="PSD")
        """
        plt.plot(
            specan.f/1e6,
            specan.Pxx_dB,
            '-b' 
        )
        plt.axvline(x=self.conf.lc*self.conf.b/1e6/2, color='red', linestyle='--', linewidth=1.5)
        plt.axvline(x=-self.conf.lc*self.conf.b/1e6/2, color='red', linestyle='--', linewidth=1.5)

        plt.ylabel("Normalised Power (dB)")
        plt.xlabel("Frequency (MHz)")
        plt.grid()

        plt.show()

    def recover_plot(self,
        index=0
        ):
        plt.rcParams['font.size'] = 15
        plt.figure(figsize=(15,7))
        #plt.title(f"Recovered Signal at Element Path={index+1}")
        if self.conf.modType=="BPSK":
            plt.title(f"Despread Signal of Element={index+1}")
            plt.plot(signal.symbol_wf_time, signal.symbol_waveform, '--b', label=f"Tx symbol waveform")
            plt.plot(self.recovered.time, self.recovered.signals[index], '-g', linewidth='2', label="Recovered signal")
        elif self.conf.modType=="QPSK":
            plt.subplot(2, 1, 1)
            plt.plot(self.recovered.time, np.real(self.recovered.signals[index]), '-g', linewidth='3', label="Recovered signal")
            plt.plot(signal.symbol_wf_time, np.real(signal.symbol_waveform), '-b', label="Tx symbol waveform")

            plt.title("Real Part (In-Phase / I)")
            plt.grid()

            plt.subplot(2, 1, 2)
            plt.plot(self.recovered.time, np.imag(self.recovered.signals[index]), '-g', linewidth='3', label="Recovered signal")
            plt.plot(signal.symbol_wf_time, np.imag(signal.symbol_waveform), '-b', label="Tx symbol waveform ")

            plt.title("Imaginary Part (Quadrature / Q)")
            plt.xlabel("Sample Index")
        
        plt.legend(loc='upper right')
        plt.grid()
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")

        specan = SpecAn().calculate_psd(
            self.recovered.fs,
            self.recovered.signals[index]
                        )


        plt.figure(figsize=(15,6))
        plt.plot(specan.f/1e6, specan.Pxx_dB, '-b', label="PSD")
        plt.axvline(x=self.conf.b/1e6/2, color='red', linestyle='--', linewidth=1.5)
        plt.axvline(x=-self.conf.b/1e6/2, color='red', linestyle='--', linewidth=1.5)

        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Normalised Power (dB)")
        plt.title(f"Recovered Signal at Element Path={index+1}")
        
        """
        plt.text(
            0.02, 0.95,
            f"Est. BW={specan.bw/1e6:.2f}MHz",
            transform=plt.gca().transAxes,
            ha='left',
            va='top',
            bbox=dict(facecolor='white', edgecolor='black')
        )
        """
        plt.ylabel("Power (1/Hz)")
        plt.xlabel("Frequency (MHz)")
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
    symbols = 50
    totaltime = symbols*Tb
    lc = 16
    config = CDMcfg(
        fb = Rb,
        lc = lc, 
        sps_chip= 2,
        alpha = 0.5,
        simtime = totaltime,
        no_elem = 1,
        codefam="WH",
        modType="QPSK",
        codeids=np.array([1,2]),
        SPREADING_STAGE = "IF"           #Location of the spreading [IF or BB]

        )
    #create signal 
    signal = DiscreteSignal(config)
    #signal.show_signal_plot()

    #initiate and assemble receiver 
    rx = Receiver(config, {PlotType.recovered})
    rx.setup(signal)
    rx.assemble()