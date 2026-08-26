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
        self.rx_fs = signal.fs                          #sampling frequency
        self.rx_time = signal.time                      #element signal time axis
        self.element_delays = signal.psi                #element signal spatial delay 
        
        if PlotType.signal in self.plot:
            self.signal_plot(
                "Rx Signal at Element 1",
                self.rx_time,
                self.rxsignals[0],
                self.rx_fs,
            )
          
    def assemble(self):
        if not self.conf.CALIBRATE: 
            #==============================
            #        D/C PER ELEMENT     |
            #==============================            
            baseband = self.mixer.superhet(
                self.rxsignals,
                self.rx_fs
            )

            if PlotType.signal in self.plot:
                self.signal_plot(
                    "After D/C at Element=1",
                    self.rx_time, baseband[0],
                    self.rx_fs
                )
            #===============================
            #       LPF PER ELEMENT        |
            #   TO REMOVE SUPERHET IMAGE   |
            #===============================
            flow = self.conf.fbb - self.conf.rb
            fhigh = self.conf.fbb + self.conf.rb
            self.filtered_bb = self.filter.butterworth(
                cutoff = np.array([flow,fhigh]),
                fs = self.rx_fs,
                signals =  baseband,
                filter_type="bandpass"
            )
            
            if PlotType.signal in self.plot:
                self.signal_plot(
                    "After LPF at Element=1",
                    self.rx_time, self.filtered_bb[0],
                    self.rx_fs
                )
            #============================
            #   SPREADER PER ELEMENT     |
            # CHANGES SAMPLING FREQUENCY |
            #============================
            #time series goes from FROM RX_TIME -> SPREAD_BB.TIME
            self.spread_bb = self.cdmx.spreader(
                signals=self.filtered_bb,
                fs_signal= self.rx_fs,
                codes=self.codes
            )

            if PlotType.spread in self.plot:
                self.spread_plot(
                    index=0
                )
    
            #====================
            #     COMBINER      |
            #====================
            summed_signal = [sum(values) for values in zip(*self.spread_bb.signals)]
            if PlotType.summed in self.plot: 
                self.sum_plot(spread_bb.time, summed_signal)
            
            if PlotType.summed in self.plot:
                self.spread_plot(
                    self.rx_time, filtered_bb,  
                    self.spread_bb.time, [np.array(summed_signal)], 
                    self.spread_bb.fs
                )

            #=====================================
            #   DIGITAL DESPREAD  PER ELEMENT     |
            #   NOT RESAMPLED BUT MF DELAY TRUNC  |
            #=====================================
            recovered = self.cdmx.recover(
                summed_signal,
                self.spread_bb.codes,
                self.spread_bb.fs
                )
            
            if PlotType.recovered in self.plot:
                self.recover_plot(
                    self.rx_time, filtered_bb, 
                    recovered.time, recovered.signals,
                    self.spread_bb.fs,
                    0
                )
            
        else: 
            #WHEN CALIBRATING: ONLY PASSING THROUGH MATCHED FILTER
            recovered = self.cdmx.recover(
                filtered_signals,
                np.zeros(self.conf.no_elem),
                self.conf.fs
                )
            if PlotType.recovered in self.plot:
                self.recover_plot(
                    self.time, filtered_bb, 
                    recovered.time, recovered.signals,
                    spread_bb.fs,
                    0
                )

        if self.conf.BEAMFORM:
        #=============================
        #     DIGITAL BEAMFORMING    |
        #=============================
            bf = self.bf.align_combine_signals(
                self.element_delays,
                recovered.signals
            )

            if PlotType.bf in self.plot:
                self.bf_plot(
                    self.rx_time, filtered_bb, 
                    recovered.time, bf,
                    0
                )  
            return ResampledSignals(bf, recovered.time, spread_bb.fs)
        else:
            return ResampledSignals(recovered.signals, recovered.time, spread_bb.fs)

    def sum_plot(self, time, signal):
        plt.rcParams['font.size'] = 15
        plt.figure()
        plt.plot(time, signal)
        plt.title(f"Combined Signal from {self.conf.no_elem} Elements")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.show()

    def signal_plot(self, 
        title,
        time,
        asignal,
        fsample
        ):
        plt.rcParams['font.size'] = 15
        plt.figure()
        plt.plot(time, asignal, '-o', label="RRC Filtered Signal")
        plt.title(f"{title}")

        if self.conf.modType == "BPSK":
            #Real symbol waveform recovered after downconversion 
            #otherwise QPSK does not look like anything useful
            plt.stem(signal.symbol_time, signal.rrcsymbols, 'orange', label="Upsampled Symbols")
    
        plt.legend(loc='upper right')
        plt.grid()
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
    
        plt.figure(figsize=(15,7))
        plt.title(f"PSD of {title}")
        specan = SpecAn().calculate_psd(
            fsample,
            asignal,
            )
        
        """
        plt.semilogy(
            specan.f,
            specan.Pxx_dB,
            color="blue",
            labqel="PSD")
        """
        plt.plot(
            specan.f,
            specan.Pxx_dB,
            '-b' 
        )
        
        plt.ylabel("Power")
        plt.xlabel("Frequency (MHz)")
        plt.grid()
        plt.show()

    def spread_plot(self, 
        index=0
        ):

        plt.rcParams['font.size'] = 15
        plt.figure(figsize=(15,7))
        plt.title(f"Spread Signals of Element={index}")
        plt.plot(self.spread_bb.time, self.spread_bb.signals[index], '-g', label="Spread signal")
        plt.plot(self.rx_time, self.filtered_bb[index], '-b', label=f"Rx signal")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        #plt.stem(tbefore[:len(symbols)],symbols, 'orange')
        plt.legend(loc='upper right')
        plt.grid()

        specan = SpecAn().calculate_psd(
            self.spread_bb.fs,
            self.spread_bb.signals[index]
            )
        
        #FORPLOT
        #np.save(f"sps_{self.conf.lc}_psd_f", specan.f)
        #np.save(f"sps_{self.conf.lc}_psd_Pxx_dB", specan.Pxx_dB)

        plt.figure(figsize=(15,7))
        plt.plot(specan.f, specan.Pxx_dB, '-g')
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Amplitude")
        plt.title(f"PSD of Spread Signals of Element={index}")
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
        plt.yticks(np.arange(-30, 1, 20))
        plt.ylim(-90,0)
        plt.grid()
        plt.show()

    def recover_plot(self,
        tbefore,
        before_signal,
        tafter,
        after_signal,
        fs,
        index=0
        ):
        plt.rcParams['font.size'] = 15
        plt.figure(figsize=(15,7))
        plt.title(f"Recovered Signal at Element Path={index+1}")
        plt.plot(tbefore, before_signal[index], '--b', label=f"Rx signal")
        plt.plot(tafter, after_signal[index], '-g', linewidth='3', label="Recovered signal")
        plt.legend(loc='upper right')
        plt.grid()
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")


        specan = SpecAn().calculate_psd(
            fs,
            after_signal[index]
            )


        plt.figure(figsize=(15,7))
        plt.plot(specan.f, specan.Pxx_dB, label="PSD")
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Amplitude")
        plt.title(f"PSD Recovered Signal at Element Path={index+1}")
        
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
    
    Rb = 1e3
    Tb = 1/Rb
    symbols = 500
    totaltime = symbols*Tb
    lc = 4
    config = CDMcfg(
        rb = Rb,
        lc = lc,
        sps_chip= 2,
        alpha = 0.5,
        simtime = totaltime,
        no_elem = 2,
        codefam="WH",
        modType="BPSK",
        codeids=(np.arange(1,lc))
        )
    #create signal 
    signal = DiscreteSignal(config)

    #initiate and assemble receiver 
    rx = Receiver(config, {PlotType.signal}) 
    rx.setup(signal)
    rx.assemble()