from enum import Enum, auto

import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.signal import resample_poly, upfirdn

import utils as u
from components import CDMx, SpecAn, Beamformer, Filter, Mixer
from config import CDMcfg
from signals import DiscreteSignal


class PlotType(Enum):
    signal = auto()
    spread = auto()
    sum = auto()
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
        self.time = np.arange(len(signal.elements[0]))/self.conf.fs
        self.cdmx = CDMx(self.conf)
        self.bf = Beamformer()
        self.mixer = Mixer(self.conf)
        self.filter = Filter(self.conf)

        self.codes = self.cdmx.generate_codes()

        #recieve the signal and get its attributes relevant to operations 
        self.rxsignals = copy.deepcopy(signal.elements) 
        self.rx_fs = signal.fs
        self.rx_time = signal.time 
        self.psi_delays = signal.psi

        if PlotType.signal in self.plot:
            self.signal_plot(
                "Rx Signal at Element 1",
                signal.time,
                signal.elements[0],
                signal.rrcsymbols,
                self.conf.fs,
            )

    def assemble(self):
        if not self.conf.CALIBRATE: 
            #==============================
            #        D/C  PER ELEMENT     |
            #==============================            
            IFsignal = self.mixer.superhet(
                self.rxsignals,
                self.rx_fs
            )

            #============================
            #       LPF  PER ELEMENT    |
            #===========================
            filtered_signals = self.filter.butterworth(
                cutoff = self.conf.rb/2,
                fs = IFsignal.fs,
                signal = IFsignal.elements
            )
            
            if PlotType.signal in self.plot:
                self.signal_plot(
                    "Rx Signal at Element 1",
                    signal.time,
                    filtered_signals[0],
                    signal.rrcsymbols,
                    self.conf.fs
                )
            
            #====================
            #     SPREADER      |
            #====================
            spread = self.cdmx.spreader(
                filtered_signals,
                self.codes,
            )
            
            if PlotType.spread in self.plot:
                self.spread_plot(
                    self.time, filtered_signal,  
                    spread.time, spread.signals, 
                    spread.fs,
                    0
                )
    
            #====================
            #     COMBINER      |
            #====================
            summed_signal = [sum(values) for values in zip(*spread.signals)]
            if PlotType.sum in self.plot: 
                self.sum_plot(spread.time, summed_signal)
            
            if PlotType.spread in self.plot:
                self.spread_plot(
                    self.time, filtered_signals,  
                    spread.time, [np.array(summed_signal)], 
                    spread.fs,
                    0
                )

            #====================
            #     DESPREAD      |
            #====================
            recovered = self.cdmx.recover(
                summed_signal,
                spread.codes,
                spread.fs
                )
            
            if PlotType.recovered in self.plot:
                self.recover_plot(
                    self.time, filtered_signals, 
                    recovered.time, recovered.signals,
                    recovered.fs,
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
                    self.time, filtered_signals, 
                    recovered.time, recovered.signals,
                    recovered.fs,
                    0
                )

        if self.conf.BEAMFORM:
        #====================
        #     BEAMFORMED     |
        #====================
            bf = self.bf.align_combine_signals(
                self.psi_delays,
                despread.signals
            )

            if PlotType.bf in self.plot:
                self.bf_plot(
                    self.time, filtered_signals, 
                    despread.time, bf,
                    0
                )  
            return bf  
        else:
            return recovered

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
        symbols,
        Fsample
        ):
        plt.rcParams['font.size'] = 15
        plt.figure()
        plt.plot(time, asignal, '-o', label="RRC Filtered Signal")
        plt.title(f"{title}")
        plt.stem(time[:len(symbols)],symbols, 'orange', label="Upsampled Symbols")
    
        plt.legend(loc='upper right')
        plt.grid()
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")

        plt.figure()
        plt.title(f"PSD of {title}")
        specan = SpecAn().calculate_psd(
            self.conf.fs,
            asignal,
            psdtype=-30
            )
        
        plt.plot(specan.f/1e6, specan.Pxx_dB, label="PSD")

        
        plt.text(
            0.02, 0.95,
            f"Est. BW={specan.bw/1e6:.2f}MHz",
            transform=plt.gca().transAxes,
            ha='left',
            va='top',
            bbox=dict(facecolor='white', edgecolor='black')
        )
        
        plt.ylabel("Power (dB)")
        plt.xlabel("Frequency (MHz)")
        plt.grid()
        plt.show()

    def spread_plot(self, 
        tbefore,
        before_signal,
        tafter,
        after_signal,
        fs,
        index=0
        ):

        plt.rcParams['font.size'] = 15
        plt.figure()
        plt.title(f"Spread Signals Combined")
        plt.plot(tafter, after_signal[index], '-g', label="Spread signal")
        #plt.plot(tbefore, before_signal[index], '-b', label=f"Rx signal")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        #plt.stem(tbefore[:len(symbols)],symbols, 'orange')
        plt.legend(loc='upper right')
        plt.grid()
        
        """
        plt.figure()
        plt.title(f"Spread Composite Signal")
        plt.plot(tafter, after_signal, '-', label="Composite signal")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        #plt.legend()
        plt.grid()
        """

        specan = SpecAn().calculate_psd(
            fs,
            after_signal[index],
            psdtype=-25.8)
        
        #FORPLOT
        #np.save(f"sps_{self.conf.lc}_psd_f", specan.f)
        #np.save(f"sps_{self.conf.lc}_psd_Pxx_dB", specan.Pxx_dB)

        plt.figure()
        plt.plot(specan.f/1e6, specan.Pxx_dB, '-g')
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Amplitude")
        plt.title(f"PSD Spread Signals Combined")
        plt.text(
            0.02, 0.95,
            f"Est. BW={specan.bw/1e6:.2f}MHz",
            transform=plt.gca().transAxes,
            ha='left',
            va='top',
            bbox=dict(facecolor='white', edgecolor='black')
        )
        plt.ylabel("Power (dB)")
        plt.ylim(-40, 0)
        plt.yticks(np.arange(-30, 1, 20))
        plt.xlabel("Frequency (MHz)")
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
        plt.figure()
        plt.title(f"Recovered Signal at Element Path={index+1}")
        plt.plot(tbefore, before_signal[index], '--b', label=f"Rx signal")
        plt.plot(tafter, after_signal[index], '-g', linewidth='3', label="Recovered signal")
        plt.legend(loc='upper right')
        plt.grid()
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")


        specan = SpecAn().calculate_psd(
            fs,
            after_signal[index],
            psdtype=-25)


        plt.figure()
        plt.plot(specan.f/1e6, specan.Pxx_dB, label="PSD")
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Amplitude")
        plt.title(f"PSD Recovered Signal at Element Path={index+1}")
        
        plt.text(
            0.02, 0.95,
            f"Est. BW={specan.bw/1e6:.2f}MHz",
            transform=plt.gca().transAxes,
            ha='left',
            va='top',
            bbox=dict(facecolor='white', edgecolor='black')
        )
        
        plt.ylabel("Power (dB)")
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
        plt.figure()
        plt.title(f"Combined and Phase aligned Signal")
        plt.plot(tbefore, before_signal[index], '--b',alpha=0.5, label=f"Rx signal + Interference")
        plt.plot(tafter, after_signal, '-g', linewidth=3, label="Beamformed Signal")

        #INTERFERENCE PLOT ONLY
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
    symbols = 100
    totaltime = symbols*Tb

    config = CDMcfg(
        rb = Rb,
        lc = 32,
        sps_chip= 2,
        alpha = 0.5,
        simtime = totaltime,
        no_elem = 2,
        codefam="WH",
        modType="BPSK"
        )
    #create signal 
    signal = DiscreteSignal(config)

    #initiate and assemble receiver 
    rx = Receiver(config, PlotType.all()) 
    rx.setup(signal)
    rx.assemble() 