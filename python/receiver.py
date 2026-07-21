from enum import Enum, auto

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample_poly, upfirdn

import utils as u
from components import CDMx, SpecAn
from config import CDMcfg
from signals import DiscreteSignal


class PlotType(Enum):
    signal = auto()
    spread = auto()
    sum = auto()
    recovered = auto()

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
        self.codes = self.cdmx.generate_codes()
        self.rxsignals = signal.elements.copy()

        if PlotType.signal in self.plot:
            self.signal_plot(
                "",
                signal.time,
                signal.elements[0],
                signal.rrcsymbols,
                self.conf.fs
            )
        
        
    def assemble(self):
        #====================
        #     SPREADER      |
        #====================
        spread = self.cdmx.spreader(
            self.rxsignals,
            self.codes,
        )
        
        """
        if PlotType.spread in self.plot:
            self.spread_plot(
                self.time, self.rxsignals,  
                spread.time, spread.signals, 
                signal.rrcsymbols,
                spread.fs,
                0
            )
        """

        #====================
        #     COMBINER      |
        #====================
        summed_signal = [sum(values) for values in zip(*spread.signals)]
        if PlotType.sum in self.plot: 
            self.sum_plot(spread.time, summed_signal)
        
        #FORTEST
        #summed_signal = spread.signals

        
        if PlotType.spread in self.plot:
            self.spread_plot(
                self.time, self.rxsignals,  
                spread.time, np.array(summed_signal), 
                signal.rrcsymbols,
                spread.fs,
                0
            )
        
        #====================
        #     DESPREAD      |
        #====================
        despread = self.cdmx.despreader(
            summed_signal,
            spread.codes,
            spread.fs
            )

        if PlotType.recovered in self.plot:
            self.despread_plot(
                signal.time, signal.elements,  
                despread.time, despread.signals,
                signal.rrcsymbols,
                spread.fs,
                0
            )

        return spread, despread

    def sum_plot(self, time, signal):
        plt.figure()
        plt.plot(time, signal)
        plt.title(f"Combined Signal from {self.conf.no_elem} Elements")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.show()

    def signal_plot(self, title, time, asignal, symbols, Fsample):
        plt.figure()
        plt.plot(time, asignal, '-o', label="RRC Filtered Signal")
        plt.title(f"{title} Tx Signal")
        plt.stem(signal.time[:len(symbols)],symbols, 'orange', label="Upsampled Symbols")
        plt.legend()
        plt.grid()
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")

        plt.figure()
        plt.title(f"PSD of {title} Tx Signal")
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

    def spread_plot(self,tbefore, before_signal, tafter, after_signal, symbols, fs, index=0):
        plt.rcParams['font.size'] = 18 
        
        """
        plt.figure()
        plt.title(f"Spread Element={index+1} Signal")
        plt.plot(tbefore, before_signal[index], '--ob', label=f"Tx signal")
        plt.plot(tafter, after_signal[index], '-og', label="Spread signal")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        #plt.stem(tbefore[:len(symbols)],symbols, 'orange')
        plt.legend()
        plt.grid()
        """

        plt.figure()
        plt.title(f"Spread Composite Signal")
        plt.plot(tafter, after_signal, '-', label="Composite signal")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        #plt.stem(tbefore[:len(symbols)],symbols, 'orange')
        #plt.legend()
        plt.grid()

        specan = SpecAn().calculate_psd(
            fs,
            after_signal,
            psdtype="normal")
        
        #FORPLOT
        #np.save(f"sps_{self.conf.lc}_psd_f", specan.f)
        #np.save(f"sps_{self.conf.lc}_psd_Pxx_dB", specan.Pxx_dB)

        plt.figure()
        plt.plot(specan.f/1e6, specan.Pxx_dB, '-' ,label="PSD")
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Amplitude")
        plt.title(f"PSD Spread Element={index+1} Signal")
        plt.text(
            0.02, 0.95,
            f"Est. BW={specan.bw/1e6:.2f}MHz",
            transform=plt.gca().transAxes,
            ha='left',
            va='top',
            bbox=dict(facecolor='white', edgecolor='black')
        )
        plt.ylabel("Power (dB)")
        plt.ylim(-80, 0)
        plt.yticks(np.arange(-80, 1, 20))
        plt.xlabel("Frequency (MHz)")
        plt.grid()
        plt.show()

    def despread_plot(self,tbefore, before_signal, tafter, after_signal, symbols, fs, index=0):
        plt.figure()
        plt.title(f"Decorrelated Element={index+1} Signal")
        plt.plot(tbefore, before_signal[index], '--b', label=f"Tx signal")
        plt.plot(tafter, after_signal[index], '-g', label="Recovered signal")
        #plt.stem(tbefore[:len(symbols)]*1e9,symbols, 'orange')
        plt.legend()
        plt.grid()

        specan = SpecAn().calculate_psd(
            fs,
            after_signal[index],
            psdtype=-50)


        plt.figure()
        plt.plot(specan.f/1e6, specan.Pxx_dB, label="PSD")
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Amplitude")
        plt.title(f"PSD Decorrelated Element={index+1} Signal")
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
    
    Rb = 6.25e6
    Tb = 1/Rb
    symbols = 300
    totaltime = symbols*Tb

    config = CDMcfg(
        rb = Rb,
        lc = 32,
        sps_chip= 2,
        alpha = 0.5,
        simtime = totaltime,
        no_elem = 2
        )
    
    #create signal 
    signal = DiscreteSignal(config)

    #initiate and assemble receiver 
    rx = Receiver(config, PlotType.all())
    rx.setup(signal)
    rx.assemble() 


    """ 
    plt.rcParams['font.size'] = 20 
    f1 = np.load(f"sps_4_psd_f.npy")
    Pxx1 = np.load(f"sps_4_psd_Pxx_dB.npy")
    f2 = np.load(f"sps_16_psd_f.npy")
    Pxx2 = np.load(f"sps_16_psd_Pxx_dB.npy")
        
    plt.figure()
    plt.plot(f1/1e6,  Pxx1, '-' ,label="Lc=4")
    plt.plot(f2/1e6,  Pxx2, '-' ,label="Lc=16")

    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Amplitude")
    plt.title("PSD of Spread Composite Signal")

    plt.ylabel("Power (dB)")
    plt.xlabel("Frequency (MHz)")
    plt.legend()
   
    plt.grid()
    plt.show()
    """
    