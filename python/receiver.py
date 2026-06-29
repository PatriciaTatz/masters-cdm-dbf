# TODO you can use isort to sort these
from components import CDMx, SpecAn
from signals import DiscreteSignal
from scipy.signal import upfirdn, resample_poly
import numpy as np
import utils as u
from config import CXBF
import matplotlib.pyplot as plt 

class Receiver():
    def __init__(self, config, plot):
        self.conf  = config 
        self.plot = plot
        if self.plot == "all":
            self.plot="signal,spread,sum,recovered"

    def setup(self, signal):
        self.time = np.arange(len(signal.elements[0]))/self.conf.Fs

        self.cdmx = CDMx(self.conf)
        self.codes = self.cdmx.generate_code(self.time)
        self.rxsignal = signal.elements.copy()

        if "signal" in self.plot:
            self.signal_plot(
                signal.time,
                signal.elements[0],
                signal.bpsksymbols,
                self.conf.Fs
            )
        
        dt = 1/self.conf.Fs 
        if self.conf.Lc < self.conf.sps_symbol:
            ups = 1
            ds = self.conf.sps_symbol//self.conf.Lc
            newdt = dt * ds 
        elif self.conf.Lc > self.conf.sps_symbol:
            ds = 1
            ups = 2* self.conf.Lc // self.conf.sps_symbol
            newdt  = dt / ups
        else:
            ds = 1 
            ups = 1

        for i in range(self.conf.no_elem):
            self.rxsignal[i] = resample_poly(signal.elements[i], ups, ds)

        self.newFs = 1/newdt
        self.time = np.arange(len(self.rxsignal[0]))*newdt

        if "signal" in self.plot:
            self.signal_plot(
                self.time,
                self.rxsignal[0],
                signal.bpsksymbols,
                self.newFs
            )
   
        
    def assemble(self):
        #====================
        #     SPREADER      |
        #====================
        self.rxspread = []
        for i in range(self.conf.no_elem):
            nSpread = self.cdmx.spreader(
                self.codes[i],
                self.rxsignal[i]
            )
            self.rxspread.append(nSpread)

        if "spread" in self.plot:
            self.spread_plot(
                self.time, self.rxsignal,  
                self.time, self.rxspread, 
                signal.bpsksymbols,
                0
            )

        #====================
        #     COMBINER      |
        #====================
        self.sum = [sum(values) for values in zip(*self.rxspread)]

        if "sum" in self.plot: 
            self.sum_plot()


        #====================
        #     DESPREAD      |
        #====================
        self.rxdespread= []
        for i in range(self.conf.no_elem):
            nDespread = self.cdmx.despreader(
                self.codes[i],
                self.sum
                )
            self.rxdespread.append(nDespread)

        if "recovered" in self.plot:
            self.spread_plot(
                signal.time, signal.elements,  
                self.time, self.rxdespread, 
                signal.bpsksymbols,
                0
            )


        return signal

    def sum_plot(self):
        plt.figure()
        plt.plot(self.time, self.sum)
        plt.title(f"Combined Signal from {self.conf.no_elem} Elements")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.show()

    def signal_plot(self, time, asignal, symbols, Fsample):
        plt.figure()
        plt.plot(time*1e9, asignal, '--o')
        plt.title(f"Simulated Rx Signal Tsim={config.simtime}")
        plt.stem(signal.time[:len(symbols)]*1e9,symbols, 'orange')
        plt.grid()
        plt.xlabel("Time (nsec)")
        plt.ylabel("Amplitude")

        plt.figure()
        f, Pxx, bw = SpecAn().calculate_psd(
            self.conf.Fs,
            asignal,
            threshold=-20
            )
        plt.plot(f/1e6, Pxx, label="PSD")
        plt.text(
            0.02, 0.95,
            f"99.9% BW={bw/1e6:.2f}MHz",
            transform=plt.gca().transAxes,
            ha='left',
            va='top',
            bbox=dict(facecolor='white', edgecolor='black')
        )
        plt.show()

    def spread_plot(self,tbefore, before_signal, tafter, after_signal, symbols, index=0):
        plt.figure()
        plt.title(f"Spread Signal of Element {index}")
        plt.plot(tbefore, before_signal[index], '--ob', label=f"Before element {index} signal")
        plt.plot(tafter, after_signal[index], '-og', label="After spread")
        #plt.stem(tbefore[:len(symbols)]*1e9,symbols, 'orange')
        plt.legend()
        plt.grid()


        f, Pxx, bw = SpecAn().calculate_psd(
            self.newFs,
            after_signal[index],
            threshold=-23)

        plt.figure()
        plt.plot(f/1e6, Pxx, label="PSD")
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Amplitude")
        plt.text(
            0.02, 0.95,
            f"99.99% BW={bw/1e6:.2f}MHz",
            transform=plt.gca().transAxes,
            ha='left',
            va='top',
            bbox=dict(facecolor='white', edgecolor='black')
        )

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

    Rb =6.25e6
    Tb = 1/Rb
    symbols = 100
    totaltime = symbols*Tb

    config = CXBF(
        Rb = Rb,
        Lc = 32,
        sps_chip= 1,
        alpha = 0.5,
        simtime = totaltime,
        no_elem = 2
    )

    #create signal 
    signal = DiscreteSignal(config)

    #initiate and assemble receiver 
    rx = Receiver(config, "spread")
    rx.setup(signal)
    rx.assemble() 
