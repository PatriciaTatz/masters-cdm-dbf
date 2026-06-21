# TODO you can use isort to sort these
from components import CDMx, SpecAn
from signals import DiscreteSignal
from scipy.signal import upfirdn
from scipy.interpolate import interp1d
import numpy as np
import utils as u
from config import CXBF
import matplotlib.pyplot as plt 

class Receiver():
    def __init__(self, config):
        self.conf  = config 

    def setup(self, signal, plot="off"):
        self.ups = int(self.conf.Lc * self.conf.SF)
        self.fs = self.conf.samples_symbol*self.ups / self.conf.Ts 
        siglen = len(signal.elements[0])

        #upsampled time axis
        self.rxsignal = [
            np.repeat(sig, self.ups)
            for sig in signal.elements
        ]
        self.time = np.arange(len(self.rxsignal[0])) * signal.dt / self.ups 

        upsiglen = len(self.rxsignal[0])
        assert upsiglen == siglen*self.ups, "Upsampling error in length!"

        if plot=="show":
            plt.title("Upsampled Signal to Chip Rate")
            plt.plot(self.time, self.rxsignal[0], '-x', label="Upsampled element 0 signal")
            plt.plot(signal.time, signal.elements[0], 'o', label="Original Tx element 0 signal" )
            plt.legend()
            plt.grid()
            plt.show()


        self.cdmx = CDMx()
        self.codes = self.cdmx.generate_code(
            self.conf.no_arrays,
            self.conf.Lc,
            self.conf.codeids,
            self.conf.codefam, 
            )


    def assemble(self, plot="off"):
        self.rxspread = []
        self.rxdespread= []

        #spreader
        for i in range(self.conf.no_arrays):
            nSpread = self.cdmx.spreader(
                self.codes[i],
                self.conf.Lc,
                self.rxsignal[i]
            )
            self.rxspread.append(nSpread)

        if "spread" in plot:
            self.time_psd_plot(
                signal.time, signal.elements, 
                self.time, self.rxspread,
                0
            )

        #combiner , element wise 
        self.sum = [sum(values) for values in zip(*self.rxspread)]

        if "sum" in plot: 
            self.sum_plot()

        for i in range(self.conf.no_arrays):
            nDespread = self.cdmx.despreader(
                self.conf.samples_symbol,
                self.ups,
                self.conf.Ts, 
                self.codes[i],
                self.conf.Lc,
                self.sum)

            self.rxdespread.append(nDespread)

        if "recovered" in plot:
            self.time_psd_plot(
                signal.time, signal.elements, 
                self.time, self.rxdespread,
                0)

        return signal

    def sum_plot(self):
        plt.figure()
        plt.plot(self.time, self.sum)
        plt.title(f"Combined Signal from {self.conf.no_arrays} Elements")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.show()

    def time_psd_plot(self,tbefore, before_signal, tafter, after_signal, index):
        plt.figure()
        plt.title(f"Spread Signal of Element {index}")
        plt.plot(tbefore, before_signal[index], '-xb', label=f"Before element {index} signal")
        plt.plot(tafter, after_signal[index], '--xg', label=f"After element {index} signal" )
        plt.legend()
        plt.grid()

        f, Pxx, bw = SpecAn().calculate_psd(self.fs, np.array(after_signal[index]))
        plt.figure()
        plt.semilogy(f/1e6, Pxx, label="PSD")
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
    no_arrays: number of array signals t 
    """
    config = CXBF(
        Ts = 1/6.25e6,
        samples_symbol = 10,
        simtime=50/6.25e6, 
        modType="BPSK",
        Lc = 4,
        SF = 1,
        no_arrays=2
    )

    #create signal 
    signal = DiscreteSignal(config)

    #initiate and assemble receiver 
    rx = Receiver(config)
    rx.setup(signal)
    rx.assemble(plot="spread, recovered") 
