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
        ups= int(self.conf.Lc * self.conf.SF)
        self.fs = ups/signal.dt
        siglen = len(signal.elements[0])

        #upsampled time axis
        self.rxsignal = [
            np.repeat(sig, ups)
            for sig in signal.elements
        ]
        self.time = np.arange(len(self.rxsignal[0])) * signal.dt / ups 

        upsiglen = len(self.rxsignal[0])
        assert upsiglen == siglen*ups, "Upsampling error in length!"

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

        if plot=="show":
            self.spread_plot(0)

        #combiner , element wise 
        self.sum = [sum(values) for values in zip(*self.rxspread)]

        #note: changes amplitude 
        #despreader + decorrelator
        for i in range(self.conf.no_arrays):
            nDespreadSignal = self.cdmx.despreader(
                self.conf.Lc,
                self.conf.Ts,
                self.code_list[i],
                self.sum)

            self.signal.Despread.append(nDespreadSignal)
        return signal

    def spread_plot(self,index):
        plt.figure(1)
        plt.title(f"Spread Signal of Element {index}")
        plt.plot(self.time, self.rxsignal[index], '-x', label=f"Upsampled element {index} signal")
        plt.plot(self.time, self.rxspread[index], '--x', label=f"Spread element {index} signal" )
        plt.legend()
        plt.grid()

        f, Pxx, bw = SpecAn().calculate_psd(self.fs, self.rxsignal[index])
        plt.figure(2)
        plt.semilogy(f/1e6, Pxx, label="PSD")
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Amplitude")
        plt.text(
            0.02, 0.95,
            f"99.9% BW={bw/1e6:.2f}MHz",
            transform=plt.gca().transAxes,
            ha='left',
            va='top',
            bbox=dict(facecolor='white', edgecolor='black')
        )
        plt.text(
            0.02, 0.85,
            f"Theoretical BW={1/config.Ts*(1+0.5)*self.conf.Lc/1e6}MHz",
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
    sps: samples per symbol 
    Ns: no of symbols = e.g. 5 unique symbols which lasts 10 seconds each post filter 
    modType: modulation type = BPSK 
    no_arrays: number of array signals t 
    """
    config = CXBF(
        Ts = 1/6.25e6,
        sps = 10,
        simtime=500/6.25e6, 
        modType="BPSK",
        Lc = 4,
        SF = 1,
        no_arrays=2
    )

    #create signal 
    signal = DiscreteSignal(config)

    #initiate and assemble receiver 
    rx = Receiver(config)
    rx.setup(signal, plot="show")
    rx.assemble(plot="show") 
