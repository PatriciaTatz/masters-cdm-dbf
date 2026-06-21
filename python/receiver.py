# TODO you can use isort to sort these
from components import CDMx
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
        ups= self.conf.Lc * self.conf.SF
        siglen = len(signal.elements[0])

        #upsampled time axis
        self.time = np.linspace(
            signal.time[0],
            signal.time[-1],
            (siglen - 1) * ups + 1
            )

        self.rxsignals = [
            interp1d(signal.time, nsignal, kind='linear')(self.time) 
            for nsignal in signal.elements
            ]

        upsiglen = len(self.rxsignals[0])
        assert upsiglen == (siglen - 1) * ups + 1, "Upsampling error in length!"

        if plot=="show":
            plt.plot(self.time, self.rxsignals[0], '-x', label="Upsampled element 0 signal")
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


    def assemble(self):
        self.spread = []
        self.despread= []

        #spreader
        for i in range(self.conf.no_arrays):
            nSpread = self.cdmx.spreader(
                self.codes[i],
                self.conf.Lc,
                self.rxsignals[i]
            )
            corrlen =  len(self.rxsignals[i])*self.conf.Lc*self.conf.SF
            assert len(nSpread) == corrlen, \
                f"Signal did not spread to correct length, expected={corrlen} but got {len(nSpread)}"

            self.spread.append(nSpread)


        #combiner , element wise 
        self.sum = [sum(values) for values in zip(*self.spread)]

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


if __name__ == "__main__":


    #Method to simulate CDM4DBF

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
        SF = 2,
        no_arrays=2
    )

    #create signal 
    signal = DiscreteSignal(config)

    #initiate and assemble receiver 
    rx = Receiver(config)
    rx.setup(signal, plot="show")
    rx.assemble() 

    #show plots to see how it evolves & verify its working 
    path = 0 #array path to plot 
    plt.figure()
    plt.plot(signal.input, '-o', label='Original Rx Signal')
    plt.plot(signal.Despread[path][0], '--o', label='Decorrelated Signal')
    plt.legend()
    plt.xlabel("Sample Index")
    plt.title("c2xc3 Rx Signal")
    plt.grid()
    plt.show()