# TODO you can use isort to sort these
from components import CDMx, CodeGenerator
from signals import DiscreteSignal
import scipy
import numpy as np
import utils as u
from config import CXBF
import matplotlib.pyplot as plt 

class Receiver():
    def __init__(self, config):
        self.conf  = config 
        self.cdmx = CDMx()
        code = CodeGenerator()
        self.code_list = [code.generate(k, self.conf.Lc, self.conf.codefam) for k in self.conf.codeids]

    def assemble(self, signal):
        self.spread = []
        self.despread= []

        #spreader
        for i in range(self.conf.no_arrays):
            nSpread = self.cdmx.spreader(
                self.code_list[i],
                signal.element[i])
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
        no_symbols= 500, 
        simtime=500*1/6.25e6, 
        modType="BPSK",
        Lc = 4,
        no_arrays=2
    )

    #create signal 
    signal = DiscreteSignal(config)

    #initiate and assemble receiver 
    rx = Receiver(config)
    rx.assemble(signal) 

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