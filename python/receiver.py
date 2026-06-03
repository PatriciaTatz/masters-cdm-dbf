# TODO you can use isort to sort these
import components as rf
from signals import DiscreteSignal
import scipy
import numpy as np
import utils as u
from config import dbfCfg, signalCfg
import matplotlib.pyplot as plt 

class Receiver:
    def __init__(self, config):
        self.dbf= config
        self.multiplexer = rf.Multiplexer()

    def assemble(self, signal):
        #create mf 
        self.signal = signal 
        self.signal.create_rxsignals() #empty lists 

        self.Ts = self.signal.conf.Ts
        self.Ns = self.signal.conf.Ns

        #idealised code set indexes
        code = rf.CodeGenerator(self.dbf)
        code_list = [code.generate(k) for k in self.dbf.codeids]
        #all empty lists 
        #spreader
        for i in range(self.dbf.no_arrays):
            nSpreadSignal = self.multiplexer.encoder(
                code_list[i],
                self.signal.inputSignals[i])
            self.signal.Spread.append(nSpreadSignal)

        #combiner , element wise 
        self.signal.SprSum = [sum(values) for values in zip(*self.signal.Spread)]

        #note: changes amplitude 
        #despreader + decorrelator
        for i in range(self.dbf.no_arrays):
            nDespreadSignal = self.multiplexer.decoder(
                self.dbf.Lc,
                self.Ts,
                code_list[i],
                self.signal.SprSum)

            self.signal.Despread.append(nDespreadSignal)
        return signal


if __name__ == "__main__":
    #Method to simulate CDM4DBF

    #config settings 
    sigconf = signalCfg(Ts=16, modType="BPSK", Ns=1)  
    dbfconf = dbfCfg(Lc=16, code="WH", no_arrays=1, codeids=[12,15])   

    #create signal 
    signal = DiscreteSignal(sigconf)
    signal.input = signal.add_awgn(signal.input, SNR=-10)
    signal.create_inputsignals(dbfconf.no_arrays) 

    #initiate and assemble receiver 
    rx = Receiver(dbfconf)
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