# TODO you can use isort to sort these
import components as rf
from signals import DiscreteSignal, ArraySignals
import scipy
import numpy as np
import utils as u
from config import dbfCfg, signalCfg
import matplotlib.pyplot as plt 

class Receiver:
    def __init__(self, config):
        self.dbf= config
        self.multiplexer = rf.Multiplexer()


    def assemble(self, array):
        #create mf 
        self.array = array 
        self.Ts = self.array.signal.conf.Ts
        self.Ns = self.array.signal.conf.Ns

        #idealised code set indexes
        code = rf.CodeGenerator(self.dbf)
        code_list = [code.generate(2), code.generate(3)]
    
        #spreader
        for i in range(self.dbf.no_arrays):
            nSpreadSignal = self.multiplexer.encoder(
                code_list[i],
                self.array.inputSignals[i])
            self.array.Spread.append(nSpreadSignal)

        #combiner , element wise 
        self.array.SprSum = [sum(values) for values in zip(*self.array.Spread)]

        #note: changes amplitude 
        #despreader + decorrelator
        for i in range(self.dbf.no_arrays):
            nDespreadSignal = self.multiplexer.decoder(
                self.dbf.Lc,
                self.Ts,
                code_list[i],
                self.array.SprSum)

            self.array.Despread.append(nDespreadSignal)
        return array


if __name__ == "__main__":
    #Method to simulate CDM4DBF

    #config settings 
    sigconf = signalCfg(Ts=5, modType="BPSK", Ns=1)  
    dbfconf = dbfCfg(Lc=5, code="mLength", no_arrays=1)   

    #create signal 
    signal = DiscreteSignal(sigconf)
    txsignal = signal.add_awgn(signal.transmit, SNR=5)
    array = ArraySignals()
    array.create_inputsignals(signal, dbfconf.no_arrays)
    
    #preserve memory for processed signals
    Lss = len(txsignal) * dbfconf.Lc
    array.create_rxsignals(Lss, sigconf.Ns)

    #initiate and assemble receiver 
    rx = Receiver(dbfconf)
    rx.assemble(array) 

    #show plots to see how it evolves & verify its working 
    path = 0 #array path to plot 
    plt.figure()
    plt.plot(array.inputSignals[path], '-o', label='Original Rx Signal')
    plt.plot(array.Despread[path][0], '--o', label='Decorrelated Signal')
    plt.legend()
    plt.xlabel("Sample Index")
    plt.title("c2xc3 Rx Signal")
    plt.grid()
    plt.show()