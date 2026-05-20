# TODO you can use isort to sort these
import components as rf
from signals import DiscreteSignal, ArraySignals, SymbolSampler
import scipy
import numpy as np
import utils as u
from config import dbfCfg, signalCfg
import matplotlib.pyplot as plt 

class Receiver:
    def __init__(self, config, array):
        self.dbf= config
        self.multiplexer = rf.Multiplexer()
        self.array = array 

        #create mf 
        self.Ts = self.array.signal.conf.Ts
        self.Ns = self.array.signal.conf.Ns

        #sampler 
        self.sampler = SymbolSampler()
    

    def assemble(self):
        #idealised code set indexes
        code = rf.CodeGenerator(self.dbf)
        code_list = [code.generate(2), code.generate(3)]

        #spreader
        for i in range(self.dbf.no_arrays):
            nSpreadSignal = self.multiplexer.encoder(
                code_list[i],
                self.array.inputSignals[i])

            self.array.Spread = np.vstack((self.array.Spread, nSpreadSignal))

        #combiner 
        self.array.SprSum = np.sum(self.array.Spread, axis=0)
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
    sigconf = signalCfg(Ts=4, modType="BPSK", Ns=4)  
    dbfconf = dbfCfg(Lc=4, code="WH", no_arrays=2)   

    #create signal 
    signal = DiscreteSignal(sigconf, channel="noNoise", SNR=5)
    txsignal = signal.transmit
    array = ArraySignals()
    array.create_inputsignals(signal, dbfconf.no_arrays)
    
    #preserve memory for processed signals
    Lss = len(txsignal) * dbfconf.Lc
    array.create_rxsignals(Lss, sigconf.Ns)

    #initiate and assemble receiver 
    rx = Receiver(dbfconf, array)
    rx.assemble() 



    #show plots to see how it evolves & verify its working 
    path = 0 #array path to plot 
    fig, axs = plt.subplots(3, 1)
    axs[0].stem(array.signal.symbols)
    axs[0].set_title("Tx Symbols")

    axs[1].plot(array.inputSignals[path], '--o')
    axs[1].set_title("Rx Signal")

    axs[2].stem(array.Despread[path][0])
    axs[2].set_title("Downsampled Despread Signal")

    for ax in axs:
        ax.grid(True)

    plt.show()

