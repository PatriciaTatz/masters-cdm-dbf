# TODO you can use isort to sort these
import components as rf
from mysignal import DiscreteSignal
import scipy
import numpy as np
import utils as u
from config import cdmCfg, signalCfg
import matplotlib.pyplot as plt 

class Receiver:
    class ArraySignals: 
        def __init__(self, arraySpread, arraySpreadSum, arrayDespread, arrayDecisionSymbols):
            self.SpreadSignals = arraySpread
            self.SpreadSumSignals = arraySpreadSum
            self.DeSpreadSignals = arrayDespread
            self.DecisionSymbols = arrayDecisionSymbols

    def __init__(self, config):
        self.codecfg = config

        #TODO preserve memory 
        self.multiplexer = rf.Multiplexer()

    def assemble(self, signal):
        self.Lss = self.codecfg.Lc*signal.conf.Ls

        #spreading of array signals    
        # TODO fix up multiplexer
        arraySpread = np.empty((0,self.Lss))

        #idealised code set indexes
        code = rf.CodeGenerator(self.codecfg)
        code_list = [code.generate(2), code.generate(3)]

        #spreader
        for i in range(signal.conf.no_arrays):
            nSpreadSignal = self.multiplexer.encoder(
                code_list[i],
                signal.arrSignals[i])

            arraySpread = np.vstack((arraySpread,nSpreadSignal))

        #combiner 
        arraySpreadSum = np.sum(arraySpread, axis=0)

        #despreader
        arrayDespread = np.empty((0,self.Lss))
        arrayDecisionSymbols = np.empty((0,signal.conf.Ls))
        for i in range(signal.conf.no_arrays):
            nDespreadSignal = self.multiplexer.decoder(
                self.codecfg.Lc,
                signal.conf.Ts,
                code_list[i],
                arraySpreadSum)

            nDecisionSymbols = self.multiplexer.downsample(nDespreadSignal)

            arrayDespread = np.vstack((arrayDespread,nDespreadSignal))
            arrayDecisionSymbols = np.vstack((arrayDecisionSymbols,nDecisionSymbols))

        self.arraySignals = self.ArraySignals(arraySpread, arraySpreadSum, arrayDespread, arrayDecisionSymbols)

        return self.arraySignals


if __name__ == "__main__":
    #configure
    codeCfg = cdmCfg(Lc=32, code="WH") 
    sigCfg = signalCfg(Ts=1, modType="BPSK", Ls=30, no_arrays=2)
    snr = 25
    path = 0
    #assemble reciever 
    inputSignal = DiscreteSignal(sigCfg, channel="noNoise", SNR=snr)     
    rx = Receiver(codeCfg)
    rx.assemble(inputSignal) 

    #show plots to see how it evolves & verify its working 
    fig, axs = plt.subplots(2, 1)
    axs[0].stem(inputSignal.arrSignals[path])
    axs[0].set_title("Input Signal")

    axs[1].stem(rx.arraySignals.DecisionSymbols[path])
    axs[1].set_title("Downsampled Despread Signal")

    for ax in axs:
        ax.grid(True)

    #show plots to see how it evolves & verify its working 
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(inputSignal.arrSignals[path])
    axs[0].set_title("Input Signal")

    axs[1].plot(rx.arraySignals.DeSpreadSignals[path])
    axs[1].set_title("Despread Signal")

    for ax in axs:
        ax.grid(True)

    #show plots to see how it evolves & verify its working 
    fig, axs = plt.subplots(2, 1)
    axs[0].stem(inputSignal.arrSignals[path])
    axs[0].set_title("Input Signal")

    axs[1].plot(rx.arraySignals.SpreadSignals[path])
    axs[1].set_title("Spread Signal")

    plt.tight_layout(pad=1.0)
    for ax in axs:
        ax.grid(True)
    plt.show()
