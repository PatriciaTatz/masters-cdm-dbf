# TODO you can use isort to sort these
import components as rf
import mysignal as s 
import measure as measure
import scipy
import numpy as np
import utils as u
from config import * 

class Receiver:
    class Signals: 
        def __init__(self, arraySpread, arraySpreadSum, arrayDespread, arrayDecisionSymbols):
            self.arraySpreadSignals = arraySpread
            self.arraySpreadSumSignals = arraySpreadSum
            self.arrayDeSpreadSignals = arrayDespread
            self.arrayDecisionSymbols = arrayDecisionSymbols

    def __init__(self, codeconfig, signal):
        self.codecfg = codeconfig
        self.signal = signal
        self.Lss = rxconfig.Lc*signal.conf.Ls

        self.multiplexer = rf.Multiplexer()

        self.signals = self.assemble(self.multiplexer)

    def assemble(self, multiplexer):
        #spreading of array signals    
        # TODO fix up multiplexer
        arraySpread = np.empty((0,self.Lss))

        #idealised code set indexes
        code = rf.CodeGenerator(self.codecfg)
        
        code_list = [code.generate(2), code.generate(3)]

        #spreader
        for i in range(self.signal.conf.no_arrays):
            nSpreadSignal = self.multiplexer.encoder(
                                code_list[i],
                                self.signal.arrSignals[i])

            arraySpread = np.vstack((arraySpread,nSpreadSignal))

        #combiner 
        arraySpreadSum = np.sum(arraySpread, axis=0)

        #despreader
        arrayDespread = np.empty((0,self.Lss))
        arrayDecisionSymbols = np.empty((0,self.signal.conf.Ls))
        for i in range(self.signal.conf.no_arrays):
            nDespreadSignal = self.multiplexer.decoder(
                                self.rxcfg.Lc,
                                self.signal.conf.Ts,
                                code_list[i],
                                arraySpreadSum)

            nDecisionSymbols = self.multiplexer.downsample(nDespreadSignal)

            arrayDespread = np.vstack((arrayDespread,nDespreadSignal))
            arrayDecisionSymbols = np.vstack((arrayDecisionSymbols,nDecisionSymbols))

        return self.Signals(arraySpread, arraySpreadSum, arrayDespread, arrayDecisionSymbols)


if __name__ == "__main__":
    #configure
    codeCfg = cdmCfg(Lc=32, code="WH") 
    sigCfg = signalCfg(Ts=1, modType="BPSK", Ls=10, no_arrays=2)

    #BER simulation
    ber = measure.BER(sigCfg, codeCfg, arrpath=0)
    ber.curve(maxSims=1000, maxSNR=5)
    ber.plot_curve(show_plot="on")
