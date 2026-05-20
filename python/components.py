"""
RF components
- Code generator (WH, Gold, and ZC) 
- Spreader
- Despreader
- Combiner
- Filters
- Matched Filter 
"""

from scipy.linalg import hadamard  
import numpy as np  
from enum import Enum 
import commpy.filters as filter
import komm as km
import matplotlib.pyplot as plt 

class CodeGenerator:
    #Method to generate all code families studied 
    def __init__(self, config):
        self.config = config        

    def generate(self, codeindex):
        #L: code length used to generate code sequences 
        if self.config.code == "WH":
            #construct n-by-n Hadamard matrix using Sylvesters construction
            #n: must be a power of 2 and int 

            WHcode = km.WalshHadamardSequence(length=self.config.Lc, index=codeindex, ordering='sequency')
            return WHcode.polar_sequence
        else: 
            raise Exception("Unrecognised code family! Choose from [WH, ]")


class Multiplexer:
    #Method to spread the signal using a code sequence

    def encoder(self, code, symbols):
        """
        input: 
            code sequence [c1...L]
            symbol [s1 .. s] 
        output: 
            ss = [s1*c1 s1*c2 ... s3*L
                  s2*c1 s2*c2 ... s2*L
                  ....      :      ...
                  s*c1  s*c2  ... s*L]
       """
        ss = np.array([]) #spread symbols
        
        for i in range(len(symbols)):
            #Assumes Ts >> Tx, s is constant over code sequence 
            #ss: spread symbol
            spread_symbol = symbols[i]*code
            ss = np.append(ss, spread_symbol)

        return ss     

    def decoder(self, Lc, Ts, code, spreadsum):   
        #Depends on Decorrelator being run first. ASK ALIX
        despread_symbols = [] 
        for i in range(len(spreadsum)):
            #restarts the code indexing at the end of sequence 
            despread_symbols.append((spreadsum[i]*code[i%Lc]))

        #matched filter / correlator too! 
        sps = Lc #samples per symbol 
        Fs = sps/Ts #sample per second, take Lc samples per second 
        alpha = 0.25 #does this need to match?
        span = 8 #consider 8 symbols bc we upsampled it by that much in tx but thats probs not right because its unknown irl
        Ntaps = span*sps

        dt_filter, h_rrc = filter.rrcosfilter(Ntaps, alpha, Ts, Fs)
        h_delay = (Ntaps-1)/2

        conv_symbols = np.convolve(despread_symbols, h_rrc, mode='full')

        h_delay = int(np.round(h_delay))


        Nsamples = len(spreadsum) // Lc   # number of samples representing symbols that was spread
        idxs = h_delay  + np.arange(Nsamples)*sps
        samples = conv_symbols[idxs]

        return [samples, idxs]


class MatchedFilter:
    def txrrcfilter(self, symbols, Ts):
        """
        rrc filter for transmit
        independent of Lc / Rx settings as its suppose to simulate real rxd signals
        what it does:
        - upsamples
        - applies filter 
        - is called by mysignal.py 
        """
        self.txsps = 4 #samples per symbol, 
        Fs = self.txsps/Ts #Hz; sampling rate, sample per second 
        alpha = 0.25#roll off factor 
        span = 8 #dictates how far a symbols influene spreads across time
        Ntaps = span*self.txsps   #total length of the filter 
        upsampled = np.array([])
        for symbol in symbols:
            pulse = np.zeros(self.txsps)
            pulse[0] = symbol
            upsampled = np.concatenate((upsampled, pulse)) #adds sps*0s after each pulse[0]  

        #apply filter
        dt_filter, h_rrc = filter.rrcosfilter(Ntaps, alpha, Ts, Fs)
        txsignal = np.convolve(upsampled, h_rrc, mode='full')
        group_delay = (Ntaps-1)/2

        return txsignal, group_delay 


class SymbolSampler:
    def symbols(self, signal, delay, interval, Ns):
        delay = int(np.round(delay))

        interval = int(np.round(interval)) #symbol period

        idxs = delay + np.arange(Ns) * interval*2
        samples = signal[idxs]
        return samples, idxs

