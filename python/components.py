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
            return WHcode
        else: 
            raise Exception("Unrecognised code family! Choose from [WH, ]")


class Multiplexer:
    #Method to spread the signal using a code sequence
    def __init__(self):
        pass

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

    def decoder(self, Lc, Ts, code, symbols):   
        #This is a two part method, a despread followed by a matched filter. 
        #A: despread and B: decorrelate via matched filtering. 

        #matched filter settings
        symbols_span = 4 #Used to calculate Nspan , no of symbols considered in a window
        self.Nspan = Lc*symbols_span + 1#Span of symbols to consider per filter window
        self.Lc = Lc
        Fs = Lc #Take Lc samples per symbol (sampling rate)
        alpha = 0.25

        dt_filter, h_rrc = filter.rrcosfilter(self.Nspan, alpha, Ts, Fs)

        #loop does both part A and part B 
        despread_symbols = [] 
        decorr_symbols = [] 
        for i in range(len(symbols)):
            j = i%Lc #restarts the code indexing at the end of sequence 
            despread_symbols.append(symbols[i]*code[j])
            
        decorr_symbols = np.convolve(despread_symbols, h_rrc, mode='same')
        #mode=same accounts for the filter delay 

        #decimate the signal by Lc 
        return decorr_symbols

    def downsample(self, signal):
        decision_symbols = signal[::self.Lc] #start:stop:step, 
        return decision_symbols




