"""
RF components
- Code generator (WH, Gold, and ZC) 
- Spreader
- Despreader
- Combiner
- Filters
- Matched Filter 
"""

from scipy.signal import max_len_seq
import numpy as np  
from enum import Enum 
import commpy.filters as filter
from commpy.sequences import pnsequence

import komm as km
import matplotlib.pyplot as plt 
from math import log2

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

        if self.config.code == "Gold":
            m = round(log2(self.config.Lc + 1))
            N_valid = 2**m - 1

            #pre-derived poly masks required by pn sequence
            gold_masks = {
                        3: {
                            "mask1": "101",   # x^3 + x + 1
                            "mask2": "110",   # x^3 + x^2 + 1
                        },
                        5: {
                            "mask1": "10100",      # x^5 + x^2 + 1
                            "mask2": "11110",      # x^5 + x^4 + x^3 + x^2 + 1
                        },
                        7: {
                            "mask1": "1000100",    # x^7 + x^3 + 1
                            "mask2": "1001110",    # x^7 + x^6 + x^5 + x^2 + 1
                        },
                        10: {
                            "mask1": "1000000100", # x^10 + x^3 + 1
                        }}
            if m not in gold_masks:
                raise ValueError(f"No standard polynomial pair stored for m={m}")
            
            s1 = pnsequence(m, [1]*m, gold_masks[m]["mask1"], N_valid)
            s2 = pnsequence(m, [1]*m, gold_masks[m]["mask1"], N_valid)

            # Step 5: generate all Gold codes
            code = np.logical_xor(s1, np.roll(s2, codeindex+1)).astype(int)
    
            return code

        else: 
            raise Exception("Unrecognised code family! Choose from [WH, ]")


class Multiplexer:
    #Method to spread the signal using a code sequence

    def encoder(self, code, signal):
        """
        direct time multiplication
        """
        ss = np.array([]) #spread symbols
        codets = np.resize(code, len(signal))
        ss = signalts*codets

        return ss     

    def decoder(self, Lc, Ts, code, spreadsum):   
        #Depends on Decorrelator being run first. ASK ALIX
        despread_symbols = [] 
        for i in range(len(spreadsum)):
            #restarts the code indexing at the end of sequence 
            despread_symbols.append((spreadsum[i]*code[i%Lc]/ Lc))

        #matched filter / correlator too! 
        sps = Lc #samples per symbol 
        Fs = sps/Ts #sample per second, take Lc samples per second 
        alpha = 0.25 #does this need to match?
        span = 8 #consider 8 symbols, i believe this is normal number 
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
    def txrrcfilter(self, symbols, Ts, ups):
        """
        rrc filter for transmit
        independent of Lc / Rx settings as its suppose to simulate real rxd signals
        what it does:
        - upsamples
        - applies filter 
        - is called by mysignal.py 
        """
        Fs = ups/Ts #Hz; sampling rate, sample per second 
        alpha = 0.5 #roll off factor 
        span =  16 #dictates how far a symbols influence spreads across time
        Ntaps = span*ups + 1  #must be odd, total length of the filter 

        upsampled = np.zeros(len(symbols)*ups) 
        upsampled[::ups] = symbols
  
        #apply filter
        dt_filter, h_rrc = filter.rrcosfilter(Ntaps, alpha, Ts, Fs)
        txsignal = np.convolve(upsampled, h_rrc, mode='full')
        group_delay = int((Ntaps-1)/2)

        return txsignal[group_delay:-group_delay], upsampled


