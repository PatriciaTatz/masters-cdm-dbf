"""
Simulate effects of channel
- AWGN
- Doppler shift 
"""
import commpy.channels as ch
from math import pi 
import random 
import numpy as np  


class DiscreteSignal:
    #Method to generate discrete signal on the phase array 

    def __init__(self, sigconfig, channel, SNR):
        self.conf = sigconfig
        #generate signal/s
        self.symbols()
        if channel=="withNoise":
            self.symbols = self.add_awgn(self.symbols, SNR)
        else:
            pass
        self.create_arraysignals()
        

    def symbols(self):
        #modType: requires 
        if self.conf.modType == "BPSK":
            values = [1, -1]
        elif self.conf.modType == "QPSK":
            values = [pi/4, 3*pi/4, 5*pi/4, 7*pi/4]

        self.symbols = np.random.choice(values, self.conf.Ls)

    def create_arraysignals(self):
        timedelay = 1 #time delay between array elements on antenna 
        indexshift = int(timedelay/self.conf.Ts)

        self.arrSignals = np.empty((0,len(self.symbols)))

        for i in range(self.conf.no_arrays):
            nSignal =  np.roll(self.symbols, shift=i*indexshift)
            nSignal[:i*indexshift] = 0
            self.arrSignals = np.vstack((self.arrSignals, nSignal))

    def add_awgn(self, signal, SNR):
        return ch.awgn(signal,SNR)
    


