"used to configure simulation"

from dataclasses import dataclass, field
import numpy as np


@dataclass
class CDMcfg():
    #Simulator 
    simtime: float 

    #FEATURES
    BEAMFORM: bool=False                #To combine recovered elements into one received beam
    CALIBRATE: bool=False               #Model an ideal matched-filter reciever
    INTERFERENCE: bool=False            #Inject interference as part of Signal Generator
    CARRIER: bool=True                  #Modulate baseband to high RF carrier frequency
    SPREADING_STAGE: str="BB"           #Location of the spreading [IF or BB]

    #Signal properties
    fc: float = 500e6                   #Carrier frequency
    fb: float = 30e6                    #Baseband symbol frequency
    alpha: float = 0.50                 #RRC roll off factor
    span: float = 10                    #RRC symbol span
    modType: str = "BPSK"               #Modulation scheme [BPSK or QPSK]

    #Receiver properties 
    fIF: float = 80e6                   #IF frequency
    lc: int = 32                        #Code length
    sps_chip: int = 2                   #Samples per chip >=2
    codefam: str = "WH"                 #Code family 
    no_elem: int = 2                    #No elements multiplexed
    degree: int = 1                     #Degree used by Gold Codes
    codeids: np.ndarray = field(        #Code index
        default_factory=lambda: 
        np.array([])
        )

    #Calculated parameters
    def __post_init__(self):
        #Signal properties 
        self.sps_symbol = 8                         #Samples per symbol in RRC upsampler            
        self.ts = 1/self.fb                         #Symbol period
        self.fs = self.sps_symbol * self.fb         #Sampling frequency after RRC upsampler
        self.b = self.fb * (1 + self.alpha)         #RRC limited bandwidth

        #Receiver properties 
        if len(self.codeids) == 0:
            self.codeids = np.arange(0,self.lc)     #If no code ids given, go from to 0-Lc
