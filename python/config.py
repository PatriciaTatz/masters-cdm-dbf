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
    INTERFERENCE: bool=True             #Inject interference as part of Signal Generator
    CARRIER: bool=True                  #Modulate baseband to high RF carrier frequency
    SPREADING_STAGE: str="BB"           #Location of the spreading [IF or BB]
    DOPPLER: bool=False

    #Signal properties
    fc: float = 1e9                     #Carrier frequency
    fb: float = 10e6                    #Baseband symbol frequency
    elevation: float = 90               #Elevation in degrees
    alpha: float = 0.25                 #RRC roll off factor
    beta: float = 0                     #Doppler ratio 
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
        self.offset = 5e6
        self.fIF = (self.b) + self.offset
        self.fd = self.fc*self.beta                  #Doppler shift 
  
        #Receiver properties 
        if len(self.codeids) == 0:
            self.codeids = np.arange(0,self.lc)     #If no code ids given, go from to 0-Lc
