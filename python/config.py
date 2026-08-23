"used to configure simulation"

from dataclasses import dataclass, field
import numpy as np


@dataclass
class CDMcfg():
    simtime: float 
    CALIBRATE: bool=False
    f_lo: float = 1e6
    rb: float = 6.25e6
    lc: int = 32
    sps_chip: int = 2 
    alpha: float = 0.5
    modType: str = "BPSK"
    codefam: str = "WH"
    no_elem: int = 1 
    degree: int = 1
    span: int = 10
    codeids: np.ndarray = field(default_factory=lambda: np.array([]))
    def __post_init__(self):
        #calculate and assign derived parameter
        self.sps_symbol = 10 #becomes sps_chip*(1+alpha)*Lc
        self.ts = 1/self.rb 
        self.fs = self.sps_symbol / self.ts 
        self.b = self.rb * (1 + self.alpha)
        self.rc = self.rb * self.lc
        if len(self.codeids) == 0:
            self.codeids = np.arange(0,self.lc)
