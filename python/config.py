"used to configure simulation"

from dataclasses import dataclass, field
import numpy as np

@dataclass
class CXBF():
    rb: float
    lc: int 
    sps_chip: int
    alpha: float
    simtime: float 
    modType: str = "BPSK"
    codefam: str = "WH"
    no_elem: int = 1 
    degree: int = 1
    span: int = 8
    codeids: list = field(default_factory=lambda: [1, 2])

    def __post_init__(self):
        #calculate and assign derived parameter
        self.sps_symbol = 10 #becomes sps_chip*(1+alpha)*Lc
        self.ts = 1/self.rb 
        self.fs = self.sps_symbol / self.ts 
        self.b = self.rb * (1 + self.alpha)
        self.rc = self.rb * self.lc