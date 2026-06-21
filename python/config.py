"used to configure simulation"

from dataclasses import dataclass, field
import numpy as np

@dataclass
class CXBF():
    Ts: float
    sps: int 
    simtime: float
    modType: str
    Lc: int = 1
    SF: int = 1
    codefam: str = "WH"
    no_arrays: int = 1 
    degree: int = 1
    codeids: list = field(default_factory=lambda: [1, 2])
