"used to configure software"

from dataclasses import dataclass, field
import numpy as np
@dataclass
class dbfCfg:
    Lc: int 
    code: str
    no_arrays: int = 1 
    degree: int = 1
    codeids: list = field(default_factory=lambda: [1])


@dataclass
class signalCfg:
    Ts:int
    modType: str
    Ns: int
