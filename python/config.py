"used to configure software"

from dataclasses import dataclass

@dataclass
class cdmCfg:
    Lc: int
    code: str


@dataclass
class signalCfg:
    Ts:int
    modType: str
    Ls: int
    no_arrays:int 
