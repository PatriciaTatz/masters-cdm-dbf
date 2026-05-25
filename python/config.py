"used to configure software"

from dataclasses import dataclass

@dataclass
class dbfCfg:
    Lc: int
    code: str
    no_arrays: int
    degree: int = 1


@dataclass
class signalCfg:
    Ts:int
    modType: str
    Ns: int
