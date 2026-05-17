"used to configure software"

class cdmCfg:
    def __init__(self, Lc, code):
        self.Lc = Lc 
        self.code = code

class signalCfg:
    def __init__(self,Ts, modType, Ls, no_arrays):
        self.Ts = Ts
        self.modType = modType
        self.Ls = Ls
        self.no_arrays = no_arrays
