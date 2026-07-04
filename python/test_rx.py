import pytest
import numpy as np
import matplotlib.pyplot as plt 
from scipy.signal import find_peaks, welch, peak_widths
from signals import DiscreteSignal
from config import CXBF

#test for signal 

@pytest.fixture
def base_cfg():
    "Checks time series of the signal for a full block of message. No repeats"
    return CXBF(
        symbol_period= 10,
        no_symbols= 10, 
        simtime=200+5, 
        modType="BPSK",
    )

