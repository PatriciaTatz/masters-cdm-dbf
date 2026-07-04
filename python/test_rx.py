import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.signal import find_peaks, peak_widths, welch

from config import CXBF
from signals import DiscreteSignal

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

