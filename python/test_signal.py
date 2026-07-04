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

def test_symbol_period_count(base_cfg):
    sig = DiscreteSignal(base_cfg)
    peaks, _ = find_peaks(np.abs(sig.signalts), height=0.8)
    pk_spacing = np.diff(peaks) 

    #check count
    simsymbols = base_cfg.simtime // base_cfg.no_symbols
    assert len(peaks)  == pytest.approx(simsymbols, abs=1)
    #check periods 
    assert np.allclose(pk_spacing, base_cfg.symbsol_period, atol=5)


def test_filtered_bw(base_cfg):
    sig = DiscreteSignal(base_cfg)
    fs = 1
    f, Pxx = welch(sig.signalts, fs, return_onesided=False)
    
    f = np.fft.fftshift(f)
    Pxx = np.fft.fftshift(Pxx)

    Pxx_norm = Pxx / np.sum(Pxx)
    cumsum = np.cumsum(Pxx_norm)
    low = f[np.searchsorted(cumsum, 0.005)]
    high = f[np.searchsorted(cumsum, 0.995)]
    bw = high - low

    alpha = 0.25
    Rs = 1/base_cfg.symbol_period*2
    idealbw = Rs*(1+alpha)
    
    assert bw == pytest.approx(idealbw, abs=0.05)