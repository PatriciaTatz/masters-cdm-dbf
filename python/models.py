from dataclasses import dataclass
import numpy as np


@dataclass
class DecorrOutput:
    signals: list[np.ndarray]
    time: np.ndarray

@dataclass
class SpreadOutput:
    signals: list[np.ndarray]
    time: np.ndarray
    codes: list[np.ndarray]
    fs: float

@dataclass
class Specan:
    f: np.ndarray
    Pxx_dB: np.ndarray
    bw: float
